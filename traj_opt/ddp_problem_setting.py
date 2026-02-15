import numpy as np
import pinocchio as pin
import crocoddyl
import hppfcl
# import mim_solvers

from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

class CollisionStopException(Exception):
    """Custom exception to halt the solver upon collision."""
    pass

class CollisionStopCallback(crocoddyl.CallbackAbstract):
    def __init__(self, snake_ddp_instance, safe_store):
        crocoddyl.CallbackAbstract.__init__(self)
        self.snake = snake_ddp_instance
        self.store = safe_store  # Mutable dictionary to cache safe trajectories

    def __call__(self, solver):
        # 1. Check for collisions in the current solver candidate
        collision_detected = False
        
        # We iterate through the trajectory. 
        # (Optimization: You could skip frames or check only specific links if slow)
        for x in solver.xs:
            q = x[:self.snake.nq]
            
            # Update geometry
            pin.updateGeometryPlacements(
                self.snake.model, 
                self.snake.data,
                self.snake.collision_model, 
                self.snake.collision_data, 
                q
            )
            
            # Check collision (stops at first contact for speed)
            if pin.computeCollisions(self.snake.collision_model, self.snake.collision_data, True):
                collision_detected = True
                break

        # 2. Logic: Save if safe, Stop if colliding
        if collision_detected:
            # Do NOT update the store. Raise exception to stop solver.
            raise CollisionStopException("Optimization halted: Trajectory hit an obstacle.")
        else:
            # Trajectory is safe. Cache it as the new "best safe".
            # We must copy the data to avoid reference issues as solver updates.
            self.store['xs'] = [x.copy() for x in solver.xs]
            self.store['us'] = [u.copy() for u in solver.us]
            self.store['cost'] = solver.cost
class SnakeReachingDDP():
    """
    Crocoddyl problem factory for 
    """

    def __init__(
        self,
        urdf_model_path=None,
        srdf_model_path=None,
        package_dirs=None,
        ee_frame_name=None,
        link_frame_names=None,
        is_gravity_on=False,
        base_placement=None,
        ):
        
        if urdf_model_path is None:
            raise ValueError("urdf_model_path must be provided.")
        
        self.urdf_model_path = urdf_model_path
        self.srdf_model_path = srdf_model_path
        
        # Robot
        if package_dirs is None:
            self.robot = RobotWrapper.BuildFromURDF(
                urdf_model_path,
            )
        else:
            self.robot = RobotWrapper.BuildFromURDF(
                urdf_model_path,
                package_dirs = package_dirs
            )
        
        # Gravity
        if not is_gravity_on:
            self.robot.model.gravity.linear[:] = 0.0

        # ee tracking mod will be overrided by link tracking mod
        if ee_frame_name is not None:
            self.ee_frame_id = self.robot.model.getFrameId(ee_frame_name)
            self.tracking_mod = "ee_pose"
        elif link_frame_names is not None:
            self.link_frame_ids = [self.robot.model.getFrameId(name) for name in link_frame_names]
            self.tracking_mod = "link_poses"
        else:
            self.tracking_mod = "state"

        self.model = self.robot.model
        self.data = self.robot.data
        
        self.visual_model = self.robot.visual_model
        self.collision_model = self.robot.collision_model

        if base_placement is not None:
            # 1. Apply transform to the Kinematic Chain (starts at first moving joint)
            first_moving_joint = "reu_joint_out2in_0"
            if self.robot.model.existJointName(first_moving_joint):
                joint_id = self.robot.model.getJointId(first_moving_joint)
                
                # Get current placement (includes any URDF offsets)
                current_joint_placement = self.robot.model.jointPlacements[joint_id]
                
                # Apply the base transform
                self.robot.model.jointPlacements[joint_id] = base_placement * current_joint_placement
                
                # 2. Apply transform to the Base Link Geometry (attached to Universe/Joint 0)
                for geom_model in [self.visual_model, self.collision_model]:
                    for geom in geom_model.geometryObjects:
                        if geom.parentJoint == 0:  # 0 is the Universe/Root joint
                            geom.placement = base_placement * geom.placement
                
                for frame in self.model.frames:
                    # If the frame is attached to the Universe (Joint 0), it needs to move.
                    if frame.parent == 0: 
                        frame.placement = base_placement * frame.placement
            else:
                print(f"Warning: Joint {first_moving_joint} not found. Base tf assumed to be identity.")
        
        self._setCollisionPairs()

        # State / actuation
        self.state = crocoddyl.StateMultibody(self.model)
        
        self.actuation = crocoddyl.ActuationModelFull(self.state)
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.actuation.nu        
        
    # =================================================
    # DDP Methods
    # =================================================
    def CreateProblem(
        self,
        dt = 0.025,
        T = 1,
        weights=None,
        x_lb=None,
        x_ub=None,
        u_lb=None,
        u_ub=None,
    ):        
        """
        Create a shooting problem for EE pose tracking.

        Parameters
        ----------
        dt : float
            Integration time step
        T : float
            Total time
        weights : dict (optional)
            Cost weights
        """
        
        N = int(T / dt)

        if weights is None:
            weights = {
                # State regularization
                "x_reg": np.array([1e-2]*self.state.nx),
                # Control regularization 
                "u_reg": np.array([1e-2]*self.nu),

                # terminal
                "x_reg_terminal": 1e-2,
                "u_reg_terminal": 1e-2,
            }
            
        # -------------------------
        # Cost models
        # -------------------------
        running_cost = crocoddyl.CostModelSum(self.state, self.nu)
        terminal_cost = crocoddyl.CostModelSum(self.state, self.nu)
        
        running_constraint = crocoddyl.ConstraintModelManager(
            self.state, self.nu
        )
        terminal_constraint = crocoddyl.ConstraintModelManager(
            self.state, self.nu
        )       
                  
        # -------------------------
        # State regularization
        # -------------------------        
        xf = np.zeros(self.state.nx)
            
        self.x_res = crocoddyl.ResidualModelState(self.state, xf, self.nu)
        w_x_reg = weights["x_reg"]
        
        if self.tracking_mod == "ee_pose" or self.tracking_mod == "link_poses":
            w_x_reg[:self.state.nq] = 0.0  # disable vel reg if tracking ee pose
        
        x_reg, x_reg_scale = self._make_cost(self.x_res, w_x_reg)
        
        # add to running and terminal costs
        running_cost.addCost("x_reg", x_reg, x_reg_scale, True)
        terminal_cost.addCost("x_reg_term", x_reg, weights["x_reg_terminal"],True)
        
        # -------------------------
        # Pose regularization
        # -------------------------
        if self.tracking_mod == "ee_pose":
            ee_f = pin.SE3.Identity()
            self.ee_res = crocoddyl.ResidualModelFramePlacement(
                self.state,
                self.ee_frame_id,
                ee_f,
                self.nu
            )
            w_ee_reg = weights["ee_reg"]
            ee_reg, ee_reg_scale = self._make_cost(self.ee_res, w_ee_reg)
            
            # add to running and terminal costs
            running_cost.addCost("ee_reg", ee_reg, ee_reg_scale, True)
            terminal_cost.addCost("ee_reg_term", ee_reg, weights["ee_reg_terminal"], True)
            
        elif self.tracking_mod == "link_poses":            
            self.link_poses_res = [None] * len(self.link_frame_ids)
            for i in range(len(self.link_frame_ids)):
                self.link_poses_res[i] = crocoddyl.ResidualModelFramePlacement(
                    self.state,
                    self.link_frame_ids[i],
                    pin.SE3.Identity(),
                    self.nu
                )
                w_pose_reg = weights["link_poses_reg"][i]
                pose_reg, pose_reg_scale = self._make_cost(self.link_poses_res[i], w_pose_reg)
            
                # add to running and terminal costs
                running_cost.addCost("link_poses_reg" + str(i), pose_reg, pose_reg_scale, True)
                terminal_cost.addCost("link_poses_reg_term" + str(i), pose_reg, weights["link_poses_reg_terminal"][i], True)
            
            
        # -------------------------
        # Control regularization
        # -------------------------
        self.u_res = crocoddyl.ResidualModelControl(self.state, self.nu)
        u_reg, u_reg_scale = self._make_cost(self.u_res, weights["u_reg"])        
            
        # add to running and terminal costs     
        running_cost.addCost("u_reg", u_reg, u_reg_scale, True)  
        terminal_cost.addCost("u_reg_term", u_reg, weights["u_reg_terminal"],True)
        
        # -------------------------
        # State limits
        # -------------------------
        if x_lb is not None and x_ub is not None:
            x_bound = crocoddyl.ActivationBounds(x_lb, x_ub, 1.0)
            x_bound_ref = np.zeros(self.state.nx)
            x_bound_ref[6] = 1
            x_bound_res = crocoddyl.ResidualModelState(self.state, x_bound_ref, self.nu)
            x_bound_activation = crocoddyl.ActivationModelQuadraticBarrier(x_bound)
            x_bound_cost = crocoddyl.CostModelResidual(self.state, x_bound_activation, x_bound_res)
            
            running_cost.addCost("x_bounds", x_bound_cost, 1e3, True)
            terminal_cost.addCost("x_bounds_term", x_bound_cost, 1e3, True)
            
            # x_bound_constraint = crocoddyl.ConstraintModelResidual(
            #     self.state,
            #     x_bound_res,
            #     x_lb,
            #     x_ub,
            #     True
            # )
            
            # running_constraint.addConstraint("x_bounds", x_bound_constraint)
            # terminal_constraint.addConstraint("x_bounds_term", x_bound_constraint)
            
        # -------------------------
        # Action models
        # -------------------------
        running_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, running_cost, running_constraint
        )
        terminal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, terminal_cost, terminal_constraint
        )

        control = crocoddyl.ControlParametrizationModelPolyZero(self.nu)

        running_model = crocoddyl.IntegratedActionModelEuler(
            running_diff, control, dt
        )
        terminal_model = crocoddyl.IntegratedActionModelEuler(
            terminal_diff, control, 0.0
        )
        
        if u_lb is not None and u_ub is not None:
            running_model.u_lb = u_lb
            running_model.u_ub = u_ub
            terminal_model.u_lb = u_lb
            terminal_model.u_ub = u_ub
           
        x0 = np.zeros(self.state.nx)
        self.problem = crocoddyl.ShootingProblem(
            x0, [running_model] * N, terminal_model
        )        
        
        self.solver = crocoddyl.SolverBoxFDDP(self.problem)     
        
    def _make_cost(self, residual, w):
        """
        w:
        - np.ndarray : element-wise activation weight
        - float      : scalar cost weight
        """
        if isinstance(w, np.ndarray):
            activation = crocoddyl.ActivationModelWeightedQuad(w)
            cost = crocoddyl.CostModelResidual(self.state, activation, residual)
            scale = 1.0
        else:
            cost = crocoddyl.CostModelResidual(self.state, residual)
            scale = float(w)
        return cost, scale

    def Solve(        
        self,
        x0,
        xf,
        maxiter = None,
    ):   
        
        self.problem.x0 = x0
        
        if self.tracking_mod == "state":
            self.x_res.reference = xf
            
            # --------------------------------------------
            # Warm-start
            # --------------------------------------------            
            xs_init = [
                (1 - alpha) * x0 + alpha * xf
                for alpha in np.linspace(0.0, 1.0, self.problem.T + 1)
            ]
        elif self.tracking_mod == "ee_pose":
            self.ee_res.reference = xf # this should be a pin.SE3
            xs_init = [x0] * (self.problem.T + 1)
        elif self.tracking_mod == "link_poses":
            for i in range(len(self.link_frame_ids)):
                self.link_poses_res[i].reference = xf[i] # this should be a pin.SE3 list
            xs_init = [x0] * (self.problem.T + 1)


        us_init = list(self.problem.quasiStatic([x0] * self.problem.T))
        
        
        if maxiter is None:
            ok = self.solver.solve(xs_init, us_init)
        else:
            ok = self.solver.solve(xs_init, us_init, maxiter=maxiter)

        # --------------------------------------------
        # Receding horizon
        # --------------------------------------------
        xs = np.array(self.solver.xs)
        us = np.array(self.solver.us)
        
        return ok, xs, us, self.solver.cost
    
    # =================================================
    # Collision methods
    # =================================================    
    def _setCollisionPairs(self):
        # Collision model
        self.collision_model.addAllCollisionPairs()
        
        if self.srdf_model_path is not None:
            pin.removeCollisionPairs(self.model, self.collision_model, self.srdf_model_path)           
                        
            # Load reference configuration
            pin.loadReferenceConfigurations(self.model, self.srdf_model_path)            
        
        self.collision_data = self.collision_model.createData()        
        
    def AddWallSet(
        self,
        position: pin.SE3,
        size=np.array([0.1, 1.0, 1.0]),   # thickness, width, height
        gap=0.5,
        color=np.array([0.7, 0.7, 0.7, 1.0]),
    ):
        """
        Add a pair of facing walls (left / right) to the world.

        position : SE3
            Center pose between the two walls
        """
        if not hasattr(self, "wall_sets"):
            self.wall_sets = []

        wall_geom = hppfcl.Box(*size)

        # offset along ±y
        y_offset = gap / 2 + size[1] / 2

        offsets = {
            "left":  pin.SE3(np.eye(3), np.array([0.0,  y_offset, 0.0])),
            "right": pin.SE3(np.eye(3), np.array([0.0, -y_offset, 0.0])),
        }

        wall_ids = {"visual": {}, "collision": {}}

        for side, offset in offsets.items():
            T_wall = position * offset

            # --- Visual ---
            wall_vis = pin.GeometryObject(
                f"wall_{len(self.wall_sets)}_{side}",
                0,
                0,
                wall_geom,
                T_wall,
            )
            wall_vis.meshColor = color
            self.visual_model.addGeometryObject(wall_vis)
            wall_ids["visual"][side] = len(self.visual_model.geometryObjects) - 1

            # --- Collision ---
            wall_col = pin.GeometryObject(
                f"wall_{len(self.wall_sets)}_{side}",
                0,
                0,
                wall_geom,
                T_wall,
            )
            self.collision_model.addGeometryObject(wall_col)
            wall_ids["collision"][side] = len(self.collision_model.geometryObjects) - 1

        self.wall_sets.append({
            "ids": wall_ids,
            "position": position,
            "offsets": offsets,
        })

        self._setCollisionPairs()
            


    def collisionCheck(self, q):
        # 1. Update geometry placements for the new configuration q
        pin.updateGeometryPlacements(
            self.model, 
            self.data,
            self.collision_model, 
            self.collision_data, 
            q
        )

        # 2. Compute all collisions (do not stop at first, so we can filter)
        pin.computeCollisions(
            self.collision_model, 
            self.collision_data, 
            False
        )

        collision_results = []
        
        # 3. Define the specific pairs to ignore
        # Using a set of sets for order-independent checking
        ignored_pairs = [
            {'reu_module_out_0_0', 'reu_module_body_1_0'}
        ]
    
        # 4. Filter results
        for k in range(len(self.collision_model.collisionPairs)):
            cr = self.collision_data.collisionResults[k]
            cp = self.collision_model.collisionPairs[k]
            
            if cr.isCollision():
                name1 = self.collision_model.geometryObjects[cp.first].name
                name2 = self.collision_model.geometryObjects[cp.second].name
                
                # Check if this pair is in our ignore list
                current_pair = {name1, name2}
                
                if current_pair not in ignored_pairs:
                    collision_info = (name1, name2)
                    collision_results.append(collision_info)
            
        n_collisions = len(collision_results)
        
        return collision_results, n_collisions
    
    def collisionDistances(self, q):
        pin.updateGeometryPlacements(
            self.model, self.data,
            self.collision_model, self.collision_data, q
        )
        pin.computeDistances(self.collision_model, self.collision_data)

        n = len(self.collision_model.collisionPairs)
        distances = np.zeros(n)
        for k, pair in enumerate(self.collision_model.collisionPairs):
            d = self.collision_data.distanceResults[k].min_distance
            distances[k] = d
        return distances
    
    def InitDisplay(self):
        self.robot.setVisualizer(MeshcatVisualizer())
        self.robot.initViewer(open=True)
        self.robot.loadViewerModel()

    def ShowSnakeConfiguration(self, q):
        self.robot.display(q)

    def RefineTrajectory(self, xs_init, us_init=None, maxiter=50):
        """
        Refines a trajectory. Stores the last valid (collision-free) iteration.
        If a collision is found, stops and returns the LAST SAFE trajectory.
        """
        # 1. Prepare Initial State
        if us_init is None:
            us_init = list(self.problem.quasiStatic(list(xs_init[:-1])))
        
        # 2. Check if the SEED itself is colliding
        # If the start is bad, we can't optimize "until" collision.
        # We reuse the logic from the callback manually for the seed.
        is_seed_colliding = False
        for x in xs_init:
            q = x[:self.nq]
            pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, q)
            if pin.computeCollisions(self.collision_model, self.collision_data, True):
                is_seed_colliding = True
                break
        
        if is_seed_colliding:
            print("Warning: Initial seed trajectory already in collision. Returning seed.")
            return False, np.array(xs_init), np.array(us_init), float('inf')

        # 3. Setup Safe Storage
        # Initialize with the seed (which we now know is safe)
        safe_store = {
            'xs': [x.copy() for x in xs_init],
            'us': [u.copy() for u in us_init],
            'cost': self.problem.calc(list(xs_init), list(us_init)) # Initial cost
        }

        # 4. Setup Callback
        stop_callback = CollisionStopCallback(self, safe_store)
        self.solver.setCallbacks([stop_callback])
        
        self.problem.x0 = xs_init[0]
        
        # 5. Solve
        converged = False
        try:
            self.solver.solve(list(xs_init), list(us_init), maxiter)
            converged = True # Finished maxiter without collision
            # If we finished safely, the callback updated safe_store with the final result
        except CollisionStopException:
            print("Collision detected during optimization. Reverting to last safe trajectory.")
            converged = False
        
        # 6. Return the CACHED safe trajectory, not the solver's current (colliding) state
        return converged, np.array(safe_store['xs']), np.array(safe_store['us']), safe_store['cost']