import numpy as np
import pinocchio as pin
import crocoddyl
import hppfcl
import mim_solvers

from pinocchio.robot_wrapper import RobotWrapper

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
        
        return ok, xs, us
    
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
        
        pin.computeCollisions(self.model,self.data,self.collision_model,self.collision_data,q,False)

        collision_results = []
    
        # Print the status of collision for all collision pairs
        for k in range(len(self.collision_model.collisionPairs)):
            cr = self.collision_data.collisionResults[k]
            cp = self.collision_model.collisionPairs[k]
            
            if cr.isCollision():
                collision_info = (
                    self.collision_model.geometryObjects[cp.first].name,
                    self.collision_model.geometryObjects[cp.second].name
                    )
                
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