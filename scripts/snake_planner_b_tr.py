import numpy as np
import heapq
import tqdm
import time
import math
import pinocchio as pin

import pickle
from copy import copy
import scipy.io
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


import os
import sys

import time
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from traj_opt.snake_functions import calculate_h, get_ee_problem, get_problem, do_traj_opt, get_ee_pose, add_square_tunnel, get_tunnel_wall_aabbs, get_curve_problem, lift_and_do_traj_opt


class Node:
    """Internal helper class for search nodes."""
    def __init__(self, data, node_id):
        self.data = data  # Latent state: [control_x, control_y, control_z, end_x, end_y, end_z]
        self.parent = None
        self.children = []
        self.H = None     # Joint configuration (q)
        self.H_traj = []  # Trajectory from root to here
        self.g = 0        # Cost from start
        self.h = 0        # Heuristic cost
        self.f = 0        # Total cost
        self.node_id = node_id
        self.b_curve_points=[]

    @property
    def priority(self):
        return self.h  # Priority based on heuristic (Greedy Best-First)


class SnakePlanner:
    """
    A planner for a snake robot that searches in a latent Bezier-curve space
    and maps to joint angles using Inverse Kinematics, while verifying 
    dynamic feasibility (torque limits) and collisions.
    """


    def __init__(self, 
                 num_links=16, 
                 link_length=0.0506,
                #  link_length=0.05105, 
                 link_mass=0.250,
                 base_loc=None,
                 torque_limit=16,
                 goal_proximity=0.04,
                 cost_grid_size=0.03,
                 bounds=0.9,
                 obs = None):
        
        # Robot Physical Parameters
        self.num_links = num_links
        self.link_length = link_length
        self.snake_length = num_links * link_length
        self.link_mass = link_mass
        self.base_loc = base_loc if base_loc is not None else np.array([0, 0, 0])
        self.base_tf = np.eye(4)  # Assuming base rotation is identity for now

        self.ddp = get_problem()
        self.ddp_ee = get_ee_problem()
        self.ddp_bb = get_curve_problem()
        
        # Planning Constraints & Thresholds
        self.jammed_links = 1  # Number of links at base assumed "jammed" or highly constrained
        self.goal_proximity = goal_proximity
        self.bounds_limit = bounds
        self.length_thresh = 1.05
        
        # Search Parameters
        self.delta_t = 0.03
        self.delta_ts = 0.01
        self.delta_tl = 0.1
        self.delta_cl = 0.15
        self.delta_cs = 0.05
        self.interp_steps = 10
        self.cost_grid_size = cost_grid_size
        self.dt = 0.1  # Time step for dynamics check

        # Internal Counter
        self._node_counter = 0

        self.first_locs = np.load("first_four_locs_shifted.npy")
        self.first_locs = self.first_locs[:3,:]
    


        self.default_wall_thick = 0.04  # 2 cm thick walls
        self.default_gap_len = 0.3
        self.default_gap_height = 0.12
        self.default_gap_width = 0.12
        walls =[]
        if obs is not None:
            for ob in obs:
                _, walls_sol = get_tunnel_wall_aabbs(center_pos=ob[0], 
                                                           orientation_axis=ob[1], 
                                                           width=self.default_gap_width, 
                                                           height=self.default_gap_height, 
                                                           length=self.default_gap_len, 
                                                           wall_thick=self.default_wall_thick)
                # walls.extend(walls_sol)
                walls = walls+ walls_sol
                center, direction = ob
                add_square_tunnel(self.ddp,
                                  center, 
                                  direction,
                                  width=self.default_gap_width,
                                  height=self.default_gap_height,
                                  length=self.default_gap_len,
                                  wall_thick=self.default_wall_thick)
        back = np.array([20, -45,  10, 50,  -35,  30])/100

        self.obstacles = walls
        self.obstacles.append(back)
    # =========================================================================
    #    Helper Methods (Dynamics, Kinematics, Collision, Heuristics)
    # =========================================================================

    def _get_id(self):
        self._node_counter += 1
        return self._node_counter
    def _get_neighbors(self, node):
        """Generates neighbors by perturbing control points and end points."""
        neighbors = []
        import itertools
        
        # 1. Axis-Aligned Moves (Large steps)
        # We keep these restricted to axes to prevent combinatorial explosion on large jumps
        axis_moves = [
            (range(3, 6), self.delta_tl), # Perturb End Point (Large)
            (range(3), self.delta_cl)    # Perturb Control Point (Large)
        ]

        for indices, delta in axis_moves:
            for dx in [-delta, delta]:
                for ind in indices:
                    new_data = copy(node.data)
                    new_data[ind] += dx
                    new_data = np.round(new_data, decimals=2)
                    
                    # Check arc length constraint
                    if self._check_arc_length(new_data[3:6], new_data[:3]):
                        continue
                        
                    # Check workspace bounds
                    if not self._check_bounds(new_data):
                        continue
                        
                    new_node = Node(new_data, self._get_id())
                    new_node.parent = node
                    neighbors.append(new_node)

        # 2. Cube/Corner Moves (Small/Fine steps)
        cube_moves = [
            (range(3, 6), self.delta_t), # Perturb End Point (Fine)
            (range(3, 6), self.delta_ts),     # Perturb Control Point (Fine)
            (range(3), self.delta_cs)     # Perturb Control Point (Coarse)
        ]

        for indices, delta in cube_moves:
            perturbations = itertools.product([-delta, 0, delta], repeat=3)
            
            for p in perturbations:
                # Skip the case where there is no movement (0, 0, 0)
                if all(x == 0 for x in p):
                    continue

                new_data = copy(node.data)
                # Apply the perturbation to the current subset of indices (e.g., x,y,z)
                valid_move = True
                for i, offset in enumerate(p):
                    new_data[indices[i]] += offset
                
                new_data = np.round(new_data, decimals=2)
                
                if self._check_arc_length(new_data[3:6], new_data[:3]):
                    continue
                
                if not self._check_bounds(new_data):
                    continue
                    
                new_node = Node(new_data, self._get_id())
                new_node.parent = node
                neighbors.append(new_node)
                
        return neighbors

    # def _get_neighbors(self, node):
    #     """Generates neighbors by perturbing control points and end points."""
    #     neighbors = []
        
    #     # Perturbations defined in original code
    #     moves = [
    #         (range(3, 6), self.delta_tl), # Perturb End Point
    #         (range(3, 6), self.delta_t),  # Fine Perturb End Point
    #         (range(3), self.delta_cl)     # Perturb Control Point
    #     ]

    #     for indices, delta in moves:
    #         for dx in [-delta, delta]:
    #             for ind in indices:
    #                 new_data = copy(node.data)
    #                 new_data[ind] += dx
    #                 new_data = np.round(new_data, decimals=2)
                    
    #                 # Check arc length constraint (snake shouldn't stretch too much)
    #                 if self._check_arc_length(new_data[3:6], new_data[:3]):
    #                     continue
                        
    #                 # Check workspace bounds
    #                 if not self._check_bounds(new_data):
    #                     continue
                        
    #                 new_node = Node(new_data, self._get_id())
    #                 new_node.parent = node
    #                 neighbors.append(new_node)
    #     return neighbors
    
    def _generate_b_curve_points(self, node):
        """
        Maps latent Bezier parameters to joint angles via IK.
        Now includes Warm Starting and Feasibility Checks.
        """
        link_locs = []
        control = node.data[:3]
        end = node.data[3:]
        
        for i in range(self.num_links + 1):
            s = i / self.num_links
            # Quadratic Bezier
            loc = (1-s)*(1-s)*self.base_loc + 2*(1-s)*s*control + s*s*end
            link_locs.append(loc)

        return link_locs


    def r_lift_to_high(self, node):
        """
        Maps latent Bezier parameters to joint angles via IK.
        Now includes Warm Starting and Feasibility Checks.
        """
        link_locs = []
        control = node.data[:3]
        end = node.data[3:]
        
        # Generate target points from Bezier curve
        for i in range(self.num_links + 1):
            s = i / self.num_links
            # Quadratic Bezier
            loc = (1-s)*(1-s)*self.base_loc + 2*(1-s)*s*control + s*s*end
            link_locs.append(loc)
            
        link_locs = np.array(link_locs).T

        initial_guess = node.parent.H[:self.ddp.nq] if node.parent is not None else None
        if initial_guess is not None:
            initial_guess = initial_guess[3:]
        
        # Solve IK
        angles, success = self.r_solve_ik(link_locs, initial_guess)
        
        if not success:
            return None
        
        state = np.zeros(self.ddp.nq + self.ddp.nv)
        state[3:self.ddp.nq] = angles
        state[self.ddp.nq:] = np.zeros(self.ddp.nv)
        
            
        return state

    def _lift_and_do_traj_opt(self,parent, node):

        q_init = parent.H[:self.ddp.nq] if parent is not None else np.zeros(self.ddp.nq)
        link_locs = self._generate_b_curve_points(node)
        true_link_locs = [self.first_locs[i] for i in range(self.first_locs.shape[0])]
        for i in range(len(link_locs)):
            true_link_locs.append(link_locs[i])
        link_locs = true_link_locs
        ok, traj, us, cost = lift_and_do_traj_opt(self.ddp_bb, q_init, link_locs,threshold= 0.15,goal = self.goal_pose)
        # ok, traj, us, cost = lift_and_do_traj_opt(self.ddp_bb, q_init, link_locs)
        ok_c = self._check_collision(traj)
        return ok, ok_c, traj, us, cost


    def r_solve_ik(self, target_pts, initial_q=None):
        """
        Optimizes joint angles to match target points.
        Includes Regularization to prevent joint flipping.
        """
        if initial_q is not None:
            q_guess = initial_q
            # Weight for keeping joints close to previous state
            lambda_reg = 0.05 
        else:
            q_guess = np.zeros(self.num_links)
            lambda_reg = 0.0  # No history to regularize against
            
        bounds = [(-np.pi/2, np.pi/2) for _ in range(self.num_links)]
        
        def objective(q):
            # 1. Primary Objective: Match the Bezier Curve positions
            P_world, _ = self.r_forward_kinematics(q)
            diff = P_world - target_pts
            fit_error = np.sum(diff**2)
            
            # 2. Regularization Term: Minimize joint movement from parent
            reg_term = 0.0
            if lambda_reg > 0:
                reg_term = lambda_reg * np.sum((q - q_guess)**2)
                
            return fit_error + reg_term

        result = minimize(
            objective, 
            q_guess, 
            method='SLSQP', 
            bounds=bounds,
            options={'ftol': 1e-4, 'disp': False}
        )
        
        # --- FEASIBILITY THRESHOLD ---
        # Check if the robot actually matched the curve.
        P_final, _ = self.r_forward_kinematics(result.x)
        final_error = np.sum((P_final.T - target_pts.T)**2)
        final_error = 0
        for i in range(P_final.T.shape[0]):
            final_error += np.linalg.norm(P_final.T[i] - target_pts.T[i])
        
        is_feasible = final_error < 0.05  # Adjustable threshold
        
        return result.x, is_feasible
    
    def r_forward_kinematics(self, q):
        """Computes positions of all joints and EE transform."""
        q = np.atleast_1d(q)
        
        T_current = np.eye(4)
        T_links = [np.eye(4)] # Include base

        for i in range(self.num_links):
            # Alternating joints: Even=Z (Yaw), Odd=Y (Pitch) - based on original code
            if i % 2 == 0:
                T_joint = self._Rz(q[i])
            else:
                T_joint = self._Ry(q[i])
                
            T_current = T_current @ T_joint @ self._Tx(self.link_length)
            T_links.append(T_current)

        # Apply Base Transform
        P_world = np.zeros((3, self.num_links + 1))
        T_ee_world = self.base_tf @ T_links[-1]
        
        for i in range(self.num_links + 1):
            T_world = self.base_tf @ T_links[i]
            P_world[:, i] = T_world[0:3, 3]

        return P_world, T_ee_world
    def _repair(self, node):
        """Recursively tries to connect to ancestors if immediate parent fails."""
        ancestor = node.parent
        ancestors=[]

        while ancestor is not None:
            ancestors.append(ancestor)
            ancestor = ancestor.parent
        # flip ancestors
        ancestors.reverse()
        # print(f"Attempting repair with {len(ancestors)} ancestors.")
        for a in ancestors:
            ok, traj,us,cost = self._do_traj_opt(a.H[:self.ddp_ee.nq], node.H[:self.ddp_ee.nq])
            if ok:
                node.g = a.g + cost
                node.H_traj = np.append(a.H_traj, traj, axis=0)
                return True, traj,us,cost
        return False, None, None, None

    def _is_point_in_obstacle(self, p):
        epsilon = 1e-4
        
        for obs in self.obstacles:
            x_min, x_max = min(obs[0], obs[3]), max(obs[0], obs[3])
            y_min, y_max = min(obs[1], obs[4]), max(obs[1], obs[4])
            z_min, z_max = min(obs[2], obs[5]), max(obs[2], obs[5])

            if (x_min - epsilon <= p[0] <= x_max + epsilon and 
                y_min - epsilon <= p[1] <= y_max + epsilon and 
                z_min - epsilon <= p[2] <= z_max + epsilon):
                return True
        return False
    def _generate_cost_field(self, start_point, bounds):
        """Generates a 3D grid cost map using Dijkstra from the goal."""
        def to_idx(pt):
            return tuple(np.floor(pt / self.cost_grid_size).astype(int))

        min_idx = to_idx(np.array([b[0] for b in bounds]))
        max_idx = to_idx(np.array([b[1] for b in bounds]))
        
        start_idx = to_idx(start_point)
        pq = [(0, start_idx)]
        steps_map = {start_idx: 0}
        
        neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        # Limit iterations for safety/speed
        max_voxels = 500000 
        count = 0
        
        while pq and count < max_voxels:
            steps, curr = heapq.heappop(pq)
            count += 1
            
            if steps > steps_map[curr]: continue

            for dx, dy, dz in neighbors:
                nxt = (curr[0]+dx, curr[1]+dy, curr[2]+dz)
                
                # Bounds check
                if not (min_idx[0] <= nxt[0] < max_idx[0] and
                        min_idx[1] <= nxt[1] < max_idx[1] and
                        min_idx[2] <= nxt[2] < max_idx[2]):
                    continue
                
                # Obstacle check
                center = (np.array(nxt) + 0.5) * self.cost_grid_size
                if self._is_point_in_obstacle(center):
                    continue
                    
                new_steps = steps + 1
                if nxt not in steps_map or new_steps < steps_map[nxt]:
                    steps_map[nxt] = new_steps
                    heapq.heappush(pq, (new_steps, nxt))
        print(f"Cost field generated with {len(steps_map)} voxels explored.",count)       
        return steps_map

    def _calculate_h(self, node):
        """Heuristic function: Lookup in cost map."""
        ee_pos = node.data[3:6]
        idx = tuple(np.floor(ee_pos / self.cost_grid_size).astype(int))
        steps = self.cost_map.get(idx, None)
        if steps is None:
            return 100000.0 # High cost if unreachable/out of bounds
        return steps * self.cost_grid_size

    def _do_traj_opt(self, q_init, q_f):
        # start_time = time.perf_counter()
        ok, xs, us, cost = do_traj_opt(self.ddp, q_init, q_f)
        if not ok:
            return False, None, None, None
        # end_time = time.perf_counter()
        # print(f"Trajectory optimization time: {end_time - start_time:.4f} seconds")
        # start_time = time.perf_counter()
        ok_c = self._check_collision(xs)
        # end_time = time.perf_counter()
        # print(f"Collision checking time: {end_time - start_time:.4f} seconds")
        return ok and ok_c, xs, us, cost
    

    def _calculate_ee_diff(self, node):
        ee_pose = get_ee_pose(self.ddp_ee, node.H[:self.ddp_ee.nq])
        return np.linalg.norm(ee_pose.translation - self.goal_pose.translation)
    
    def _check_collision(self, traj, stride=1):
        if not self._check_single_state(traj[0]): return False
        if not self._check_single_state(traj[-1]): return False
        for x in traj[stride:-1:stride]:
            if not self._check_single_state(x):
                return False
        return True
    
    
    def _check_single_state(self, x):
        q = x[:self.ddp.nq]
        _, n_collisions = self.ddp.collisionCheck(q)
        return n_collisions == 0
    

    def _check_arc_length(self, end, control):
        dist = 0
        loc0 = self.base_loc
        for i in range(1, self.num_links + 1):
            s = i / self.num_links
            loc = (1-s)*(1-s)*self.base_loc + 2*(1-s)*s*control + s*s*end
            dist += np.linalg.norm(loc - loc0)
            loc0 = loc
        return dist > (self.snake_length * self.length_thresh)

    def _check_bounds(self, data):
        return np.all(np.abs(data) <= self.bounds_limit)

    # Matrix Primitives
    @staticmethod
    def _Rz(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @staticmethod
    def _Ry(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def _Tx(d):
        return np.array([[1, 0, 0, d], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    
    def plan(self, x_init, goal_pose, max_iter=15000):

        self._node_counter = 0
        self.goal_pose = goal_pose
        print("Generating cost field...")
        bounds_tuple = ((-self.bounds_limit, self.bounds_limit), 
                        (-self.bounds_limit, self.bounds_limit), 
                        (-self.bounds_limit, self.bounds_limit))
        self.cost_map = self._generate_cost_field(self.goal_pose.translation, bounds_tuple)
        print("Cost field generated.")

        # 2. Initialize Root Node
        initial_latent = np.array([
            self.link_length * self.num_links / 2, 0, 0, 
            self.link_length * self.num_links, 0, 0
        ])
        
        root = Node(initial_latent, self._get_id())
        ok,ok_c,traj,us,cost = self._lift_and_do_traj_opt(root.parent,root)
        if not ok or not ok_c:
            print("Initial trajectory optimization failed. Check if the goal is reachable or if the initial guess is valid.")
            return None
        root.H = traj[-1]
        ee_tm = get_ee_pose(self.ddp_ee, root.H[:self.ddp_ee.nq])
        root.data[3:6] = ee_tm.translation
        
        root.H_traj = np.array([root.H])
        root.h = self._calculate_h(root)
        
        open_list = [root]
        closed_set = set()
        
        global_best_dist = float('inf')
        best_node = root

        # 3. Search Loop
        pbar = tqdm.tqdm(total=max_iter, desc="Planning")
        
        for _ in range(max_iter):
            if not open_list:
                print("Open list empty. No path found.")
                break

            # Sort by heuristic (Greedy Best First)
            open_list.sort(key=lambda x: x.h)
            current_node = open_list.pop(0)
            
            # State signature for visited set (tuple of latent variables)
            state_signature = tuple(np.round(current_node.data, 2))
            if state_signature in closed_set:
                continue
            closed_set.add(state_signature)

            # Check Goal
            dist_to_goal = self._calculate_ee_diff(current_node)
            if dist_to_goal < self.goal_proximity:
                print("Path found!")
                pbar.close()
                return current_node.H_traj

            # Track Best
            if dist_to_goal < global_best_dist:
                global_best_dist = dist_to_goal
                best_node = current_node
                np.save("best_traj_more_cs_points.npy", best_node.H_traj)
                pbar.set_postfix({"Best Dist": f"{global_best_dist:.3f}"})

            # Expand Neighbors
            neighbors = self._get_neighbors(current_node)
            a=1
            print("current node:",current_node.data,current_node.h)
            
            for neighbor in neighbors:
                neighbor_sig = tuple(np.round(neighbor.data, 2))
                if neighbor_sig in closed_set:
                    continue

                # h= self.r_lift_to_high(neighbor)
                ok, ok_c, traj, us, cost = self._lift_and_do_traj_opt(current_node,neighbor)
                if not ok:
                    continue
                neighbor.H = traj[-1]
                if not self._check_single_state(traj[-1]):
                    continue
                if ok and ok_c:
                    neighbor.g = root.g + cost
                    neighbor.H_traj = np.append(current_node.H_traj, traj, axis=0)
                    if neighbor not in open_list:
                        open_list.append(neighbor)

                if not ok_c:
                    repaired_ok, traj, us, cost = self._repair(neighbor)
                    if repaired_ok and neighbor not in open_list:
                        open_list.append(neighbor)
                        # neighbor.H_traj = np.append(root.H_traj, traj, axis=0)
                        neighbor.g = neighbor.g + cost
                    else:
                        continue

                ee_tm = get_ee_pose(self.ddp_ee, neighbor.H[:self.ddp_ee.nq])
                neighbor.data[3:6] = np.round(ee_tm.translation, 2)
                neighbor.h = self._calculate_h(neighbor)
                dist_to_goal = self._calculate_ee_diff(neighbor)
                # print(neighbor.data, np.round(neighbor.h,2), np.round(dist_to_goal,2))

            pbar.update(1)

        pbar.close()
        print(f"Max iterations reached. Best distance: {global_best_dist}")
        return best_node.H_traj

# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":

    tun1_center = np.array([0.25, -0.2, 0.19]) 
    tun1_dir = np.array([0, 1, 0])
    obs_pin = None
    obs_pin = [[tun1_center,tun1_dir]]
    # obs_pin = None
    
    # 2. Define Goal and Init
    pos = np.array([0.25, -0.32, 0.19]) 
    rot = np.array(([0,-1,0],[1,0,0],[0,0,1]))
    goal = pin.SE3(rot,pos)
    q_init = None 
    
    # 3. Run Planner
    planner = SnakePlanner(num_links=12, obs=obs_pin)
    path = planner.plan(x_init=None, goal_pose=goal)
    
    if path is not None:
        print(f"Success! Path length: {len(path)} steps")
        scipy.io.savemat('data_torq_limts.mat', {'my_matrix': path})
        np.save('data_torq_limts.npy', path)
    else:
        print("Failed to find path.")