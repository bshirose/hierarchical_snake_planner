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
from traj_opt.snake_functions import calculate_h, get_ee_problem, get_problem, do_traj_opt, get_ee_pose, add_square_tunnel, optimize_trajectory_until_collision
from traj_opt.snake_functions import invert_snake, get_curve_problem, lift_and_do_traj_opt,get_link_pose, get_tunnel_wall_aabbs    

DEBUG = True
MAKE_TUNNEL_0_LOOSE = True


link_frame_names=[
        "reu_module_out_0",
        "reu_module_out_1",
        "reu_module_out_2",
        "reu_module_out_3",
        "reu_module_out_4",
        "reu_module_out_5",
        "reu_module_out_6",
        "reu_module_out_7",
        "reu_module_out_8",
        "reu_module_out_9",        
        "reu_module_out_10",
        "reu_module_out_11",
        "reu_module_out_12",
        "reu_module_out_13",
        "reu_module_out_14",
        "reu_module_out_15",
        "reu_module_out_16"]



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
                 goal_proximity=0.04,
                 cost_grid_size=0.01,
                 obs = None,
                 q_init = None,
                 base_tf = None,
                 jammed_links = 4,
                 bounds_tuple = None,
                 bounds_limit = 1):
        
        if q_init is None or base_tf is None or bounds_tuple is None:
            print("fuck you")
            exit()
            raise ValueError("base_loc, q_init, and base_tf must be provided.")
        
        
        # Robot Physical Parameters
        self.num_links = num_links
        self.link_length = link_length
        self.num_jammed_links = jammed_links
        self.bezier_length = (num_links- self.num_jammed_links) * link_length
        self.bezier_points = num_links- self.num_jammed_links
        self.base_tf = base_tf  # Assuming base rotation is identity for now
        self.q0 = q_init


        self.environmental_bounds = bounds_limit

        self.ddp = get_problem(base_placement=self.base_tf)
        self.ddp_bb = get_curve_problem(base_placement=self.base_tf,jammed_links=self.num_jammed_links+1)

        


        self.first_locs = []
        for i in range(self.num_jammed_links):
            link_name = link_frame_names[i]
            link_pos = get_link_pose(self.ddp,q_init,link_name)
            self.first_locs.append(link_pos.translation)
        link_name = link_frame_names[self.num_jammed_links]
        self.bezier_base = get_link_pose(self.ddp,q_init,link_name).translation
        



        
        # Planning Constraints & Thresholds
        self.goal_proximity = goal_proximity
        # self.bounds_limit = bounds
        self.bounds_tuple = bounds_tuple
        self.length_thresh = 1.05
        
        # Search Parameters
        self.delta_t = 0.03
        self.delta_ts = 0.01
        self.delta_tl = 0.1
        self.delta_cl = 0.15
        self.delta_cs = 0.05
        self.cost_grid_size = cost_grid_size

        # Internal Counter
        self._node_counter = 0


        self.default_wall_thick = 0.04  # 2 cm thick walls
        self.default_gap_len = 0.3
        self.default_gap_height = 0.07
        self.default_gap_width = 0.07
        walls =[]
        if obs is not None:
            for ob in obs:
                if MAKE_TUNNEL_0_LOOSE and np.linalg.norm(ob[0]- np.array([-0.05,0.03,0]))<0.001:

                    _, walls_sol = get_tunnel_wall_aabbs(center_pos=ob[0], 
                                                        orientation_axis=ob[1], 
                                                        width=0.13, 
                                                        height=0.13, 
                                                        length=0.2, 
                                                        wall_thick=self.default_wall_thick)
                    # walls_sol.pop(-2)
                    walls = walls+ walls_sol
                    center, direction = ob
                    add_square_tunnel(self.ddp,
                                    center, 
                                    direction,
                                    width=0.12,
                                    height=0.12,
                                    length=0.2,
                                    wall_thick=self.default_wall_thick)
                else:
                    _, walls_sol = get_tunnel_wall_aabbs(center_pos=ob[0], 
                                            orientation_axis=ob[1], 
                                            width=self.default_gap_width, 
                                            height=self.default_gap_height, 
                                            length=self.default_gap_len, 
                                            wall_thick=self.default_wall_thick)
                    walls = walls+ walls_sol
                    center, direction = ob
                    add_square_tunnel(self.ddp,
                                    center, 
                                    direction,
                                    width=self.default_gap_width,
                                    height=self.default_gap_height,
                                    length=self.default_gap_len,
                                    wall_thick=self.default_wall_thick)
        back = np.array([36, -45,  10, 55,  -35,  30])/100 
        back2 = np.array([46.,  11.,  70,  64.,  29.,75])/100

        self.obstacles = walls
        self.obstacles.append(back)
        self.obstacles.append(back2)
    # =========================================================================
    #    Helper Methods (Dynamics, Kinematics, Collision, Heuristics)
    # =========================================================================
    def _fit_cp_to_state(self, q_full):
        """
        Calculates the Control Point that best fits the actual robot shape
        using Linear Least Squares.
        """
        # P_world is shape (3, N+1)
        P_world = []
        for i in range(self.num_jammed_links, self.num_links+1):
            link_name = link_frame_names[i]
            link_pos = get_link_pose(self.ddp,q_full,link_name)
            P_world.append(link_pos.translation)
        P_world = np.array(P_world).T
        
        base = P_world[:, 0]
        ee = P_world[:, -1]
        numerator = np.zeros(3)
        denominator = 0.0
        
        # We skip 0 and N because Bezier curves match endpoints exactly by definition
        for i in range(1, self.bezier_points): 
            s = i / self.bezier_points
            L_i = P_world[:, i]
            
            # Bezier terms
            # P(s) = A_i + w_i * P1
            # where A_i is the fixed part contributed by Base and EE
            w_i = 2 * s * (1 - s)
            A_i = ((1 - s)**2 * base) + (s**2 * ee)
            
            # Accumulate terms
            numerator += w_i * (L_i - A_i)
            denominator += w_i**2
            
        if denominator < 1e-6:
            return (base + ee) / 2.0 # Fallback (shouldn't happen for N > 2)
            
        best_cp = numerator / denominator
        return best_cp
    

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

    def _generate_b_curve_points(self, node):
        """
        Maps latent Bezier parameters to joint angles via IK.
        Now includes Warm Starting and Feasibility Checks.
        """
        link_locs = []
        control = node.data[:3]
        end = node.data[3:]
        
        for i in range(self.bezier_points + 1):
            s = i / self.bezier_points
            # Quadratic Bezier
            loc = (1-s)*(1-s)*self.bezier_base + 2*(1-s)*s*control + s*s*end
            link_locs.append(loc)

        return link_locs


    def _lift_and_do_traj_opt(self,q_init, node):

        # q_init = parent.H[:self.ddp.nq] if parent is not None else np.zeros(self.ddp.nq)
        link_locs = self._generate_b_curve_points(node)
        true_link_locs = [self.first_locs[i] for i in range(len(self.first_locs))]
        for i in range(len(link_locs)):
            true_link_locs.append(link_locs[i])
        link_locs = true_link_locs
        ok, traj, us, cost = lift_and_do_traj_opt(self.ddp_bb, q_init, link_locs,threshold= 0.15,goal = self.goal_pose)
        ok_c = self._check_collision(traj)
        # if ok and ok_c:
        #     ok_new, traj_new, us, cost = optimize_trajectory_until_collision(self.ddp, traj)
        #     if ok_new:
        #         traj = traj_new
        #         ok = ok_new

        return ok, ok_c, traj, us, cost
    
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
            ok, traj,us,cost = self._do_traj_opt(a.H[:self.ddp.nq], node.H[:self.ddp.nq])
            if ok:
                node.g = a.g + cost
                node.H_traj = np.append(a.H_traj, traj, axis=0)
                node.H = traj[-1]
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
        max_voxels = 1000000 
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
        ok, xs, us, cost = do_traj_opt(self.ddp, q_init, q_f)
        if not ok:
            return False, None, None, None
        ok_c = self._check_collision(xs)
        return ok and ok_c, xs, us, cost
    

    def _calculate_ee_diff(self, node):
        ee_pose = self._get_ee_pose(node.H[:self.ddp.nq])
        return np.linalg.norm(ee_pose.translation - self.goal_pose.translation)
    
    def _check_collision(self, traj, stride=5):
        if not self._check_single_state(traj[0]): return False
        if not self._check_single_state(traj[-1]): return False
        for x in traj[stride:-1:stride]:
            if not self._check_single_state(x):
                return False
        return True
    
    
    def _check_single_state(self, x):
        q = x[:self.ddp.nq]
        _, n_collisions = self.ddp.collisionCheck(q)
        if n_collisions > 0:
            a=1
        return n_collisions == 0
    

    def _check_arc_length(self, end, control):
        dist = 0
        loc0 = self.bezier_base
        for i in range(1, self.num_links + 1):
            s = i / self.num_links
            loc = (1-s)*(1-s)*self.bezier_base + 2*(1-s)*s*control + s*s*end
            dist += np.linalg.norm(loc - loc0)
            loc0 = loc
        return dist > (self.bezier_length * self.length_thresh)

    def _check_bounds(self, data):
        return np.all(np.abs(data) <= self.environmental_bounds)

    def _get_ee_pose(self,q):
        return get_link_pose(self.ddp,q,link_frame_names[-1])


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
    
    


    def plan(self, goal_pose, max_iter=15000):

        self._node_counter = 0
        self.goal_pose = goal_pose
        print("Generating cost field...")
        self.cost_map = self._generate_cost_field(self.goal_pose.translation, self.bounds_tuple)
        print("Cost field generated.")

        cp = self._fit_cp_to_state(self.q0)
        ee_tm = self._get_ee_pose(self.q0)
        initial_latent = np.array([cp[0], cp[1], cp[2], ee_tm.translation[0], ee_tm.translation[1], ee_tm.translation[2]])
        
        root = Node(initial_latent, self._get_id())
        ok,ok_c,traj,us,cost = self._lift_and_do_traj_opt(self.q0,root)
        if not ok or not ok_c:
            print("Initial trajectory optimization failed. Check if the goal is reachable or if the initial guess is valid.")
            return None
        root.H = traj[-1]
        ee_tm = self._get_ee_pose(root.H[:self.ddp.nq])
        root.data[3:6] = ee_tm.translation
        
        root.H_traj = np.array(traj[0])
        root.H_traj = np.vstack((root.H_traj,traj))
        root.h = self._calculate_h(root)
        
        open_list = [root]
        closed_set = set()
        
        global_best_dist = float('inf')
        best_node = root

        # 3. Search Loop
        pbar = tqdm.tqdm(total=max_iter, desc="Planning")

        if DEBUG:
            self.ddp.InitDisplay()
        
        for _ in range(max_iter):
            if not open_list:
                print("Open list empty. No path found.")
                break

            # Sort by heuristic (Greedy Best First)
            open_list.sort(key=lambda x: x.h)
            current_node = open_list.pop(0)
            
            if DEBUG:
                self.ddp.ShowSnakeConfiguration(current_node.H[:self.ddp.nq])
            # State signature for visited set (tuple of latent variables)
            state_signature = tuple(np.round(current_node.data, 2))
            if state_signature in closed_set:
                continue
            closed_set.add(state_signature)

            # Check Goal
            dist_to_goal = self._calculate_ee_diff(current_node)
            if dist_to_goal < self.goal_proximity:
                print("Path found!", f"Distance to goal: {dist_to_goal:.3f}")
                print(current_node.H_traj[-1,:16])
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
                ok, ok_c, traj, us, cost = self._lift_and_do_traj_opt(current_node.H[:self.ddp.nq],neighbor)
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

                ee_tm = self._get_ee_pose(neighbor.H[:self.ddp.nq])
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

    tun1_center = np.array([0.45, -0.22, 0.19]) 
    tun1_dir = np.array([0, 1, 0])

    tun2_center = np.array([0.55,0.2,0.55]) 
    tun2_dir = np.array([0, 0, 1])

    tun3_center = np.array([-0.05,0.03,0.0]) 
    tun3_dir = np.array([1, 0, 0])
    obs_pin = None
    obs_pin = [[tun1_center,tun1_dir],[tun2_center,tun2_dir]]
    # obs_pin = [[tun1_center,tun1_dir],[tun2_center,tun2_dir],[tun3_center,tun3_dir]]
    # obs_pin = None
    bounds_limit = 0.65
    bounds_tuple = ((-0.1+0.253, bounds_limit+0.253), 
                (-bounds_limit, bounds_limit), 
                (-0.1, bounds_limit))

    # 2. Define Goal and Init
    pos = np.array([0.20+0.26, -0.34, 0.19]) 
    rot = np.array(([0,-1,0],[1,0,0],[0,0,1]))
    goal = pin.SE3(rot,pos)
    q_init = np.zeros(16)
    base_rotation = pin.utils.rpyToMatrix(0, 0, 1.57079632679)
    base_translation = np.array([0, 0.0, 0.0])
    my_base_pose = pin.SE3(base_rotation, base_translation)
    t = time.time()
    
    # 3. Run Planner
    planner = SnakePlanner(num_links=16, obs=obs_pin,q_init=q_init, base_tf=my_base_pose,jammed_links=4,bounds_tuple=bounds_tuple)
    path = planner.plan(goal_pose=goal)
    seg = {}
    seg["obs"] = obs_pin
    seg["obs_width"] = planner.default_gap_width    
    seg["obs_height"] = planner.default_gap_height
    seg["obs_length"] = planner.default_gap_len
    seg["wall_thick"] = planner.default_wall_thick
    seg["total_segments"] = 2

    seg1 = {}
    seg1["goal"] = goal  
    seg1["base"] = my_base_pose
    seg1["q_init"] = q_init
    seg1["path"] = path
    
    if path is not None:
        print(f"Success! Path length: {len(path)} steps")
        scipy.io.savemat('p1.mat', {'my_matrix': path})
        np.save('p1.npy', path)
    else:
        print("Failed to find path.")
        exit()
    # path = np.load("p1.npy")
    new_base_tf, q_init_2 = invert_snake(planner.ddp, path[-1][:16])

    bounds_tuple = ((-0.1, 0.6), 
                    (-0.1, 0.65), 
                    (-0.1, 0.7))
    
    rot = np.array(([ 0, 0, 1],
                    [ 0, 1, 0],
                    [-1, 0, 0]))
    rot_x_180 = pin.utils.rpyToMatrix(np.pi,0,0)
    rot = rot @ rot_x_180
    pos =np.array([0.55,0.2,0.6])
    goal2 = pin.SE3(rot,pos)



    planner2 = SnakePlanner(num_links=16, obs=obs_pin,q_init=q_init_2, base_tf=new_base_tf,jammed_links=4,bounds_tuple=bounds_tuple)
    path2 = planner2.plan(goal_pose=goal2)
    elapsed = time.time() - t
    print(f"Planning completed in {elapsed:.2f} seconds.")
    if path2 is not None:
        print(f"Success! Path length: {len(path2)} steps")
        scipy.io.savemat('p2.mat', {'my_matrix': path2})
        np.save('p2.npy', path2)
    else:
        print("Failed to find path.")
    
    seg2 = {}
    seg2["goal"] = goal2
    seg2["base"] = new_base_tf
    seg2["q_init"] = q_init_2
    seg2["path"] = path2

    segs = [seg1, seg2]
    seg["segs"] = segs

    with open('segmented_paths_.pkl', 'wb') as f:
        pickle.dump(seg, f)



    