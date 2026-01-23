import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from copy import copy
from scipy.optimize import minimize
import numpy as np
# import pinocchio as pin
# import crocoddyl
import pickle   
import heapq
import time

import math

import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import casadi as ca

NUM_LINKS = 16
BASE_LOC = np.array([0,0,0])
MAX_ITER = 1000
LINK_LENGTH = 0.05105
SNAKE_LENGTH = NUM_LINKS*LINK_LENGTH
LENGTH_THRESH = 1.05
LINK_MASS = 0.250
BASE_TF = np.eye(4)
dt = 0.1
TORQUE_LIMIT = 16
JAMMED_LINKS = 4
GOAL_PROXIMITY = 0.04
DELTA_T = 0.03
DELTA_TL = 0.1
DELTA_CL = 0.4
INTERP_STEPS = 10
COST_GRID_SIZE = 0.03
BOUNDS = 0.9

debug = False

class node():
    def __init__(self, L, id):
        self.data = L
        self.parent = None
        self.children = []
        self.H = None
        self.g = 0
        self.f = 0
        self.h = 0
        self.reach_time =0
        self.H_traj = []
        self.node_id = id


# def generate_cost_field(start_point, voxel_size, bounds, is_obstacle_func,obstacles):
#     """
#     Expands from start_point to calculate movement costs to all reachable voxels.
    
#     Args:
#         start_point (tuple): (x, y, z) float coordinates.
#         voxel_size (float): The size of the grid cells.
#         bounds (tuple): ((min_x, max_x), (min_y, max_y), (min_z, max_z)) in world coords.
#         is_obstacle_func (func): A function that takes (x,y,z) and returns True/False.
        
#     Returns:
#         dict: A map of voxel_index (ix, iy, iz) -> cost (float)
#     """
    
#     # 1. Helper to discretize coordinates
#     def to_index(pt):
#         return (math.floor(pt[0] / voxel_size), 
#                 math.floor(pt[1] / voxel_size), 
#                 math.floor(pt[2] / voxel_size))

#     # 2. Helper to get center of a voxel (for collision checking)
#     def to_center(idx):
#         return ((idx[0] + 0.5) * voxel_size, 
#                 (idx[1] + 0.5) * voxel_size, 
#                 (idx[2] + 0.5) * voxel_size)
#     x_range = bounds[0][1] - bounds[0][0]
#     y_range = bounds[1][1] - bounds[1][0]
#     z_range = bounds[2][1] - bounds[2][0]
    
#     total_volume = x_range * y_range * z_range
#     estimated_total_voxels = int(total_volume / (voxel_size ** 3))
#     # 3. Initialize structures
#     start_idx = to_index(start_point)
    
#     # Priority Queue stores tuples: (current_cost, voxel_index_tuple)
#     pq = [(0.0, start_idx)]
    
#     # Dictionary stores: voxel_index -> min_cost
#     # This acts as our 'visited' set and result storage
#     costs = {start_idx: 0.0}
    
#     # Pre-calculate bound indices for faster checking
#     min_indices = to_index((bounds[0][0], bounds[1][0], bounds[2][0]))
#     max_indices = to_index((bounds[0][1], bounds[1][1], bounds[2][1]))

#     # 6 Neighbor directions (Up, Down, Left, Right, Forward, Back)
#     neighbors = [
#         (1,0,0), (-1,0,0), 
#         (0,1,0), (0,-1,0), 
#         (0,0,1), (0,0,-1)
#     ]

    
#     with tqdm.tqdm(total=estimated_total_voxels, desc="Expanding Grid", unit="vox") as pbar:
        
#         while pq:
#             current_cost, current_idx = heapq.heappop(pq)
            
#             # Update progress bar
#             pbar.update(1)

#             if current_cost > costs[current_idx]:
#                 continue

#             for dx, dy, dz in neighbors:
#                 neighbor_idx = (current_idx[0] + dx, current_idx[1] + dy, current_idx[2] + dz)
                
#                 # A. Check Bounds
#                 if not (min_indices[0] <= neighbor_idx[0] < max_indices[0] and
#                         min_indices[1] <= neighbor_idx[1] < max_indices[1] and
#                         min_indices[2] <= neighbor_idx[2] < max_indices[2]):
#                     continue

#                 # B. Check Obstacles
#                 if is_obstacle_func(to_center(neighbor_idx),obstacles):
#                     continue

#                 # C. Update Path
#                 new_cost = current_cost + voxel_size
#                 if neighbor_idx not in costs or new_cost < costs[neighbor_idx]:
#                     costs[neighbor_idx] = new_cost
#                     heapq.heappush(pq, (new_cost, neighbor_idx))
                
#     return costs
#     # # while pq:
#     # with tqdm(total=estimated_total_voxels, desc="Expanding Grid", unit="vox") as pbar:
#     #     current_cost, current_idx = heapq.heappop(pq)

#     #     # Optimization: If we found a shorter path to this node already, skip
#     #     if current_cost > costs[current_idx]:
#     #         continue

#     #     # Expand to neighbors
#     #     for dx, dy, dz in neighbors:
#     #         neighbor_idx = (current_idx[0] + dx, current_idx[1] + dy, current_idx[2] + dz)
            
#     #         # A. Check Bounds
#     #         if not (min_indices[0] <= neighbor_idx[0] < max_indices[0] and
#     #                 min_indices[1] <= neighbor_idx[1] < max_indices[1] and
#     #                 min_indices[2] <= neighbor_idx[2] < max_indices[2]):
#     #             continue

#     #         # B. Check Obstacles (convert index back to real world center point)
#     #         if is_obstacle_func(to_center(neighbor_idx),obstacles):
#     #             continue

#     #         # C. Calculate new cost (assuming uniform cost of voxel_size per step)
#     #         new_cost = current_cost + voxel_size

#     #         # D. Update if we found a cheaper path or haven't visited yet
#     #         if neighbor_idx not in costs or new_cost < costs[neighbor_idx]:
#     #             costs[neighbor_idx] = new_cost
#     #             heapq.heappush(pq, (new_cost, neighbor_idx))
                
#     return costs

def generate_cost_field(start_point, voxel_size, bounds, is_obstacle_func, obstacles):
    # 1. Coordinate helpers
    def to_index(pt):
        return (math.floor(pt[0] / voxel_size), 
                math.floor(pt[1] / voxel_size), 
                math.floor(pt[2] / voxel_size))

    def to_center(idx):
        return ((idx[0] + 0.5) * voxel_size, 
                (idx[1] + 0.5) * voxel_size, 
                (idx[2] + 0.5) * voxel_size)

    # 2. Setup Bounds
    min_indices = to_index((bounds[0][0], bounds[1][0], bounds[2][0]))
    max_indices = to_index((bounds[0][1], bounds[1][1], bounds[2][1]))
    
    # Calculate volume for tqdm
    x_dim = max_indices[0] - min_indices[0]
    y_dim = max_indices[1] - min_indices[1]
    z_dim = max_indices[2] - min_indices[2]
    estimated_total_voxels = max(1, x_dim * y_dim * z_dim)

    # 3. Initialize with INTEGERS
    start_idx = to_index(start_point)
    
    # Priority Queue: (step_count (int), voxel_idx)
    pq = [(0, start_idx)]
    
    # Costs map: voxel_idx -> step_count (int)
    steps_map = {start_idx: 0}
    
    neighbors = [
        (1,0,0), (-1,0,0), 
        (0,1,0), (0,-1,0), 
        (0,0,1), (0,0,-1)
    ]

    with tqdm.tqdm(total=estimated_total_voxels, desc="Expanding Grid", unit="vox") as pbar:
        while pq:
            current_steps, current_idx = heapq.heappop(pq)
            
            # Integer comparison is exact and safe
            if current_steps > steps_map[current_idx]:
                continue
            
            pbar.update(1)

            for dx, dy, dz in neighbors:
                neighbor_idx = (current_idx[0] + dx, current_idx[1] + dy, current_idx[2] + dz)
                
                # A. Check Bounds (Indices are integers, so this is safe)
                if not (min_indices[0] <= neighbor_idx[0] < max_indices[0] and
                        min_indices[1] <= neighbor_idx[1] < max_indices[1] and
                        min_indices[2] <= neighbor_idx[2] < max_indices[2]):
                    continue

                # B. Check Obstacles
                if is_obstacle_func(to_center(neighbor_idx), obstacles):
                    continue

                # C. Update Path (Integer Math)
                new_steps = current_steps + 1
                
                if neighbor_idx not in steps_map or new_steps < steps_map[neighbor_idx]:
                    steps_map[neighbor_idx] = new_steps
                    heapq.heappush(pq, (new_steps, neighbor_idx))
                
    return steps_map



def inverse_dynamics_rnea(q, q_dot, q_ddot, n_links, L, m, g):
    """
    Recursive Newton-Euler Algorithm (RNEA) for a snake-like robot.
    
    Args:
        q, q_dot, q_ddot: Joint angles, velocities, and accelerations (1D arrays).
        n_links: Number of links (int).
        L: Link length (float).
        m: Link mass (float).
        g: Gravity vector (e.g., [0, 0, -9.81]).
    
    Returns:
        tau: Joint torques (1D array).
    """
    
    # Ensure inputs are numpy arrays
    q = np.array(q)
    q_dot = np.array(q_dot)
    q_ddot = np.array(q_ddot)
    g = np.array(g)

    # Constants and Setup
    CoM = np.array([L/2, 0, 0])
    # Inertia tensor for a thin rod (approximated as in MATLAB script)
    I_CoM = np.diag([0, (1/12)*m*L**2, (1/12)*m*L**2])
    
    z_axis = np.array([0, 0, 1])
    y_axis = np.array([0, 1, 0])

    # Initialize kinematic containers (Size N+1 to include base)
    # Python index 0 = Base (MATLAB index 1 implicit in loops)
    # Python index i+1 = Link i
    w = np.zeros((3, n_links + 1))
    w_dot = np.zeros((3, n_links + 1))
    a = np.zeros((3, n_links + 1))
    ac = np.zeros((3, n_links + 1))

    # Base acceleration (gravity effect)
    a[:, 0] = -g

    # --- Forward Pass (Kinematics) ---
    for i in range(n_links):
        # Determine rotation based on link index
        # MATLAB: odd(i=1) -> Z, even(i=2) -> Y
        # Python: even(i=0) -> Z, odd(i=1) -> Y
        if i % 2 == 0: 
            # Z-axis rotation
            c, s = np.cos(q[i]), np.sin(q[i])
            R_world_to_body = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            joint_axis = z_axis
        else:
            # Y-axis rotation
            c, s = np.cos(q[i]), np.sin(q[i])
            R_world_to_body = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            joint_axis = y_axis
            
        R_body_to_world = R_world_to_body.T # Transpose

        # Indices: i is current joint, i+1 is current link body
        # Previous body is i
        
        # Angular velocity w
        w[:, i+1] = R_body_to_world @ w[:, i] + q_dot[i] * joint_axis
        
        # Angular acceleration w_dot
        w_dot[:, i+1] = (R_body_to_world @ w_dot[:, i] + 
                         np.cross(R_body_to_world @ w[:, i], q_dot[i] * joint_axis) + 
                         q_ddot[i] * joint_axis)
        
        # Linear acceleration a
        p_link = np.array([L, 0, 0])
        a_prev_term = a[:, i] + np.cross(w_dot[:, i], p_link) + np.cross(w[:, i], np.cross(w[:, i], p_link))
        a[:, i+1] = R_body_to_world @ a_prev_term
        
        # CoM acceleration ac
        ac[:, i+1] = a[:, i+1] + np.cross(w_dot[:, i+1], CoM) + np.cross(w[:, i+1], np.cross(w[:, i+1], CoM))

    # --- Backward Pass (Dynamics) ---
    f = np.zeros((3, n_links + 1))
    n = np.zeros((3, n_links + 1))
    tau = np.zeros(n_links)

    # Iterate backwards: n_links-1 down to 0
    for i in range(n_links - 1, -1, -1):
        # Inertial Force and Torque on Link
        F = m * ac[:, i+1]
        N = I_CoM @ w_dot[:, i+1] + np.cross(w[:, i+1], I_CoM @ w[:, i+1])
        p_link = np.array([L, 0, 0])

        if i == n_links - 1:
            # Last link (Tip)
            f[:, i] = F
            n[:, i] = N + np.cross(CoM, F)
        else:
            # Inner links
            # Determine Rotation of NEXT joint (i+1)
            # Python index i+1 corresponds to MATLAB i+2 check (mod(i+2, 2))
            # MATLAB: mod(i+1, 2). If i was loop var.
            # Here next link index is j = i + 1. 
            # If j is even (Python) -> Z axis. If odd -> Y axis.
            next_idx = i + 1
            if next_idx % 2 == 0:
                c, s = np.cos(q[next_idx]), np.sin(q[next_idx])
                R_next = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
            else:
                c, s = np.cos(q[next_idx]), np.sin(q[next_idx])
                R_next = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T
            
            f[:, i] = R_next @ f[:, i+1] + F
            n[:, i] = (N + R_next @ n[:, i+1] + 
                       np.cross(CoM, F) + 
                       np.cross(p_link, R_next @ f[:, i+1]))

        # Project torque onto joint axis
        if i % 2 == 0:
            joint_axis = z_axis
        else:
            joint_axis = y_axis
            
        tau[i] = np.dot(n[:, i], joint_axis)

    return tau

def Rz(theta):
    """Rotation matrix around Z axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])

def Ry(theta):
    """Rotation matrix around Y axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ])

def Tx(d):
    """Translation matrix along X axis."""
    return np.array([
        [1, 0, 0, d],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def forward_kinematics(q, link_length, T_base):
    """
    Computes forward kinematics for a snake-like robot.
    
    Args:
        q (array-like): Joint angles.
        link_length (float): Length of each link.
        T_base (4x4 array): Transformation matrix of the base.
        base_end (str): 'HEAD_BASE' or 'TAIL_BASE'.
        
    Returns:
        P_world (3xN array): Positions of all joints in world frame.
        T_ee_world (4x4 array): Transformation of the end-effector.
    """
    q = np.atleast_1d(q) # Ensure q is iterable
    n_links = len(q)
    
    # Initialize storage for transforms
    T_links_in_head_frame = [np.eye(4)] # T_links_in_head_frame{1} equivalent
    T_current = np.eye(4)

    # Compute kinematics chain
    for i in range(n_links):
        # Determine joint rotation axis
        # MATLAB index 1 (odd) -> Python index 0 (even) -> Rz
        # MATLAB index 2 (even) -> Python index 1 (odd) -> Ry
        if i % 2 == 0:
            T_joint = Rz(q[i])
        else:
            T_joint = Ry(q[i])
            
        # Accumulate transform: T_current * T_joint * Tx(link_length)
        T_current = T_current @ T_joint @ Tx(link_length)
        T_links_in_head_frame.append(T_current)

    # Get tail transform (last element in the list)
    T_tail_in_head_frame = T_links_in_head_frame[-1]

    # Handle Base End Logic
    T_world_to_head = T_base
    T_ee_world = T_world_to_head @ T_tail_in_head_frame

    # Compute P_world
    # Initialize P_world (3 rows, n_links + 1 columns)
    P_world = np.zeros((3, n_links + 1))
    
    for i in range(n_links + 1):
        T_link_world = T_world_to_head @ T_links_in_head_frame[i]
        P_world[:, i] = T_link_world[0:3, 3] # Extract position column

    return P_world, T_ee_world

def is_point_in_obstacle(p,obstacles):
    # return False
    if obstacles is None or len(obstacles) == 0:
        return False
    for obs in obstacles:
        x_min, y_min, z_min, x_max, y_max, z_max = obs
        if (p[0] >= x_min and p[0] <= x_max and
            p[1] >= y_min and p[1] <= y_max and
            p[2] >= z_min and p[2] <= z_max):    
                return True 
    return False

def is_collision_free(q, link_length, T_base, obstacles):
    """
    Checks if a configuration 'q' collides with cuboid obstacles.
    
    Args:
        q: Joint angles.
        link_length: Length of links.
        T_base: Base transform.
        obstacles: List or array of obstacles. Each row is [xmin, ymin, zmin, xmax, ymax, zmax].
    
    Returns:
        bool: True if free (no collision), False otherwise.
    """
    
    # Handle empty obstacles case
    if obstacles is None or len(obstacles) == 0:
        return True
    
    obstacles = np.array(obstacles)
    
    # Get all joint positions in the world frame
    # (Relies on the forward_kinematics function defined previously)
    P_world, _ = forward_kinematics(q, link_length, T_base)
    
    n_links = P_world.shape[1] - 1
    
    # Iterate through obstacles
    for obs in obstacles:
        x_min, y_min, z_min, x_max, y_max, z_max = obs
        
        # Check each link segment
        for i in range(n_links):
            p1 = P_world[:, i]
            p2 = P_world[:, i+1]
            
            # Interpolate points along the link
            # MATLAB: 0:0.05:1 implies 21 points (0.0, 0.05, ... 1.0)
            # Using linspace to match this resolution
            steps = 51 
            t_values = np.linspace(0, 1, steps)
            
            for t in t_values:
                p = (1 - t) * p1 + t * p2 # Interpolated point
                
                # Check if the point 'p' is inside the cuboid
                if (p[0] >= x_min and p[0] <= x_max and
                    p[1] >= y_min and p[1] <= y_max and
                    p[2] >= z_min and p[2] <= z_max):
                    
                    return False # Collision detected

    return True # No collisions found

def get_id():
    get_id.counter += 1
    return get_id.counter
get_id.counter = -1

def solve_snake_ik(target_pts, base_tf, link_length, initial_q=None):
    """
    Calculates joint angles to align each robot link to a corresponding target point.
    
    Args:
        target_pts (np.ndarray): Shape (3, N+1). The x,y,z coordinates for joints.
        base_tf (np.ndarray): Base transformation matrix (4x4).
        link_length (float): Length of each segment.
        initial_q (np.ndarray, optional): Guess for joint angles. Shape (N,).
        
    Returns:
        np.ndarray: Optimized relative joint angles (radians).
    """
    # 1. Setup Initial Guess (Warm Start)
    # If initial_q is provided, use it. Otherwise, default to zeros.
    if initial_q is not None:
        q_guess = np.array(initial_q)
        num_links = len(q_guess)
    else:
        # Infer number of links from target points if q not provided
        # target_pts has N+1 columns (base + N joints)
        num_links = target_pts.shape[1] - 1
        q_guess = np.zeros(num_links)
    
    # Bounds: -pi/2 to pi/2 for each joint
    bounds = [(-np.pi/2, np.pi/2) for _ in range(num_links)]
    
    # 2. Vectorized Objective Function
    def objective(q):
        # Calculate FK
        # Note: 'HEAD_BASE' is assumed based on previous context
        P_world, _ = forward_kinematics(q, link_length, base_tf)
        
        # Vectorized cost calculation (No for-loops)
        # P_world and target_pts are both (3, N+1) matrices
        diff = P_world - target_pts
        
        # Sum of squared errors
        return np.sum(diff**2)

    # 3. Optimization
    # 'L-BFGS-B' is often faster than 'SLSQP' for simple bound-constrained problems
    result = minimize(
        objective, 
        q_guess, 
        method='SLSQP', # Or try 'L-BFGS-B' if you don't have complex constraints
        bounds=bounds,
        options={'ftol': 1e-4, 'disp': False} # Relaxed tolerance slightly for speed
    )
    
    return result.x




def _save(n):
    with open('node_data.pkl', 'wb') as f:
        pickle.dump(n, f)

def check_bounds(dat):
    if np.any(dat>BOUNDS):
        return False
    if np.any(dat<-BOUNDS):
        return False
    return True

def get_neighbours_less(n):
    neighbours = []
    # delta = DELTA_T
    # for dx in [-delta,delta]:
    #     for ind in range(3,6):
    #     # for ind in range(len(n.data)):
    #         # check bounds
    #         new_data = copy(n.data)
    #         new_data[ind] += dx
    #         new_data = np.round(new_data, decimals=2)
    #         if check_b_arc_length(new_data[3:6],new_data[:3]):
    #             continue
    #         if not check_bounds(new_data):
    #             continue
    #         new_node = node(new_data, get_id())
    #         new_node.parent = n
    #         n.children.append(new_node)
    #         neighbours.append(new_node)

    delta = DELTA_TL
    for dx in [-delta,delta]:
        for ind in range(3,6):
        # for ind in range(len(n.data)):
            # check bounds
            new_data = copy(n.data)
            new_data[ind] += dx
            new_data = np.round(new_data, decimals=2)
            if check_b_arc_length(new_data[3:6],new_data[:3]):
                continue

            if not check_bounds(new_data):
                continue
            new_node = node(new_data, get_id())
            new_node.parent = n
            n.children.append(new_node)
            neighbours.append(new_node)
    delta = DELTA_CL
    for dx in [-delta,delta]:
        for ind in range(3):
        # for ind in range(len(n.data)):
            # check bounds
            new_data = copy(n.data)
            new_data[ind] += dx
            new_data = np.round(new_data, decimals=2)
            if check_b_arc_length(new_data[3:6],new_data[:3]):
                continue
            if not check_bounds(new_data):
                continue
            new_node = node(new_data, get_id())
            new_node.parent = n
            n.children.append(new_node)
            neighbours.append(new_node)
    return neighbours


def lift_to_high(n):
    link_locs = []
    control = n.data[:3]
    end = n.data[3:]
    for i in range(NUM_LINKS+1):
        s= i/NUM_LINKS
        loc = (1-s)*(1-s)*BASE_LOC + 2*(1-s)*s*control + s*s*end
        link_locs.append(loc)
    link_locs = np.array(link_locs)
    link_locs = link_locs.T
    angles = solve_snake_ik(link_locs, BASE_TF, LINK_LENGTH,n.H)
    return angles

def calculate_h(n,costs):
    # _ , ee = forward_kinematics(n.H, LINK_LENGTH, BASE_TF)
    # ee_pos = ee[0:3,3]
    # h = np.linalg.norm(ee_pos - goal)
    h =get_point_cost(n.data[3:6], costs)
    if h >100:
        ifweonfw= 1
    return h

def calculate_ee_diff(n,goal):
    _ , ee = forward_kinematics(n.H, LINK_LENGTH, BASE_TF)
    ee_pos = ee[0:3,3]
    h = np.linalg.norm(ee_pos - goal)
    return h

def generate_traj(q0,qf):
    N = INTERP_STEPS
    traj_linear = np.linspace(q0, qf, N)
    return traj_linear

def traj_check_torque(traj):
    # check first and last points with holding torque
    tau_start = inverse_dynamics_rnea(traj[0], np.zeros(NUM_LINKS), np.zeros(NUM_LINKS), NUM_LINKS, LINK_LENGTH, LINK_MASS, np.array([0,0,9.81]))
    tau_start[:JAMMED_LINKS] = np.zeros(JAMMED_LINKS)
    if np.any(np.abs(tau_start) > TORQUE_LIMIT):
        return False
    tau_end = inverse_dynamics_rnea(traj[-1], np.zeros(NUM_LINKS), np.zeros(NUM_LINKS), NUM_LINKS, LINK_LENGTH, LINK_MASS, np.array([0,0,9.81]))
    tau_end[:JAMMED_LINKS] = np.zeros(JAMMED_LINKS)
    if np.any(np.abs(tau_end) > TORQUE_LIMIT):
        return False

    for i in range(len(traj)-2):
        q = traj[i]
        q_next = traj[i+1]
        q_n_next = traj[i+2]
        q_dot = (q_next - q)/dt
        q_dot_next = (q_n_next - q_next)/dt
        q_ddot = (q_dot_next - q_dot)/dt
        tau = inverse_dynamics_rnea(q, q_dot, q_ddot, NUM_LINKS, LINK_LENGTH, LINK_MASS, np.array([0,0,9.81]))
        tau[:JAMMED_LINKS] = np.zeros(JAMMED_LINKS)
        if np.any(np.abs(tau) > TORQUE_LIMIT):
            return False
    return True

def check_b_arc_length(e,c1):
    dist = 0
    loc0 =BASE_LOC
    for i in range(1,NUM_LINKS+1):
        s= i/NUM_LINKS
        loc = (1-s)*(1-s)*BASE_LOC + 2*(1-s)*s*c1 + s*s*e
        dist = dist+np.linalg.norm(loc - loc0)
        loc0 = loc
    return dist>(SNAKE_LENGTH*LENGTH_THRESH)

def get_voxel_index(point, voxel_size):
    """
    Takes a 3D point (x, y, z) and returns its corresponding
    voxel grid index based on the specified voxel_size.
    """
    x, y, z = point
    return (
        math.floor(x / voxel_size),
        math.floor(y / voxel_size),
        math.floor(z / voxel_size)
    )


def check_collision_traj(traj, obstacles):
    for q in traj:
        if not is_collision_free(q, LINK_LENGTH, BASE_TF, obstacles):
            return False
    return True

def solve(q0,qf,obstacles):
    traj = generate_traj(q0,qf)
    ok =True
    ok = traj_check_torque(traj)
    ok = ok and check_collision_traj(traj, obstacles)
    return ok, traj

def repair(n,obstacles):
    pa = n.parent
    ok  =False
    while pa.parent is not None:
        ok, traj = solve(pa.H, n.H,obstacles=obstacles)
        if ok:
            n.g = pa.g +1
            n.H_traj = np.append(pa.H_traj, traj, axis=0)
            n.parent = pa
            return ok 
        pa = pa.parent
    return ok
# def get_point_cost(point, cost_map):
#     """
#     Takes a continuous 3D point, voxelizes it, and retrieves the cost 
#     from the generated cost_map.
    
#     Returns:
#         float: The cost if the point is reachable.
#         float('inf'): If the point is in an obstacle, out of bounds, or unreachable.
#     """
#     # 1. Discretize the point (Voxelize)
#     voxel_idx = (
#         math.floor(point[0] / COST_GRID_SIZE),
#         math.floor(point[1] / COST_GRID_SIZE),
#         math.floor(point[2] / COST_GRID_SIZE)
#     )
    
#     # 2. Retrieve cost
#     # We return infinity if the voxel isn't in our map
#     return cost_map.get(voxel_idx, float('inf'))

def get_point_cost(point, cost_map):
    """
    Retrieves the cost (step_count * grid_size).
    """
    # 1. Discretize
    voxel_idx = (
        math.floor(point[0] / COST_GRID_SIZE),
        math.floor(point[1] / COST_GRID_SIZE),
        math.floor(point[2] / COST_GRID_SIZE)
    )
    
    # 2. Retrieve steps (int)
    steps = cost_map.get(voxel_idx, None)
    
    if steps is None:
        return float('inf')
        
    # 3. Convert int steps to float cost
    return steps * COST_GRID_SIZE

root = node(np.array([LINK_LENGTH*NUM_LINKS/2,0,0,LINK_LENGTH*NUM_LINKS,0,0]), 0)
root.H = lift_to_high(root)

# goal = np.array([LINK_LENGTH*NUM_LINKS,0,0])
goal = np.array([0.3,-0.2,0.30])
# bounds = tuple((-BOUNDS,BOUNDS),(-BOUNDS,BOUNDS),(-BOUNDS,BOUNDS))
bounds = ((-BOUNDS,BOUNDS),(-BOUNDS,BOUNDS),(-BOUNDS,BOUNDS))

root.g = 0
root.H = lift_to_high(root)
root.H_traj = np.array([root.H])
# # a= np.load("node_data_in2.pkl", allow_pickle=True)
# # root = a
# root.parent = None
open_list = [root]
closed_list = []
found_path = False

obstacles = None


snake_thickness = 3  # Adjust this value as needed

actual_obs = np.array([
    # [-20, 6 - snake_thickness, 0, 0, 6 + snake_thickness, 10],
    # [-20, -6 - snake_thickness, 0, 0, -6 + snake_thickness, 10],
    [22 - snake_thickness, -35, 27, 22 + snake_thickness, -10, 37],
    [36 - snake_thickness, -35, 27, 36 + snake_thickness, -10, 37],
    # [44, 4 - snake_thickness, 45, 54, 4 + snake_thickness, 65],
    # [44, 16 - snake_thickness, 45, 54, 16 + snake_thickness, 65],
    [24.5, -35, 16, 35.5, -10, 27],   # 1st floor
    # [-30, -30, -5, 50, 50, 0],        # floor main
    # [-20, -5.5, 10, 0, 5.5, 11],      # zero roof
    [24.5, -35, 37, 35.5, -10, 38],   # 1st roof
    # [53, 4.5, 45, 54, 15.5, 65],
    # [44, 4.5, 45, 45, 15.5, 65]
])
obstacles = actual_obs/100

costs = generate_cost_field(goal,COST_GRID_SIZE,bounds,is_point_in_obstacle,obstacles)
root.h = calculate_h(root, costs)

# a= np.load("node_data.pkl", allow_pickle=True)

# is_collision_free(a.H_traj[-1], LINK_LENGTH, BASE_TF, obstacles)

global_best = 10000
global_best_node = None
for iter in tqdm.tqdm(range(15000)):
    if len(open_list) == 0:
        _save(current_node)
        print("No path found")
        break
    open_list.sort(key=lambda x: x.h)
    current_node = open_list.pop(0)
    closed_list.append(tuple(current_node.data))
    print(tuple(current_node.data))
    if calculate_ee_diff(current_node,goal) < GOAL_PROXIMITY:
        found_path = True
        print("Path found")
        _save(current_node)
        break
    if calculate_ee_diff(current_node,goal) < global_best:
        global_best = calculate_ee_diff(current_node,goal)
        global_best_node = current_node
        print("New best h:", global_best)
    neighbours = get_neighbours_less(current_node)
    print("*******",current_node.h,"*******")
    start_time_loop = time.perf_counter()
    for neighbour in neighbours:
        if tuple(neighbour.data) in closed_list:
            continue
        start_time = time.perf_counter()
        neighbour.H = lift_to_high(neighbour)
        if not is_collision_free(neighbour.H, LINK_LENGTH, BASE_TF, obstacles):
            continue
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        # print("IK time:", exec_time)
        start_time = time.perf_counter()
        ok, traj = solve(root.H, neighbour.H,obstacles=obstacles)
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        # print("Solve time:", exec_time)
        if ok:
            neighbour.g = current_node.g +1
            neighbour.H_traj = np.append(root.H_traj, traj, axis=0)
            if neighbour not in open_list:
                open_list.append(neighbour)
        else:
            ok = repair(neighbour,obstacles)
            if ok:
                if neighbour not in open_list:
                    open_list.append(neighbour)
            else:
                print("Could not repair", neighbour.node_id)
        _, ee = forward_kinematics(neighbour.H, LINK_LENGTH, BASE_TF)
        loc = ee[0:3,3]
        loc = np.round(loc, decimals=2)
        neighbour.data[3:6] = loc 
        neighbour.h = calculate_h(neighbour, costs)
        # _, ee = forward_kinematics(neighbour.H, LINK_LENGTH, BASE_TF)
        # loc = ee[0:3,3]
        # loc = np.round(loc, decimals=2)
        print(" h:", neighbour.h, " loc:", loc, " ee_diff:", calculate_ee_diff(neighbour,goal))
    end_time_loop = time.perf_counter()
    exec_time_loop = end_time_loop - start_time_loop
    # print("Loop time:", exec_time_loop)
print("no path found")
_save(global_best_node)