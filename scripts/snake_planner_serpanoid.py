import numpy as np
import heapq
import tqdm
import time
import pinocchio as pin
import math
import pickle
from copy import copy
import scipy.io
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from traj_opt.snake_functions import calculate_h, get_ee_problem, get_problem, do_traj_opt, get_ee_pose


def get_id():
    get_id.counter += 1
    return get_id.counter
get_id.counter = -1
class Node:
    """Internal helper class for search nodes."""
    def __init__(self, data, node_id):
        self.data = data  # Latent state: [alpha_y, alpha_z, omega, beta, delta_phase, gamma_y, gamma_z]
        self.parent = None
        self.children = []
        self.H = None     # Joint configuration (q)
        self.H_traj = []  # Trajectory from root to here
        self.g = 1000000        # Cost from start
        self.h = 0        # Heuristic cost
        self.f = 0        # Total cost
        self.node_id = node_id

    @property
    def priority(self):
        return self.h  # Priority based on heuristic (Greedy Best-First)

class SerpenoidKinematics3D:
    def __init__(self, num_links=15, link_length=0.05105, base_tf=None):
        self.num_links = num_links
        self.link_length = link_length
        self.base_tf = base_tf if base_tf is not None else np.eye(4)
        
    def get_joint_angles(self, params):
        """
        Maps 3D wave parameters to 16 joint angles.
        
        params: [alpha_y, alpha_z, omega, beta, delta_phase, gamma_y, gamma_z]
        
        alpha_y, alpha_z : Amplitudes for Pitch (odd) and Yaw (even)
        omega            : Spatial Frequency (Shared)
        beta             : Base Phase (moves the wave forward)
        delta_phase      : Phase Offset (Crucial for 3D shape/helix)
        gamma_y, gamma_z : Steering Biases
        """
        alpha_y, alpha_z, omega, beta, delta_phase, gamma_y, gamma_z = params
        q = np.zeros(self.num_links)
        
        for i in range(self.num_links):
            # Even joints (0, 2, 4...) = Z-axis / Yaw
            if i % 2 == 0:
                # Uses base phase 'beta'
                angle = alpha_z * np.sin(omega * i + beta) + gamma_z
                q[i] = angle
                
            # Odd joints (1, 3, 5...) = Y-axis / Pitch
            else:
                # Uses offset phase 'beta + delta_phase'
                # This offset allows for Helical/Corkscrew shapes
                angle = alpha_y * np.sin(omega * i + beta + delta_phase) + gamma_y
                q[i] = angle
                
        return q

    def forward_kinematics_ee(self, params):
        """Standard FK to get End Effector position from parameters."""
        q = self.get_joint_angles(params)
        T = self.base_tf.copy()
        
        for i in range(self.num_links):
            theta = q[i]
            c, s = np.cos(theta), np.sin(theta)
            
            # Rotation Logic
            if i % 2 == 0: # Z-axis (Yaw)
                R_mat = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            else: # Y-axis (Pitch)
                R_mat = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
                
            trans_x = np.array([[1, 0, 0, self.link_length], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            T = T @ R_mat @ trans_x
            
        return T[0:3, 3]

    def calculate_jacobian(self, params, epsilon=1e-5):
        """
        Computes Jacobian (sensitivity) of End Effector position w.r.t parameters.
        Returns J of shape (3, 7).
        """
        current_ee = self.forward_kinematics_ee(params)
        J = np.zeros((3, len(params)))
        
        for i in range(len(params)):
            p_perturbed = params.copy()
            p_perturbed[i] += epsilon
            ee_perturbed = self.forward_kinematics_ee(p_perturbed)
            J[:, i] = (ee_perturbed - current_ee) / epsilon
            
        return J
    
    def compute_parameter_update(self, current_params, goal_pos, step_size=0.1):
        """
        Returns the optimal parameter change to reach the goal using Damped Least Squares.
        """
        current_ee = self.forward_kinematics_ee(current_params)
        error_vec = goal_pos - current_ee
        
        J = self.calculate_jacobian(current_params)
        
        # Damped Least Squares
        damping = 0.01
        H = J @ J.T + damping * np.eye(3)
        inv_H = np.linalg.inv(H)
        delta_p = J.T @ inv_H @ error_vec
        
        # Clip step size
        norm = np.linalg.norm(delta_p)
        if norm > step_size:
            delta_p = delta_p / norm * step_size
            
        return current_params + delta_p, delta_p, np.linalg.norm(error_vec)
    
    def get_neighbors(self, node, goal_pos, kinematics_solver):
        """
        Generates neighboring nodes by perturbing Serpenoid parameters.
        
        Args:
            node: The current search node (contains node.params).
            goal_pos: Target [x, y, z] for the end effector.
            kinematics_solver: Instance of SerpenoidKinematics3D.
            
        Returns:
            List of neighbor Nodes.
        """
        neighbors = []
        current_params = node.data  # Assuming node.data holds the 9 params
        
        # --- STRATEGY 1: The "Smart" Neighbor (Exploitation) ---
        # Use the Jacobian to generate ONE neighbor that is mathematically 
        # guaranteed to be closer to the goal.
        # This acts like a "Gradient Descent" step within the A* search.
        try:
            guided_params, _, _ = kinematics_solver.compute_parameter_update(
                current_params, goal_pos, step_size=0.1
            )
            # Create node for this guided step
            n = Node(guided_params,get_id())
            n.parent = node
            node.children.append(n)
            neighbors.append(n)
        except np.linalg.LinAlgError:
            pass # Jacobian inversion failed (singular), skip guided step

        try:
            guided_params, _, _ = kinematics_solver.compute_parameter_update(
                current_params, goal_pos, step_size=0.05
            )
            # Create node for this guided step
            n = Node(guided_params,get_id())
            n.parent = node
            node.children.append(n)
            neighbors.append(n)
        except np.linalg.LinAlgError:
            pass # Jacobian inversion failed (singular), skip guided step

        # --- STRATEGY 2: Geometric Primitives (Exploration) ---
        # These are discrete actions that represent physical snake behaviors.
        
        actions = [
            # 1. LOCOMOTION (Slither Forward/Back)
            # Perturb Beta (Index 3)
            (3,  0.2), (3, -0.2), 
            
            # 2. STEERING (Bend the whole beam)
            # Perturb Gamma Y (Index 5) -> Pitch Up/Down
            (5,  0.08), (5, -0.08),
            # Perturb Gamma Z (Index 6) -> Yaw Left/Right
            (6,  0.08), (6, -0.08),
            
            # 3. RESIZING (Squeeze or Expand)
            # Perturb Amplitudes (Indices 0, 1)
            # Useful for squeezing through narrow gaps or reaching further
            (0,  0.05), (0, -0.05), # Change Vertical size
            (1,  0.05), (1, -0.05), # Change Horizontal size
            
            # 4. SHAPE SHIFTING (Twist)
            # Perturb Delta Phase (Index 4)
            # Changes shape from Sine Wave <-> Helix
            (4,  0.1), (4, -0.1),

            # 5. Change omega by a small amount
            (2,  0.02), (2, -0.02)
        ]
        
        for idx, delta in actions:
            new_params = current_params.copy()
            new_params[idx] += delta
            
            # --- VALIDITY CHECKS ---
            
            # 1. Amplitudes should not be negative (keeps phase logic consistent)
            if idx in [0, 1] and new_params[idx] < 0:
                new_params[idx] = 0.0
                
            # 2. Delta Phase normalization (keep between -pi and pi for cleanliness)
            if idx == 4: 
                new_params[idx] = (new_params[idx] + np.pi) % (2 * np.pi) - np.pi
            n = Node(new_params,get_id())
            n.parent = node
            node.children.append(n)
            neighbors.append(n)

        return neighbors

class SnakePlanner:
    """
    A planner for a snake robot that searches in a latent Bezier-curve space
    and maps to joint angles using Inverse Kinematics, while verifying 
    dynamic feasibility (torque limits) and collisions.
    """
    def __init__(self, 
                 num_links=15, 
                 link_length=0.05105, 
                 goal_proximity=0.04,
                 bounds=0.9,
                 params=[0,0,0,0,0,0,0]):
        
        # Robot Physical Parameters
        self.num_links = num_links
        self.link_length = link_length
        self.snake_length = num_links * link_length
        self.base_tf = np.eye(4)  # Assuming base rotation is identity for 

        self.serp_kin = SerpenoidKinematics3D(num_links=num_links, link_length=link_length, base_tf=self.base_tf)
        self.root_params = params


        self.ddp = get_problem()
        self.ddp_ee = get_ee_problem()

        self.goal_proximity = goal_proximity

    def plan(self, x_init, goal_pose, max_iter=15000):
        """
        Main planning method.
        
        Args:
            x_init (np.array): Initial joint angles.
            goal_pos (np.array): Target [x, y, z] for the end effector.
            obstacles (list): List of [xmin, ymin, zmin, xmax, ymax, zmax].
            max_iter (int): Maximum iterations for the search.
            
        Returns:
            np.array: The trajectory of joint angles (H_traj) if found, else None.
        """
        self.goal_pose =goal_pose

        root = Node(self.root_params, get_id())
        root.H = x_init if x_init is not None else self._lift_to_high(root)
         
        root.H_traj = np.array([root.H])
        root.h = self._calculate_h(root)
        root.g = 0
        
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
                pbar.set_postfix({"Best Dist": f"{global_best_dist:.3f}"})

            neighbors = self.serp_kin.get_neighbors(current_node, goal_pose.translation, self.serp_kin)

            for neighbor in neighbors:
                neighbor_sig = tuple(np.round(neighbor.data, 2))
                if neighbor_sig in closed_set:
                    continue

                neighbor.H = self._lift_to_high(neighbor)
                
                ok, traj,us,cost = do_traj_opt(self.ddp,root.H[:self.ddp_ee.nq], neighbor.H[:self.ddp_ee.nq])
                
                if ok:
                    neighbor.g = root.g + cost
                    neighbor.H_traj = np.append(root.H_traj, traj, axis=0)
                    if neighbor not in open_list:
                        open_list.append(neighbor)
                else:
                    # Repair strategy: try connecting to parent's parent
                    repaired_ok = self._repair(neighbor)
                    if repaired_ok and neighbor not in open_list:
                        open_list.append(neighbor)
                    else:
                        continue

                print(self._calculate_ee_diff(neighbor))
                neighbor.h = self._calculate_h(neighbor)

            pbar.update(1)

        pbar.close()
        print(f"Max iterations reached. Best distance: {global_best_dist}")
        return best_node.H_traj

    def _lift_to_high(self, node):
        return self.serp_kin.get_joint_angles(node.data)


    def _repair(self, node):
        """Recursively tries to connect to ancestors if immediate parent fails."""
        ancestor = node.parent
        ancestors=[]

        while ancestor is not None:
            ancestors.append(ancestor)
            ancestor = ancestor.parent
        # flip ancestors
        ancestors.reverse()
        for a in ancestors:
            ok, traj,us,cost = do_traj_opt(self.ddp,a.H[:self.ddp_ee.nq], node.H[:self.ddp_ee.nq])
            if ok:
                node.g = a.g + cost
                node.H_traj = np.append(a.H_traj, traj, axis=0)
                return True
        return False

    def _calculate_h(self, node):
        """Heuristic function: Lookup in cost map."""
        ok,qs, cost = calculate_h(self.ddp_ee,node.H[:self.ddp_ee.nq],self.goal_pose)
        return cost

    def _calculate_ee_diff(self, node):
        ee_pose = get_ee_pose(self.ddp_ee, node.H[:self.ddp_ee.nq])
        
        return np.linalg.norm(ee_pose.translation - self.goal_pose.translation)


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

# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":

    rot = np.array([[ 0.99979102,  0.00507982,  0.01980204],
                    [-0.00532436,  0.99990998,  0.012316  ],
                    [-0.01973769, -0.01241886,  0.99972806]])

    pos = np.array([-0.39559905, -0.33490934,  0.26886563])
    serp_kin = SerpenoidKinematics3D()
    params = [0.0, 0.0, 2*np.pi/15, 0, np.pi/2, 0.0, 0.0]
    q_init = serp_kin.get_joint_angles(params)
    x_init= np.zeros(30)
    x_init[:15]=q_init
    goal = pin.SE3(rot,pos)
    
    # 3. Run Planner
    planner = SnakePlanner(num_links=15, link_length=0.05105, params=params,goal_proximity=0.055)
    path = planner.plan(x_init=x_init, goal_pose=goal)
    
    if path is not None:
        print(f"Success! Path length: {len(path)} steps")
        scipy.io.savemat('my_data_2.mat', {'my_matrix': path})
        np.save('serp_dat_1.npy', path)

    else:
        print("Failed to find path.")