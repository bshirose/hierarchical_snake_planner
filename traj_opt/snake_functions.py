import pinocchio as pin
import numpy as np
import time

from scipy.interpolate import CubicSpline
from pinocchio.visualize import MeshcatVisualizer

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ddp_problem_setting import SnakeReachingDDP

def get_problem(dt=0.05, T=8.0):
    ddp = SnakeReachingDDP(
    urdf_model_path=current_dir+"/REU_snake_v8.urdf",
    srdf_model_path=current_dir+"/REU_snake_v8.srdf",
    package_dirs=current_dir+"/")  
    nq = ddp.nq
    nv = ddp.nv
    nu = ddp.nu
    # Cost weights

    weights = {
        "x_reg": np.array(1.0 * np.ones(nq + nv)),
        "ee_reg": np.array(1.0 * np.ones(6)),
        "u_reg": np.array(0.1 * np.ones(nu)),
        "x_reg_terminal": 1e-2,
        "ee_reg_terminal": 1e-2,
        "u_reg_terminal": 1e-2,
    }
    ddp.CreateProblem(dt=dt, weights=weights, T=T)
    return ddp




def get_ee_problem(dt=0.05, T=8.0):
    ddp_ee = SnakeReachingDDP(
            urdf_model_path=current_dir+"/REU_snake_v8.urdf",
            srdf_model_path=current_dir+"/REU_snake_v8.srdf",
            package_dirs=current_dir+"/",
            ee_frame_name="reu_module_out_16")  
    # Dimensions
    nq = ddp_ee.nq
    nv = ddp_ee.nv
    nu = ddp_ee.nu
    # Cost weights

    weights = {
        "x_reg": np.array(1.0 * np.ones(nq + nv)),
        "ee_reg": np.array(1.0 * np.ones(6)),
        "u_reg": np.array(0.1 * np.ones(nu)),
        "x_reg_terminal": 1e-2,
        "ee_reg_terminal": 1e-2,
        "u_reg_terminal": 1e-2,
    }
    ddp_ee.CreateProblem(dt=dt, weights=weights, T=T)
    return ddp_ee

def do_traj_opt(ddp,q_init,q_f):
    q0 = q_init.copy()
    v0 = np.zeros(ddp.model.nv)
    x0 = np.concatenate([q0, v0])

    vf = np.zeros(ddp.model.nv)
    xf = np.concatenate([q_f, vf])

    ok, xs, us, cost = ddp.Solve(x0, xf, maxiter=10000)
    
    return ok,xs,us,cost


def calculate_h(ddp_ee,q_init,goal):
    q0 = q_init.copy()
    v0 = np.zeros(ddp_ee.model.nv)
    x0 = np.concatenate([q0, v0])

    # Dimensions
    nq = ddp_ee.nq
    nv = ddp_ee.nv
    nu = ddp_ee.nu

    eef = goal.copy()

    T = 8.0
    dt = 0.05

    t = time.time()
    ok, xs, us, cost = ddp_ee.Solve(x0, eef, maxiter=10000)
    elapsed = time.time() - t

    ts = np.linspace(0.0, T, xs.shape[0])
    qs = xs[:, :nq]

    # Check terminal error
    pin.forwardKinematics(ddp_ee.model, ddp_ee.data, qs[-1])
    pin.updateFramePlacements(ddp_ee.model, ddp_ee.data)
    eef_a = ddp_ee.robot.data.oMf[ddp_ee.ee_frame_id]

    if np.linalg.norm(pin.SE3ToXYZQUAT(eef) - pin.SE3ToXYZQUAT(eef_a)) > 1e-2:
        ok = False
    return ok,qs, cost

def get_ee_pose(ddp_ee,q):
    pin.forwardKinematics(ddp_ee.model, ddp_ee.data, q)
    pin.updateFramePlacements(ddp_ee.model, ddp_ee.data)
    eef_a = ddp_ee.robot.data.oMf[ddp_ee.ee_frame_id]
    return eef_a




def add_square_tunnel(problem, center_pos, orientation_axis, width, height, length, wall_thick):
    """
    Adds a rectangular tunnel composed of 4 walls to the problem.
    
    Args:
        problem: The problem instance (e.g., snake_problem)
        center_pos: np.array [x, y, z] of the tunnel center
        orientation_axis: np.array [x, y, z] direction the tunnel runs along
        width: Inner width of the tunnel (gap between left/right walls)
        height: Inner height of the tunnel (gap between top/bottom walls)
        length: Length of the tunnel
        wall_thick: Thickness of the walls
    """
    
    # 1. Normalize orientation and compute base rotation
    # We want a rotation matrix R_tunnel that aligns the Local Y axis with orientation_axis.
    # We assume 'Z' is up to resolve the roll ambiguity.
    axis = orientation_axis / np.linalg.norm(orientation_axis)
    target_y = axis
    
    # Heuristic to find a valid 'Right' (X) vector
    # If axis is vertical, use X as temporary up, otherwise use Z
    if np.abs(target_y[2]) > 0.99: 
        tmp_up = np.array([1, 0, 0])
    else:
        tmp_up = np.array([0, 0, 1])
        
    # Construct base frame (Local Y = Tunnel Axis)
    target_x = np.cross(target_y, tmp_up)
    target_x /= np.linalg.norm(target_x)
    target_z = np.cross(target_x, target_y)
    
    # R_tunnel maps local [x,y,z] to global orientation
    R_tunnel = np.column_stack((target_x, target_y, target_z))

    # --- 2. Create Side Walls (Left/Right) ---
    # Logic: AddWallSet makes gap along Local Y. 
    # To put gap along Tunnel Width (Tunnel Local X), we rotate 90 deg around Z relative to R_tunnel.
    # Side walls should be slightly taller (height + 2*thick) to overlap corners perfectly.
    
    R_sides = R_tunnel @ pin.utils.rotate('z', np.pi/2)
    pos_sides = pin.SE3(R_sides, center_pos)
    
    # Size vector depends on how AddWallSet interprets it. 
    # Usually: [Wall_Length_Along_Gap, Wall_Thickness, Wall_Height]
    # After rotation: Local X -> Tunnel Y (Length), Local Z -> Tunnel Z (Height)
    size_sides = np.array([length, wall_thick, height + 2*wall_thick])
    
    problem.AddWallSet(
        position=pos_sides,
        size=size_sides,
        gap=width
    )

    # --- 3. Create Top/Bottom Walls ---
    # Logic: To put gap along Tunnel Height (Tunnel Local Z), we rotate -90 deg around X relative to R_tunnel.
    
    R_tb = R_tunnel @ pin.utils.rotate('x', -np.pi/2)
    pos_tb = pin.SE3(R_tb, center_pos)
    
    # Size: Local X -> Tunnel X (Width), Local Z -> Tunnel Y (Length)
    # Note: Width does not need padding if sides were padded, but padding both is safer for physics engines.
    size_tb = np.array([width, wall_thick, length])
    
    problem.AddWallSet(
        position=pos_tb,
        size=size_tb,
        gap=height
    )


