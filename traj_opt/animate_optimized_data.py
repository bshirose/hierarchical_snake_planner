from ddp_problem_setting import SnakeReachingDDP
import pinocchio as pin
import numpy as np
import time

from scipy.interpolate import CubicSpline
from pinocchio.visualize import MeshcatVisualizer
import os
import sys

import meshcat.animation
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from snake_functions import add_square_tunnel, get_tunnel_wall_aabbs, optimize_trajectory_until_collision
# Initialize ddp_ee problem with snake robot model
# if you provide the end-effector frame name, the DDP problem will be set for end-effector reaching
ddp_ee = SnakeReachingDDP(
    urdf_model_path=current_dir+"/REU_snake_v8.urdf",
    srdf_model_path=current_dir+"/REU_snake_v8.srdf",
    package_dirs=current_dir+"/")  
# ddp_ee = SnakeReachingDDP(
#     urdf_model_path=current_dir+"/REU_snake_v8.urdf",
#     srdf_model_path=current_dir+"/REU_snake_v8.srdf",
#     package_dirs=current_dir+"/",
#     ee_frame_name="reu_module_out_16")  
center = np.array([0.2, -0.2, 0.2])  
# center = np.array([0.3, -0.2, 0.2])  
direction = np.array([0, 1, 0]) # Tunnel runs along Y axis


nq = ddp_ee.nq
nv = ddp_ee.nv
nu = ddp_ee.nu
weights = {
"x_reg": np.array(1.0 * np.ones(nq + nv)),
"link_poses_reg": [np.hstack((
    np.array([1e1]*3),
    np.array([0]*3),
    )),] * 15 + [np.hstack((
    np.array([1e1]*3),
    np.array([0.5]*3),
    ))],
"u_reg": np.array(0.1 * np.ones(nu)),
"x_reg_terminal": 1e-2,
"link_poses_reg_terminal": [1e-1] *16,
"u_reg_terminal": 1e-2,
}
xs_seed = np.load("data_torq_limts.npy")

# T = xs.shape[0] * 0.01
T_sec = (xs_seed.shape[0] - 1) * 0.01
ddp_ee.CreateProblem(dt=0.01, T=T_sec, weights=weights)
add_square_tunnel(  
    ddp_ee,
    center_pos=center,
    orientation_axis=direction,
    width=0.08,        
    height=0.08,      
    length=0.3,       
    wall_thick=0.04    
)
print("Trajectory duration (s):", T_sec)
ok, xs_new, us_new, cost = optimize_trajectory_until_collision(ddp_ee, xs_seed)
ts = np.linspace(0.0, T_sec, xs_new.shape[0])
qs = xs_new[:, :nq]

t_interp = np.linspace(0.0, T_sec, int(T_sec * 60))
cs = CubicSpline(ts, qs, axis=0)
q_interp = cs(t_interp)

# # Animated trajectory in Meshcat
ddp_ee.robot.setVisualizer(MeshcatVisualizer())
ddp_ee.robot.initViewer(open=True)
ddp_ee.robot.loadViewerModel()
ddp_ee.robot.play(q_interp, 1/60)