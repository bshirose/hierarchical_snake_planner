from ddp_problem_setting import SnakeReachingDDP
import pinocchio as pin
import numpy as np
import time

from scipy.interpolate import CubicSpline
from pinocchio.visualize import MeshcatVisualizer
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from snake_functions import add_square_tunnel
# Initialize ddp_ee problem with snake robot model
# if you provide the end-effector frame name, the DDP problem will be set for end-effector reaching
ddp_ee = SnakeReachingDDP(
    urdf_model_path=current_dir+"/REU_snake_v8.urdf",
    srdf_model_path=current_dir+"/REU_snake_v8.srdf",
    package_dirs=current_dir+"/",
    ee_frame_name="reu_module_out_16")  
center = np.array([-0.29, -0.3, 0.3]) 
direction = np.array([1, 0, 0]) # Tunnel runs along Y axis

add_square_tunnel(
    ddp_ee,
    center_pos=center,
    orientation_axis=direction,
    width=0.1,        # 8 cm gap (33 - 25)
    height=0.12,       # 10 cm gap (37 - 27)
    length=0.25,       # 25 cm long
    wall_thick=0.02    # 2 cm thick
)

xs = np.load("debug_traj_2.npy")
nq = ddp_ee.nq
T = xs.shape[0] * 0.05

ts = np.linspace(0.0, T, xs.shape[0])
qs = xs[:, :nq]

t_interp = np.linspace(0.0, T, int(T * 60))
cs = CubicSpline(ts, qs, axis=0)
q_interp = cs(t_interp)

# Animated trajectory in Meshcat
ddp_ee.robot.setVisualizer(MeshcatVisualizer())
ddp_ee.robot.initViewer(open=True)
ddp_ee.robot.loadViewerModel()
ddp_ee.robot.play(q_interp, 1/60)