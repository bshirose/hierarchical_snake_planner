from ddp_problem_setting import SnakeReachingDDP
import pinocchio as pin
import numpy as np
import time

from scipy.interpolate import CubicSpline
from pinocchio.visualize import MeshcatVisualizer
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Initialize ddp_ee problem with snake robot model
# if you provide the end-effector frame name, the DDP problem will be set for end-effector reaching
ddp_ee = SnakeReachingDDP(
    urdf_model_path=current_dir+"/REU_snake_v8.urdf",
    srdf_model_path=current_dir+"/REU_snake_v8.srdf",
    package_dirs=current_dir+"/",
    ee_frame_name="reu_module_out_16")  

xs = np.load("serp_dat_1.npy")
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