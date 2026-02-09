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
ddp_bb = SnakeReachingDDP(
    urdf_model_path=current_dir+"/REU_snake_v8.urdf",
    srdf_model_path=current_dir+"/REU_snake_v8.srdf",
    package_dirs=current_dir+"/",
    link_frame_names=[
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
        "reu_module_out_16"])  

# Load desired link frame locations
frame_locations = np.load(current_dir+"/frame_locations.npy", allow_pickle=True)
frame_locations = np.load("my_locs.npy", allow_pickle=True)
xf = [None] * 16
for i in range(16):
    xf[i] = pin.SE3(
        np.eye(3),
        frame_locations[i]
    )

# Initial configuration and velocity (you should change it to your x0)
q0 = pin.neutral(ddp_bb.model)
v0 = np.zeros(ddp_bb.model.nv)
x0 = np.concatenate([q0, v0])

# Dimensions
nq = ddp_bb.nq
nv = ddp_bb.nv
nu = ddp_bb.nu

# Cost weights
weights = {
    "x_reg": np.array(1.0 * np.ones(nq + nv)),
    # "link_poses_reg": [np.array(1.0 * np.ones(6))] * 16,
    "link_poses_reg": [np.hstack((
        np.array([0]*3),
        np.array([1e1]*3),
        )),] * 16,
    "u_reg": np.array(0.1 * np.ones(nu)),
    "x_reg_terminal": 1e-2,
    "link_poses_reg_terminal": [1e-1] *16,
    "u_reg_terminal": 1e-2,
}

# Time horizon and discretization
T = 8.0
dt = 0.05

# Create and solve ddp_ee problem
# You just need to call ddp_bb.CreateProblem once.
# If x0 or xf is changed, you can pass the new value to ddp_bb.Solve(x0,xf)
ddp_bb.CreateProblem(dt=dt, weights=weights, T=T)
t = time.time()
ok, xs, us,cost = ddp_bb.Solve(x0, xf, maxiter=10000)
elapsed = time.time() - t

print("Solver converged:", ok)
print("Solver time:", elapsed, "s")

# Extract joint trajectories
ts = np.linspace(0.0, T, xs.shape[0])
qs = xs[:, :nq]

# Check terminal error
pin.forwardKinematics(ddp_bb.model, ddp_bb.data, qs[-1])
pin.updateFramePlacements(ddp_bb.model, ddp_bb.data)
for i in range(16):
    xf_a = ddp_bb.robot.data.oMf[ddp_bb.link_frame_ids[i]]

    if np.linalg.norm(pin.SE3ToXYZQUAT(pin.SE3(xf[i])) - pin.SE3ToXYZQUAT(xf_a)) > 1e-3:
        print("Final state error too high:", np.linalg.norm(pin.SE3ToXYZQUAT(xf[i]) - pin.SE3ToXYZQUAT(xf_a)))

# Interpolate trajectory for smooth playback
t_interp = np.linspace(0.0, T, int(T * 60))
cs = CubicSpline(ts, qs, axis=0)
q_interp = cs(t_interp)

# Animated trajectory in Meshcat
ddp_bb.robot.setVisualizer(MeshcatVisualizer())
ddp_bb.robot.initViewer(open=True)
ddp_bb.robot.loadViewerModel()
ddp_bb.robot.play(q_interp, 1/60)