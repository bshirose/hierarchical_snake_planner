from ddp_problem_setting import SnakeReachingDDP
import pinocchio as pin
import numpy as np
import time

from scipy.interpolate import CubicSpline
from pinocchio.visualize import MeshcatVisualizer

# Initialize ddp_ee problem with snake robot model
# if you provide the end-effector frame name, the DDP problem will be set for end-effector reaching
ddp_ee = SnakeReachingDDP(
    urdf_model_path="REU_snake_v8.urdf",
    srdf_model_path="REU_snake_v8.srdf",
    package_dirs="./",
    ee_frame_name="reu_module_out_16")  

# Initial configuration and velocity (you should change it to your x0)
q0 = pin.neutral(ddp_ee.model)
v0 = np.zeros(ddp_ee.model.nv)
x0 = np.concatenate([q0, v0])

# Dimensions
nq = ddp_ee.nq
nv = ddp_ee.nv
nu = ddp_ee.nu

# Final configuration and velocity (you should change it to your xf)
qf = q0.copy()
qf[0::2] = np.pi/4
qf[1::2] = 0

pin.forwardKinematics(ddp_ee.model, ddp_ee.data, qf)
pin.updateFramePlacements(ddp_ee.model, ddp_ee.data)
eef = ddp_ee.data.oMf[ddp_ee.ee_frame_id]

# Cost weights
weights = {
    "x_reg": np.array(1.0 * np.ones(nq + nv)),
    "ee_reg": np.array(1.0 * np.ones(6)),
    "u_reg": np.array(0.1 * np.ones(nu)),
    "x_reg_terminal": 1e-2,
    "ee_reg_terminal": 1e-2,
    "u_reg_terminal": 1e-2,
}

# Time horizon and discretization
T = 8.0
dt = 0.05

# Create and solve ddp_ee problem
# You just need to call ddp_ee.CreateProblem once.
# If x0 or eef is changed, you can pass the new value to ddp_ee.Solve(x0,eef)
ddp_ee.CreateProblem(dt=dt, weights=weights, T=T)
t = time.time()
ok, xs, us = ddp_ee.Solve(x0, eef, maxiter=10000)
elapsed = time.time() - t

print("Solver converged:", ok)
print("Solver time:", elapsed, "s")

# Extract joint trajectories
ts = np.linspace(0.0, T, xs.shape[0])
qs = xs[:, :nq]

# Check terminal error
pin.forwardKinematics(ddp_ee.model, ddp_ee.data, qs[-1])
pin.updateFramePlacements(ddp_ee.model, ddp_ee.data)
eef_a = ddp_ee.robot.data.oMf[ddp_ee.ee_frame_id]

if np.linalg.norm(pin.SE3ToXYZQUAT(eef) - pin.SE3ToXYZQUAT(eef_a)) > 1e-3:
    print("Final state error too high:", np.linalg.norm(pin.SE3ToXYZQUAT(eef) - pin.SE3ToXYZQUAT(eef_a)))

# Interpolate trajectory for smooth playback
t_interp = np.linspace(0.0, T, int(T * 60))
cs = CubicSpline(ts, qs, axis=0)
q_interp = cs(t_interp)

# Animated trajectory in Meshcat
ddp_ee.robot.setVisualizer(MeshcatVisualizer())
ddp_ee.robot.initViewer(open=True)
ddp_ee.robot.loadViewerModel()
ddp_ee.robot.play(q_interp, 1/60)