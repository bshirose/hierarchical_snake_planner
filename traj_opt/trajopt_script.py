from ddp_problem_setting import SnakeReachingDDP
import pinocchio as pin
import numpy as np
import time

from scipy.interpolate import CubicSpline
from pinocchio.visualize import MeshcatVisualizer

# Initialize DDP problem with snake robot model
ddp = SnakeReachingDDP(
    urdf_model_path="./traj_opt/REU_snake_v8.urdf",
    srdf_model_path="./traj_opt/REU_snake_v8.srdf",
    package_dirs="./traj_opt/")

# Initial configuration and velocity (you should change it to your x0)
q0 = pin.neutral(ddp.model)
v0 = np.zeros(ddp.model.nv)
x0 = np.concatenate([q0, v0])

# Dimensions
nq = ddp.nq
nv = ddp.nv
nu = ddp.nu

# Final configuration and velocity (you should change it to your xf)
qf = q0.copy()
qf[0::2] = np.pi/4
qf[1::2] = 0
vf = v0.copy()
xf = np.concatenate([qf, vf])

# Cost weights
weights = {
    "x_reg": np.array(1.0 * np.ones(nq + nv)),
    "u_reg": np.array(0.1 * np.ones(nu)),
    "x_reg_terminal": 1e-2,
    "u_reg_terminal": 1e-2,
}

# Time horizon and discretization
T = 8.0
dt = 0.05

# Create and solve DDP problem
# You just need to call ddp.CreateProblem once.
# If x0 or xf is changed, you can pass the new value to ddp.Solve(x0,xf)
ddp.CreateProblem(dt=dt, weights=weights, T=T)
t = time.time()
ok, xs, us = ddp.Solve(x0, xf, maxiter=10000)
elapsed = time.time() - t

print("Solver converged:", ok)
print("Solver time:", elapsed, "s")

# Extract joint trajectories
ts = np.linspace(0.0, T, xs.shape[0])
qs = xs[:, :nq]

# Check terminal error
if np.linalg.norm(qs[-1, :nq] - qf) > 1e-3:
    print("Final state error too high:", np.linalg.norm(qs[-1, :nq] - qf))

# Interpolate trajectory for smooth playback
t_interp = np.linspace(0.0, T, int(T * 60))
cs = CubicSpline(ts, qs, axis=0)
q_interp = cs(t_interp)

# Animated trajectory in Meshcat
ddp.robot.setVisualizer(MeshcatVisualizer())
ddp.robot.initViewer(open=True)
ddp.robot.loadViewerModel()
ddp.robot.play(q_interp, 1/60)