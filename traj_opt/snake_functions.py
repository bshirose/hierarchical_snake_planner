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




    # Time horizon and discretization
    T = 8.0
    dt = 0.05

    # Create and solve ddp_ee problem
    # You just need to call ddp_ee.CreateProblem once.
    # If x0 or eef is changed, you can pass the new value to ddp_ee.Solve(x0,eef)
    
    t = time.time()
    ok, xs, us, cost = ddp_ee.Solve(x0, eef, maxiter=10000)
    elapsed = time.time() - t

    # print("Solver converged:", ok)
    # print("Solver time:", elapsed, "s")

    # Extract joint trajectories
    ts = np.linspace(0.0, T, xs.shape[0])
    qs = xs[:, :nq]

    # Check terminal error
    pin.forwardKinematics(ddp_ee.model, ddp_ee.data, qs[-1])
    pin.updateFramePlacements(ddp_ee.model, ddp_ee.data)
    eef_a = ddp_ee.robot.data.oMf[ddp_ee.ee_frame_id]

    if np.linalg.norm(pin.SE3ToXYZQUAT(eef) - pin.SE3ToXYZQUAT(eef_a)) > 1e-2:
        ok = False
    return ok,qs, cost
        # print("Final state error too high:", np.linalg.norm(pin.SE3ToXYZQUAT(eef) - pin.SE3ToXYZQUAT(eef_a)))

def get_ee_pose(ddp_ee,q):
    pin.forwardKinematics(ddp_ee.model, ddp_ee.data, q)
    pin.updateFramePlacements(ddp_ee.model, ddp_ee.data)
    eef_a = ddp_ee.robot.data.oMf[ddp_ee.ee_frame_id]
    return eef_a
    # # Interpolate trajectory for smooth playback
    # t_interp = np.linspace(0.0, T, int(T * 60))
    # cs = CubicSpline(ts, qs, axis=0)
    # q_interp = cs(t_interp)

    # # Animated trajectory in Meshcat
    # ddp_ee.robot.setVisualizer(MeshcatVisualizer())
    # ddp_ee.robot.initViewer(open=True)
    # ddp_ee.robot.loadViewerModel()
    # ddp_ee.robot.play(q_interp, 1/60)