from ddp_problem_setting import SnakeReachingDDP
import pinocchio as pin
import numpy as np
import time
from scipy.interpolate import CubicSpline
from pinocchio.visualize import MeshcatVisualizer
import os
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from snake_functions import add_square_tunnel

# 1. Setup Data
filename = 'segmented_paths.pkl'
with open(filename, 'rb') as file:
    data = pickle.load(file)

# --- Global Visualizer Setup ---
# Initialize the visualizer outside the function
viz = MeshcatVisualizer()
# We start the viewer once here
viz.initViewer(open=True)

def setup_and_animate(segment_idx, visualizer):
    seg = data["segs"][segment_idx]
    
    # Initialize DDP for this segment
    ddp = SnakeReachingDDP(
        urdf_model_path=current_dir+"/REU_snake_v8.urdf",
        srdf_model_path=current_dir+"/REU_snake_v8.srdf",
        package_dirs=current_dir+"/",
        base_placement=seg["base"]
    )

    # Add obstacles to this DDP instance
    if data["obs"] is not None:
        for ob in data["obs"]:
            add_square_tunnel(ddp, ob[0], ob[1],
                              width=data["obs_width"], height=data["obs_height"],
                              length=data["obs_length"], wall_thick=data["wall_thick"])

    # Attach the robot model and geometry to the ALREADY running visualizer
    visualizer.rebuildData() # Clean slate for the new segment geometry
    ddp.robot.setVisualizer(visualizer)
    ddp.robot.loadViewerModel()

    # Prepare Trajectory
    xs = seg["path"][:, :16]
    nq = ddp.nq
    dt = 0.05 
    T = xs.shape[0] * dt
    ts = np.linspace(0.0, T, xs.shape[0])
    qs = xs[:, :nq]

    t_interp = np.linspace(0.0, T, int(T * 60))
    cs = CubicSpline(ts, qs, axis=0)
    q_interp = cs(t_interp)

    print(f"Playing Segment {segment_idx}...")
    ddp.robot.play(q_interp, 1/60)
    
    return ddp

# --- Execution ---
# Play Segment 0
ddp1 = setup_and_animate(0, viz)

# Give it a second to breathe
time.sleep(2) 

# Play Segment 1 (The viewer stays open, but geometry updates)
ddp2 = setup_and_animate(1, viz)