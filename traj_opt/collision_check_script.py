from ddp_problem_setting import SnakeReachingDDP
import pinocchio as pin
import numpy as np

from scipy.interpolate import CubicSpline
from pinocchio.visualize import MeshcatVisualizer

# Initialize DDP problem with snake robot model
ddp = SnakeReachingDDP(
    urdf_model_path="REU_snake_v8.urdf",
    srdf_model_path="REU_snake_v8.srdf",
    package_dirs="./")

ddp.AddWallSet(
    position=pin.SE3(pin.utils.rotate('z',np.pi/2), np.zeros(3)),
    size=np.array([0.5, 0.05, 0.1]),
    gap=0.075,
)

ddp.AddWallSet(
    position=pin.SE3(np.eye(3), np.array([0.25, -0.5, 0.25])),
    size=np.array([0.5, 0.05, 0.1]),
    gap=0.075,
)

# q_test = pin.neutral(ddp.model)
q_test = np.random.rand(ddp.model.nq) * 2 * np.pi - np.pi  # Random configuration
collision_datas, n_collision = ddp.collisionCheck(q_test)

if n_collision > 0:
    print(f"Test configuration in collision with {n_collision} objects.")
else:
    print("Test configuration collision-free.")

# Animated trajectory in Meshcat
ddp.robot.setVisualizer(MeshcatVisualizer())
ddp.robot.initViewer(open=True)
ddp.robot.loadViewerModel()
ddp.robot.display(q_test)