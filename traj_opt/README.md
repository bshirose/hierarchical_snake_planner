# Snake Robot Reaching & Trajectory Optimization (DDP)

This project provides a **framework** for trajectory optimization and collision checking of a **snake robot** using **Pinocchio** and **Crocoddyl**.

## Features

* **DDP-based trajectory optimization**
  * State / configuration tracking
  * End-effector pose reaching
* **Collision modeling and checking**
  * Self-collision handling via URDF/SRDF
  * Environment obstacles (wall sets)

## Code Structure

* `ddp_problem_setting.py`
  Defines the DDP problem (costs, constraints, collision handling, solver setup)
* `trajopt_script.py`
  Example of state-based DDP trajectory optimization
* `trajopt_ee_script.py`
  Example of end-effector pose reaching using DDP
* `collision_check_script.py`
  Collision checking for arbitrary robot configurations
* `REU_snake_v8.urdf / REU_snake_v8.srdf`
  Snake robot model files
* `ReU.STL`
  Mesh geometry for visualization

## Dependencies

* Python
* `pinocchio`
* `crocoddyl`
* `hppfcl`
* `mim_solvers`
* `meshcat`
좋은 포인트네요. 아래는 **자연스럽고 연구용 README에 잘 맞는 영어 문장**으로 교정해서 추가한 버전입니다.

## Notes

* To implement both **neighbor cost** and **heuristic cost**, define two separate DDP problems: `ddp` and `ddp_ee`, and solve them independently using `ddp.solve()` and `ddp_ee.solve()`.
* When initializing `SnakeReachingDDP`, providing `ee_frame_name` automatically enables the **end-effector tracking mode** (`ddp_ee`).
