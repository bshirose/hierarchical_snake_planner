# 🐍 Snake Robot Motion Planning & Visualization Framework

This repository contains a hybrid **Python/MATLAB** framework designed for the path planning, dynamic verification, and visualization of a 16-DOF hyper-redundant (snake) robot.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| **`snake_planner.py`** | **[CORE]** The main planner. Implements a greedy best-first search in a latent Bezier space, maps to joint space via optimization-based IK, and verifies dynamics using RNEA. |
| **`snake_visualizer.m`** | **[RENDERER]** MATLAB script that ingests the raw planner output, visualizes the final state of the snake and exports a trajectory file compatible with vid_recorder. |
| **`vid_recorder.m`** | **[RENDERER]** MATLAB script for high-quality playback and `.avi` video recording of the final result. |

---

## 🚀 Key Features

### 1. Latent Space Planning
Instead of searching the high-dimensional 16-DOF joint space directly, the planner searches a reduced 6-dimensional **Latent Space**.
* **State Definition:** A Quadratic Bezier curve defined by a Control Point (3D) and an End Point (3D).
* **Mapping:** The algorithm "lifts" these Bezier curves to joint angles using `scipy.optimize.minimize` (SLSQP) to solve the Inverse Kinematics.

### 2. Physics-Based Verification (RNEA)
The planner includes a custom implementation of the **Recursive Newton-Euler Algorithm (RNEA)**.
* **Torque Limits:** Every edge in the search graph is checked for dynamic feasibility.
* **Constraint:** If the required torque to move between states exceeds **16 Nm** (configurable), the path is rejected.

### 3. Obstacle Avoidance
* **Environment:** The workspace is populated with cuboid obstacles.
* **Collision Check:** The planner discretizes the robot links and checks for intersections with obstacle bounds during the search.

---

## 🛠️ Prerequisites

### Python (For Planning)
* **Python 3.x**
* Required packages:
    ```bash
    pip install numpy scipy tqdm
    ```

### MATLAB (For Visualization)
* **MATLAB R2021b** or newer.
* *Toolboxes:* Navigation Toolbox or Robotics System Toolbox (required for `rotm2quat` and transformation functions).

---

## ⚙️ Usage Workflow

The workflow is sequential: **Plan (Python)** $\rightarrow$ **Process (MATLAB)** $\rightarrow$ **Render (MATLAB)**.

### Step 1: Generate the Path
Run the Python planner to calculate the collision-free, dynamically feasible path.

```bash
python snake_planner.py
