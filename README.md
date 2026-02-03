# RoboSimKit

RoboSimKit is a lightweight simulator for learning robotics.  
Inspired by [FunMoRo_control](https://github.com/TUNI-IINES/FunMoRo_control), which was originally developed by **Made Widhi Surya Atman** from the Intelligent Networked Systems (IINES) Group, Tampere University, this RoboSimKit is extended version for easy learning platform.

## Project Structure

```
RoboSimKit/
├── examples/          # Demo scripts
│   ├── demo_go_to_goal.py
│   ├── demo_follow_figure8.py
│   ├── demo_dwa_unicycle.py
│   └── demo_template.py
├── controllers/       # Control laws
│   ├── __init__.py
│   ├── go_to_goal.py
│   ├── sim_utils.py
│   ├── trajectory_generator.py
│   └── (more controllers...)
├── models/            # Robot kinematic models
│   ├── ackermann.py
│   ├── unicycle.py
│   └── omni.py
├── utils/             # Sim utilities
│   └── angles.py
├── visualization/     # Plotting utilities
│   ├── plotter2d.py
│   └── robot_icon.py
└── pyproject.toml
```

## Running the Demos

From **any location**, just run the example file directly:

```bash
python examples/demo_go_to_goal.py
```


The template standardizes:

- **Input parameters** (MODE, Ts, t_max, init_state, goal)
- **Visualization** using `visualization/plotter2d.py`
- **History logging** (`state_hist`, `goal_hist`, `u_hist`)
- **Plots** (control inputs vs time, and states vs time)


## Example: Go-to-goal (demo_go_to_goal.py)

This demo simulates a robot moving from an initial pose to a goal pose using a simple go-to-goal controller.

### 1) Run it

```bash
python examples/demo_go_to_goal.py
```

### 2) Pick the robot type (MODE)

Open `examples/demo_go_to_goal.py` and set:

```python
MODE = "unicycle"          # or "omnidirectional" or "ackermann"
```

What each mode means:

- `"unicycle"`: control is `[v, omega]`
- `"ackermann"`: control is `[v, delta]` (car-like steering)
- `"omnidirectional"`: control is `[vx, vy, omega]` in the WORLD frame

If `MODE` is not one of these, the script raises a `ValueError`.

### 3) Set scenario + simulation parameters

In the same file you can edit:

- `init_state = [px, py, theta]`
- `goal_state = [px, py, theta]`
- `Ts` (time step) and `t_max` (simulation duration)
- `SHOW_2D` (turn live 2D animation on/off)

At the end, the script plots:

- the control inputs vs time (`u_hist`)
- the state trajectory vs time (`state_hist`) together with the constant goal (`goal_hist`)

## Configuration

Edit the top of the example scripts to switch between robot types:

```python
MODE = "unicycle"          # or "omnidirectional" or "ackermann"
```

## Notes on Models vs Controllers

- `models/` contains the kinematic update for some robotic models.
- `controllers/` contains control input (e.g., go-to-goal).
- `controllers/go_to_goal.py` simple control input for go to goal using all defined models.
- `controllers/trajectory_generator.py` contains reference generators (e.g., `figure8_goal`).
- `controllers/sim_utils.py` contains demo/simulation helpers 


## Author

**Muhamad Fikri**  
Tampere University  
muhamad.fikri@tuni.fi
