# RoboSimKit

RoboSimKit is a lightweight simulator for learning robotics.  
Inspired by [FunMoRo_control](https://github.com/TUNI-IINES/FunMoRo_control), originally developed by **Made Widhi Surya Atman** from the Intelligent Networked Systems (IINES) Group, Tampere University.

## Project Structure

```
RoboSimKit/
├── examples/          # Demo scripts
│   ├── demo_go_to_goal.py
│   └── demo_follow_figure8.py
├── controllers/       # Control laws
│   └── go_to_goal.py
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

Figure-8 trajectory tracking (reference path + robot trajectory):

```bash
python examples/demo_follow_figure8.py
```

Or in VS Code: open `examples/demo_go_to_goal.py` and click **Run Python File**.

## Go-to-goal (demo_go_to_goal.py)

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
- `controllers/` contains control laws used by examples (e.g., go-to-goal).
- `controllers/go_to_goal.py` defines `controller_unicycle`, `controller_omni`, and `controller_ackermann`. The examples import them via `controllers/__init__.py`.
- Angle wrapping is handled explicitly in the simulation scripts (FunMoRo_control style) and uses `utils/angles.py`. The model `step()` functions also support optional wrapping via a `wrap` flag, but the default is not to wrap.

## Tuning

In the figure-8 example you can tune feedback gains in one place:
The basic control implementation for figure-8 is control + feedforward

```python
K_POS = 1.2
K_THETA = 3.0
```

## Author

**Muhamad Fikri**  
Tampere University  
muhamad.fikri@tuni.fi
