# RoboSimKit

RoboSimKit is a lightweight simulator for learning robotics.  
Inspired by [FunMoRo_control](https://github.com/TUNI-IINES/FunMoRo_control), originally developed by **Made Widhi Surya Atman** from the Intelligent Networked Systems (IINES) Group, Tampere University.

## Project Structure

```
RoboSimKit/
├── examples/          # Demo scripts
│   ├── demo_go_to_goal.py
│   └── demo_follow_figure8.py
├── controllers/       # Control laws (kept separate from models)
│   └── go_to_goal.py
├── models/            # Robot kinematic models
│   ├── ackermann.py
│   ├── unicycle.py
│   └── omni.py
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

## Configuration

Edit the top of the example scripts to switch between robot types:

```python
MODE = "unicycle"          # or "omnidirectional" or "ackermann"
```

## Notes on Models vs Controllers

- `models/` contains the kinematic update functions (`step`). They do not clip/cap inputs.
- `controllers/` contains control laws used by examples (e.g., go-to-goal).
- Angle wrapping is handled explicitly in the simulation scripts (FunMoRo_control style). The model `step()` functions support optional wrapping via a `wrap` flag, but the default is not to wrap.

## Tuning

In the figure-8 example you can tune feedback gains in one place:

```python
K_POS = 1.2
K_THETA = 3.0
```

## Author

**Muhamad Fikri**  
Tampere University  
muhamad.fikri@tuni.fi
