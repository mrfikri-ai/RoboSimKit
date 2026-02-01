# RoboSimKit

RoboSimKit is a lightweight simulator for learning robotics.  
Inspired by [FunMoRo_control](https://github.com/TUNI-IINES/FunMoRo_control), originally developed by **Made Widhi Surya Atman** from the Intelligent Networked Systems (IINES) Group, Tampere University.

## Project Structure

```
RoboSimKit/
├── examples/          # Demo scripts
│   └── demo_go_to_goal.py
├── models/            # Robot kinematic models
│   ├── unicycle.py
│   └── omni.py
├── visualization/     # Plotting utilities
│   ├── plotter2d.py
│   └── robot_icon.py
└── pyproject.toml
```

## Running the Demo

From **any location**, just run the example file directly:

```bash
python examples/demo_go_to_goal.py
```

Or in VS Code: open `examples/demo_go_to_goal.py` and click **Run Python File**.

## Configuration

Edit the top of `demo_go_to_goal.py` to switch between robot types:

```python
MODE = "unicycle"          # or "omnidirectional" or "ackermann"
```

## Author

**Muhamad Fikri**  
Tampere University  
muhamad.fikri@tuni.fi
