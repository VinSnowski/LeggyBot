# NewLeggyBot simulation model

`urdf/NewLeggyBot.urdf` is the CAD-derived robot description. Its ROS
`package://NewLeggyBot` mesh paths are resolved by ROS-aware tools and by the
URDF inspection viewer.

`mjcf/NewLeggyBot.xml` is the authoritative MuJoCo model used by the fixed-ankle
walking task. It matches `../LeggyBot/mjcf/LeggyBot.xml` in solver settings,
floating base, base/hip/thigh bodies, joint limits and dynamics, actuators,
non-foot primitive collisions, sensors, and visualization. Its only mechanical
departure is the separate shin and smaller foot connected by the CAD-derived
fixed angled ankle.

Open it with:

```bash
uv run python -m locomotion.view_cad_robot \
  locomotion/NewLeggyBot/mjcf/NewLeggyBot.xml --fixed-base
```
