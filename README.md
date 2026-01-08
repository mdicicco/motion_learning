# Motion Learning

Generate articulated scenes, render images, and compute motion deltas for machine learning applications.

## Overview

This project creates synthetic training data by:
1. **Generating random MJCF scenes** with articulated objects (spheres, cylinders, boxes, capsules)
2. **Rendering** the initial scene state
3. **Perturbing** joint positions (including free, hinge, slide, and ball joints)
4. **Rendering** the perturbed state
5. **Computing motion delta** images using optical flow and frame differencing

## Installation

This project uses [Pixi](https://prefix.dev/docs/pixi/overview) for environment management.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install
```

## Usage

### Generate Data

Run a generation (default: 50 scenes):

```bash
pixi run generate
```

With custom parameters:

```bash
# Generate 100 scenes with seed for reproducibility
pixi run python src/main.py -n 100 -s 42

# Custom object/body ranges (flat distribution)
pixi run python src/main.py --min-objects 1 --max-objects 3 --min-bodies 2 --max-bodies 4

# Adjust perturbation intensity
pixi run python src/main.py -p 2.0
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `data` | Base output directory |
| `--num-scenes` | `-n` | 50 | Number of scenes to generate |
| `--min-objects` | | 1 | Minimum objects per scene |
| `--max-objects` | | 5 | Maximum objects per scene |
| `--min-bodies` | | 1 | Minimum bodies per object (each object randomized independently) |
| `--max-bodies` | | 5 | Maximum bodies per object (each object randomized independently) |
| `--perturbation-scale` | `-p` | 1.0 | Scale factor for perturbations |
| `--seed` | `-s` | None | Random seed for reproducibility |

### Output Structure

Each run creates a date-stamped folder with sequential scene folders:

```
data/
└── 20260107_143052/              # Date-stamped run folder
    ├── run_metadata.json         # Run configuration and all scene info
    ├── 0000/                     # Scene 0000
    │   ├── 0000_scene.xml        # MJCF scene definition
    │   ├── 0000_original.png     # Initial scene render
    │   ├── 0000_perturbed.png    # After joint perturbation
    │   └── 0000_motion_delta.png # Motion visualization
    ├── 0001/                     # Scene 0001
    │   └── ...
    └── 0049/                     # Scene 0049 (last of 50)
        └── ...
```

## Scene Generation Details

### Object Types
Objects are generated with random:
- **Shapes**: sphere, cylinder, box, capsule
- **Colors**: Vibrant HSV-based random colors
- **Positions**: Grid-distributed with random articulation chains

### Joint Types
- **Free**: Full 6-DOF motion (position + rotation)
- **Hinge**: Single-axis rotation
- **Slide**: Single-axis translation (prismatic)
- **Ball**: 3-DOF rotation (spherical)

### Fixed vs Free Base
Each object randomly has either:
- **Fixed base**: First body is welded to world (no base joint)
- **Free base**: First body has a free joint allowing full motion

## Motion Detection

The motion delta is computed using:
1. **Optical Flow**: Farneback dense optical flow (direction = hue, magnitude = brightness)
2. **Frame Difference**: Absolute pixel difference with enhancement
3. **Combined**: Blended visualization of both methods

## Development

Run individual components:

```bash
# Generate scene only
pixi run python src/scene_generator.py

# Render existing scene
pixi run python src/renderer.py

# Run tests
pixi run pytest
```

## Dependencies

- **MuJoCo** (>=3.1): Physics simulation and rendering
- **NumPy** (>=1.26): Numerical operations
- **OpenCV** (>=4.9): Motion detection and image processing
- **Pillow** (>=10.0): Image I/O

## License

MIT

