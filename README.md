# Chemotaxis

A Python package for automated cell detection and tracking in microscopy images using Cellpose and UlTrack.

## Features

- **Cell Segmentation**: Uses Cellpose for high-accuracy cell detection
- **Cell Tracking**: Leverages UlTrack for multi-frame cell tracking
- **Timelapse Support**: Process complete timelapse sequences
- **Flexible Configuration**: Customizable parameters for tracking and segmentation
- **CLI Interface**: Easy-to-use command-line tools

## Installation

### Prerequisites

- Mamba or Conda with the `bioimaging` environment that includes ultrack and cellpose

### Install Chemotaxis

```bash
cd /path/to/chemotaxis
pip install -e .
```

Or for development with additional tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Using the CLI

#### Track cells in a timelapse:
```bash
chemotaxis track input.tiff -o ./results -m cyto2
```

#### Segment cells only:
```bash
chemotaxis segment input.tiff -o segmentation.tiff
```

#### View available options:
```bash
chemotaxis track --help
chemotaxis segment --help
chemotaxis info
```

### Using the Python API

```python
from chemotaxis import CellTracker
import numpy as np

# Initialize tracker
tracker = CellTracker(model_type="cyto2", use_gpu=True)

# Load images
images = tracker.load_images("timelapse.tiff")  # Shape: (T, H, W) or (T, C, H, W)

# Segment and track
segmentations, tracked = tracker.process_timelapse(
    images,
    diameter=None,  # Auto-estimate cell diameter
    track_config={
        "max_distance": 50,
        "max_gap": 5,
        "min_track_length": 10,
    }
)

# Save results
tracker.save_results("./results", segmentations, tracked)
```

## Configuration

### Cellpose Models
- `cyto`: General cytoplasm segmentation
- `cyto2`: Improved cytoplasm segmentation
- `nuclei`: Nuclear segmentation

### Tracking Parameters
- `max_distance`: Maximum linking distance between cells (pixels)
- `max_gap`: Maximum number of frames to bridge tracking gaps
- `min_track_length`: Minimum number of frames required for a valid track

## Project Structure

```
chemotaxis/
├── src/chemotaxis/
│   ├── __init__.py          # Package initialization
│   ├── tracker.py           # Core CellTracker class
│   └── cli.py              # Command-line interface
├── main.py                 # Example usage script
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── data/                   # Sample data directory
```

## Dependencies

- ultrack >=0.5.0
- cellpose >=3.0.0
- numpy >=1.21.0
- scipy >=1.7.0
- scikit-image >=0.18.0
- click >=8.0.0
- tifffile >=2022.0.0

## Development

### Running tests
```bash
pytest
```

### Code formatting
```bash
black src/
isort src/
```

### Type checking
```bash
mypy src/
```

## License

MIT License

## Citation

If you use this package, please cite the underlying libraries:
- Cellpose: Stringer et al., Nature Methods (2021)
- UlTrack: Research paper [add citation when available]
