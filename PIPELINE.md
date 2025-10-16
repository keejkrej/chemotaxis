# Complete Chemotaxis Cell Tracking Pipeline

## Overview

The chemotaxis pipeline provides a complete end-to-end solution for cell tracking from raw microscopy images. The pipeline:

1. **Loads image sequences** (JPEG, PNG, TIFF from a directory or single file)
2. **Segments cells** using Cellpose deep learning model
3. **Tracks cells** across frames using UlTrack
4. **Extracts trajectories** as (x, y) coordinate sequences
5. **Exports results** in multiple formats (CSV, JSON, summary statistics)

## Output Files

The pipeline generates three categories of output:

### 1. Intermediate Segmentation Files (Optional)

- **segmentations.tiff**: Label masks from Cellpose showing individual cell boundaries
  - Format: 3D array (T, H, W) as uint16
  - Values: 0 = background, 1-N = cell IDs per frame
  - Use: Verify segmentation quality

- **tracked.tiff**: Cell identities across time
  - Format: 3D array (T, H, W) as uint16
  - Values: Cell IDs consistent across frames
  - Use: Verify tracking quality, understand cell movements

### 2. Track Data Files (Primary Output)

#### tracks.csv
Complete trajectory data - all coordinates for all cells

```
track_id,frame,x,y
1,0,245.32,512.10
1,1,247.89,515.23
1,2,250.15,518.45
2,0,123.45,234.56
2,1,124.12,236.78
...
```

**Usage**: Import into Python/R for custom analysis, plotting trajectories, computing metrics

#### tracks.json
Full structured data with track statistics

```json
{
  "tracks": {
    "1": [
      {"frame": 0, "x": 245.32, "y": 512.10},
      {"frame": 1, "x": 247.89, "y": 515.23},
      {"frame": 2, "x": 250.15, "y": 518.45}
    ],
    "2": [...]
  },
  "statistics": {
    "1": {
      "n_frames": 47,
      "start_frame": 0,
      "end_frame": 46,
      "start_position": [245.32, 512.10],
      "end_position": [342.18, 548.92],
      "total_displacement": 102.45,
      "total_distance": 247.32,
      "mean_speed": 5.25,
      "max_speed": 12.34,
      "min_speed": 0.45,
      "direction_angle_rad": 0.523
    },
    "2": {...}
  }
}
```

**Usage**: Complete track data with computed statistics in one file

#### tracks_summary.csv
One row per track with summary statistics

```
track_id,n_frames,start_frame,end_frame,start_position,end_position,total_displacement,total_distance,mean_speed,max_speed,min_speed,direction_angle_rad
1,47,0,46,"(245.32, 512.10)","(342.18, 548.92)",102.45,247.32,5.25,12.34,0.45,0.523
2,33,2,34,"(123.45, 234.56)","(189.23, 301.45)",86.12,156.78,4.75,9.87,0.12,1.234
...
```

**Usage**: Quick overview, sorting/filtering tracks by statistics, spreadsheet analysis

## Usage Examples

### Command Line - Track cells from image sequence

```bash
chemotaxis batch-track /path/to/images -o ./results \
  --max-distance 100 \
  --max-gap 3 \
  --min-track-length 5
```

**Parameters**:
- `--max-distance`: Maximum pixel distance to link cells between frames (default: 50)
- `--max-gap`: Maximum frames to bridge tracking gaps (default: 5)
- `--min-track-length`: Minimum frames for a track to be included (default: 10)
- `--no-gpu`: Use CPU instead of GPU
- `--model`: Cellpose model ("cyto", "cyto2", "nuclei")
- `--diameter`: Expected cell diameter (auto-estimate if omitted)

### Command Line - Segment only

```bash
chemotaxis segment /path/to/image.tiff -o segmentation.tiff
```

### Python API

```python
from chemotaxis import CellTracker
import numpy as np

# Initialize tracker
tracker = CellTracker(model_type="cyto2", use_gpu=True)

# Load images (shape: T, H, W or T, C, H, W)
images = tracker.load_images("timelapse.tiff")

# Segment and track
segmentations, tracked = tracker.process_timelapse(
    images,
    diameter=None,  # Auto-estimate
    track_config={
        "max_distance": 100,
        "max_gap": 3,
        "min_track_length": 5,
    }
)

# Save intermediate results
tracker.save_intermediate_results("./results", segmentations, tracked)

# Extract and export tracks
tracker.extract_and_export_tracks(
    tracked,
    "./results",
    formats=["csv", "json", "summary"],
    min_track_length=5
)

# Or manually access track data
from chemotaxis.tracks import TrackExtractor, TrackExporter

trajectories = TrackExtractor.extract_trajectories(tracked, min_track_length=5)
TrackExporter.to_csv(trajectories, "./tracks.csv")
```

## Understanding Track Statistics

For each track, the pipeline computes:

- **n_frames**: Number of frames the cell appears in
- **start_frame** / **end_frame**: First and last frame indices
- **start_position** / **end_position**: (x, y) coordinates at beginning and end
- **total_displacement**: Straight-line distance from start to end (pixels)
- **total_distance**: Sum of all movement between consecutive frames (pixels)
- **mean_speed**: Average pixels per frame
- **max_speed** / **min_speed**: Fastest and slowest frame-to-frame movement
- **direction_angle_rad**: Angle of overall movement (radians, -π to π)

## Interpreting Results

### High Quality Tracks
- Long `n_frames` (many timepoints)
- `total_distance` > `total_displacement` (non-straight paths = active tracking)
- `direction_angle_rad` consistent within populations (directional migration)

### Potential Issues
- Short `n_frames`: Cell left field of view or divided
- `total_distance` ≈ `total_displacement`: Low directional persistence
- Few tracks: Check `--min-track-length` (may be too high)
- Broken tracks: Increase `--max-gap` to bridge detection gaps
- Merged tracks: Decrease `--max-distance` to prevent over-linking

## File Formats

### CSV
- Simple, human-readable
- Easy to import into Python/R/MATLAB
- Good for programmatic analysis

### JSON
- Includes computed statistics
- Structured, machine-readable
- Good for web applications and data archiving

## Example Analysis Workflow

```python
import pandas as pd
import numpy as np

# Load track data
tracks_df = pd.read_csv("tracks.csv")
summary_df = pd.read_csv("tracks_summary.csv")

# Filter by track length
long_tracks = summary_df[summary_df["n_frames"] > 30]

# Calculate migration persistence
summary_df["directional_persistence"] = (
    summary_df["total_displacement"] / summary_df["total_distance"]
)

# Plot speed distribution
import matplotlib.pyplot as plt
plt.hist(summary_df["mean_speed"], bins=20)
plt.xlabel("Mean Speed (pixels/frame)")
plt.ylabel("Number of Cells")
plt.show()

# Analyze specific track
track_1_data = tracks_df[tracks_df["track_id"] == 1]
positions = track_1_data[["x", "y"]].values
displacements = np.diff(positions, axis=0)
distances = np.linalg.norm(displacements, axis=1)
print(f"Track 1 speed profile: {distances}")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Few/no cells detected | Increase `--diameter`, check image quality, try different model |
| Cells merge in tracking | Decrease `--max-distance` |
| Tracks break frequently | Increase `--max-gap` |
| Too many short tracks | Increase `--min-track-length` |
| GPU out of memory | Use `--no-gpu` for CPU processing |
| Processing is slow | Reduce image size or use `--no-gpu` on machines with slow CPU/GPU transfer |

## Performance

- **Segmentation**: ~5-10 sec per frame (GPU), ~30-60 sec per frame (CPU) on 2048×1532 images
- **Tracking**: ~0.5-1 sec per frame for ~100 cells
- **Track extraction & export**: <1 sec for ~1000 cells

Total time for 145 images with GPU: ~15-20 minutes
