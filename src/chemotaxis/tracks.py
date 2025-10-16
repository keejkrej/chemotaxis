"""
Cell track extraction and analysis utilities.

Converts segmentation and tracking masks into usable track data
(cell trajectories as (x, y) coordinates).
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import csv
import numpy as np
from scipy import ndimage


class TrackExtractor:
    """Extract and analyze cell trajectories from tracked segmentations."""

    @staticmethod
    def get_centroids_per_frame(segmentation_frame: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """
        Extract centroid (x, y) for each cell in a single frame.

        Parameters
        ----------
        segmentation_frame : np.ndarray
            2D segmentation mask for a single frame (H, W)

        Returns
        -------
        Dict[int, Tuple[float, float]]
            Mapping of cell_id -> (x, y) centroid coordinates
        """
        centroids = {}
        cell_ids = np.unique(segmentation_frame)
        cell_ids = cell_ids[cell_ids != 0]  # Exclude background

        for cell_id in cell_ids:
            mask = segmentation_frame == cell_id
            if mask.sum() == 0:
                continue

            # Calculate centroid
            coords = np.argwhere(mask)  # Returns (row, col) = (y, x)
            y_center = coords[:, 0].mean()
            x_center = coords[:, 1].mean()
            centroids[int(cell_id)] = (float(x_center), float(y_center))

        return centroids

    @staticmethod
    def get_cell_statistics(segmentation_frame: np.ndarray) -> Dict[int, Dict]:
        """
        Extract detailed statistics for each cell in a frame.

        Parameters
        ----------
        segmentation_frame : np.ndarray
            2D segmentation mask for a single frame

        Returns
        -------
        Dict[int, Dict]
            Mapping of cell_id -> {centroid, area, perimeter, ...}
        """
        stats = {}
        cell_ids = np.unique(segmentation_frame)
        cell_ids = cell_ids[cell_ids != 0]

        for cell_id in cell_ids:
            mask = segmentation_frame == cell_id
            if mask.sum() == 0:
                continue

            coords = np.argwhere(mask)
            y_coords, x_coords = coords[:, 0], coords[:, 1]

            # Basic statistics
            area = mask.sum()
            y_center, x_center = y_coords.mean(), x_coords.mean()

            # Perimeter approximation
            perimeter = np.sum(ndimage.binary_gradient(mask))

            # Bounding box
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            stats[int(cell_id)] = {
                "centroid": (float(x_center), float(y_center)),
                "area": int(area),
                "perimeter": int(perimeter),
                "bbox": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                },
            }

        return stats

    @staticmethod
    def extract_trajectories(
        tracked_frames: np.ndarray,
        min_track_length: int = 3,
    ) -> Dict[int, List[Tuple[int, float, float]]]:
        """
        Extract continuous cell trajectories from tracked segmentation.

        Parameters
        ----------
        tracked_frames : np.ndarray
            3D array of tracked frames (T, H, W)
        min_track_length : int
            Minimum number of frames a track must appear in

        Returns
        -------
        Dict[int, List[Tuple[int, float, float]]]
            Mapping of track_id -> [(frame_idx, x, y), ...] sorted by frame
        """
        trajectories: Dict[int, List[Tuple[int, float, float]]] = {}

        n_frames = tracked_frames.shape[0]

        # Collect centroids for each track ID across all frames
        for frame_idx in range(n_frames):
            frame = tracked_frames[frame_idx]
            centroids = TrackExtractor.get_centroids_per_frame(frame)

            for cell_id, (x, y) in centroids.items():
                if cell_id not in trajectories:
                    trajectories[cell_id] = []
                trajectories[cell_id].append((frame_idx, x, y))

        # Filter by minimum track length
        filtered = {
            track_id: points
            for track_id, points in trajectories.items()
            if len(points) >= min_track_length
        }

        return filtered

    @staticmethod
    def compute_track_statistics(trajectory: List[Tuple[int, float, float]]) -> Dict:
        """
        Compute statistics for a single trajectory.

        Parameters
        ----------
        trajectory : List[Tuple[int, float, float]]
            Track points [(frame_idx, x, y), ...]

        Returns
        -------
        Dict
            Statistics including distance, speed, direction, etc.
        """
        if len(trajectory) < 2:
            return {}

        frames = np.array([t[0] for t in trajectory])
        coords = np.array([(t[1], t[2]) for t in trajectory])

        # Calculate displacements
        displacements = np.diff(coords, axis=0)
        distances = np.linalg.norm(displacements, axis=1)

        # Frame intervals
        frame_diffs = np.diff(frames)

        # Speed (pixels per frame)
        speeds = distances / frame_diffs

        # Total displacement
        total_displacement = np.linalg.norm(coords[-1] - coords[0])

        # Track length
        total_distance = distances.sum()

        # Direction (angle from start to end)
        if total_displacement > 0:
            direction = np.arctan2(
                coords[-1][1] - coords[0][1],
                coords[-1][0] - coords[0][0],
            )
        else:
            direction = None

        return {
            "n_frames": len(trajectory),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "start_position": (float(coords[0][0]), float(coords[0][1])),
            "end_position": (float(coords[-1][0]), float(coords[-1][1])),
            "total_displacement": float(total_displacement),
            "total_distance": float(total_distance),
            "mean_speed": float(speeds.mean()),
            "max_speed": float(speeds.max()),
            "min_speed": float(speeds.min()),
            "direction_angle_rad": float(direction) if direction is not None else None,
        }


class TrackExporter:
    """Export cell tracks in various formats."""

    @staticmethod
    def to_csv(
        trajectories: Dict[int, List[Tuple[int, float, float]]],
        output_path: Path,
    ) -> None:
        """
        Export trajectories to CSV format.

        CSV columns: track_id, frame, x, y

        Parameters
        ----------
        trajectories : Dict
            Mapping of track_id -> [(frame_idx, x, y), ...]
        output_path : Path
            Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "frame", "x", "y"])

            for track_id in sorted(trajectories.keys()):
                for frame_idx, x, y in trajectories[track_id]:
                    writer.writerow([track_id, frame_idx, f"{x:.2f}", f"{y:.2f}"])

        print(f"Exported {len(trajectories)} tracks to {output_path}")

    @staticmethod
    def to_json(
        trajectories: Dict[int, List[Tuple[int, float, float]]],
        output_path: Path,
        include_statistics: bool = True,
    ) -> None:
        """
        Export trajectories to JSON format with optional statistics.

        Parameters
        ----------
        trajectories : Dict
            Mapping of track_id -> [(frame_idx, x, y), ...]
        output_path : Path
            Output file path
        include_statistics : bool
            Whether to include track statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "tracks": {},
            "statistics": {},
        }

        for track_id in sorted(trajectories.keys()):
            trajectory = trajectories[track_id]

            # Convert to list of dicts for JSON
            track_data = [
                {"frame": int(frame_idx), "x": float(x), "y": float(y)}
                for frame_idx, x, y in trajectory
            ]

            data["tracks"][str(track_id)] = track_data

            if include_statistics:
                stats = TrackExtractor.compute_track_statistics(trajectory)
                data["statistics"][str(track_id)] = stats

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(trajectories)} tracks to {output_path}")

    @staticmethod
    def to_summary_csv(
        trajectories: Dict[int, List[Tuple[int, float, float]]],
        output_path: Path,
    ) -> None:
        """
        Export summary statistics for each track as CSV.

        One row per track with statistics.

        Parameters
        ----------
        trajectories : Dict
            Mapping of track_id -> [(frame_idx, x, y), ...]
        output_path : Path
            Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for track_id in sorted(trajectories.keys()):
            stats = TrackExtractor.compute_track_statistics(trajectories[track_id])
            stats["track_id"] = track_id
            rows.append(stats)

        if not rows:
            print("No tracks to export")
            return

        # Get all keys
        fieldnames = ["track_id"] + [k for k in rows[0].keys() if k != "track_id"]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported statistics for {len(rows)} tracks to {output_path}")
