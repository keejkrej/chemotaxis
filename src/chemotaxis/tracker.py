"""
Core cell tracking functionality using ultrack and cellpose.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from cellpose.models import CellposeModel
from ultrack import config, track


class CellTracker:
    """
    A cell tracking system combining Cellpose for segmentation and UlTrack for tracking.

    This class handles the complete workflow:
    1. Load image data
    2. Segment cells using Cellpose
    3. Track cells across frames using UlTrack
    """

    def __init__(
        self,
        model_type: str = "cyto2",
        device: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize the CellTracker.

        Parameters
        ----------
        model_type : str
            Cellpose model type ("cyto", "cyto2", "nuclei", etc.)
        device : Optional[str]
            Device to use ("cpu" or "gpu"). If None, auto-detect.
        use_gpu : bool
            Whether to attempt using GPU acceleration
        """
        self.model_type = model_type
        self.device = device
        self.use_gpu = use_gpu

        # Initialize Cellpose model
        self.model = CellposeModel(
            gpu=use_gpu,
            model_type=model_type,
        )

    def segment_frame(
        self,
        image: np.ndarray,
        diameter: Optional[float] = None,
        channels: List[int] = [0, 0],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment cells in a single frame using Cellpose.

        Parameters
        ----------
        image : np.ndarray
            Input image (2D or 3D)
        diameter : Optional[float]
            Expected cell diameter in pixels. If None, auto-estimate.
        channels : List[int]
            Channels to use for segmentation [cytoplasm_channel, nuclei_channel]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Segmentation masks, flows, and styles
        """
        masks, flows, styles = self.model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
        return masks, flows, styles

    def segment_timelapse(
        self,
        images: np.ndarray,
        diameter: Optional[float] = None,
        channels: List[int] = [0, 0],
    ) -> np.ndarray:
        """
        Segment cells across a timelapse sequence.

        Parameters
        ----------
        images : np.ndarray
            Stack of images with shape (T, H, W) or (T, C, H, W)
        diameter : Optional[float]
            Expected cell diameter in pixels
        channels : List[int]
            Channels to use for segmentation

        Returns
        -------
        np.ndarray
            Segmentation masks for all frames with shape (T, H, W)
        """
        n_frames = images.shape[0]
        segmentations = []

        for i in range(n_frames):
            frame = images[i]
            masks, _, _ = self.segment_frame(frame, diameter, channels)
            segmentations.append(masks)

        return np.array(segmentations)

    def track(
        self,
        segmentations: np.ndarray,
        config_dict: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Track segmented cells across frames using UlTrack.

        Parameters
        ----------
        segmentations : np.ndarray
            Segmentation masks with shape (T, H, W)
        config_dict : Optional[dict]
            UlTrack configuration parameters

        Returns
        -------
        np.ndarray
            Tracked labels with shape (T, H, W)
        """
        # Use default configuration if not provided
        if config_dict is None:
            config_dict = {
                "max_distance": 50,
                "max_gap": 5,
                "min_track_length": 10,
            }

        # Configure UlTrack
        ultrack_config = config.Config(**config_dict)

        # Run tracking
        tracked = track(segmentations, ultrack_config)

        return tracked

    def process_timelapse(
        self,
        images: np.ndarray,
        diameter: Optional[float] = None,
        channels: List[int] = [0, 0],
        track_config: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: segment and track cells across timelapse.

        Parameters
        ----------
        images : np.ndarray
            Stack of images with shape (T, H, W) or (T, C, H, W)
        diameter : Optional[float]
            Expected cell diameter in pixels
        channels : List[int]
            Channels to use for segmentation
        track_config : Optional[dict]
            UlTrack configuration parameters

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Segmentation masks and tracked labels
        """
        print("Segmenting cells...")
        segmentations = self.segment_timelapse(images, diameter, channels)

        print("Tracking cells...")
        tracked = self.track(segmentations, track_config)

        return segmentations, tracked

    def load_images(self, image_path: Path) -> np.ndarray:
        """
        Load image file.

        Parameters
        ----------
        image_path : Path
            Path to image file (TIFF supported)

        Returns
        -------
        np.ndarray
            Loaded image data
        """
        import tifffile
        return tifffile.imread(image_path)

    def save_intermediate_results(
        self,
        output_dir: Path,
        segmentations: np.ndarray,
        tracked: np.ndarray,
    ) -> None:
        """
        Save segmentation and tracking masks as intermediate files.

        Parameters
        ----------
        output_dir : Path
            Directory to save results
        segmentations : np.ndarray
            Segmentation masks
        tracked : np.ndarray
            Tracked labels
        """
        import tifffile

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tifffile.imwrite(output_dir / "segmentations.tiff", segmentations.astype(np.uint16))
        tifffile.imwrite(output_dir / "tracked.tiff", tracked.astype(np.uint16))

        print(f"Intermediate results saved to {output_dir}")

    def extract_and_export_tracks(
        self,
        tracked: np.ndarray,
        output_dir: Path,
        formats: List[str] = ["csv", "json"],
        min_track_length: int = 3,
    ) -> None:
        """
        Extract cell tracks as (x, y) coordinates and export in multiple formats.

        Parameters
        ----------
        tracked : np.ndarray
            Tracked labels with shape (T, H, W)
        output_dir : Path
            Directory to save track exports
        formats : List[str]
            Output formats: "csv", "json", "summary"
        min_track_length : int
            Minimum number of frames for a track to be included
        """
        from .tracks import TrackExtractor, TrackExporter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Extracting cell trajectories...")
        trajectories = TrackExtractor.extract_trajectories(
            tracked,
            min_track_length=min_track_length,
        )

        print(f"Extracted {len(trajectories)} cell tracks")

        # Export in requested formats
        if "csv" in formats:
            TrackExporter.to_csv(trajectories, output_dir / "tracks.csv")

        if "json" in formats:
            TrackExporter.to_json(
                trajectories,
                output_dir / "tracks.json",
                include_statistics=True,
            )

        if "summary" in formats:
            TrackExporter.to_summary_csv(
                trajectories,
                output_dir / "tracks_summary.csv",
            )
