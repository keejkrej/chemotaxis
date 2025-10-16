"""
Core cell tracking functionality using ultrack and cellpose.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import logging
import typer
from cellpose.models import CellposeModel
from ultrack import config, track

# Setup logger
logger = logging.getLogger(__name__)


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
            Channels parameter (kept for backward compatibility, not used in v4.0.1+)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Segmentation masks, flows, and styles
        """
        logger.info(f"Processing frame with shape: {image.shape}, dtype: {image.dtype}")
        logger.debug(f"Image value range: [{image.min()}, {image.max()}]")

        # Cellpose v4.0.1+: don't pass channels parameter to avoid deprecation warning
        masks, flows, styles = self.model.eval(
            image,
            diameter=diameter,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )

        logger.info(f"Segmentation complete: found {masks.max()} cells, mask shape: {masks.shape}")
        logger.debug(f"Mask value range: [{masks.min()}, {masks.max()}], mask dtype: {masks.dtype}")

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
        logger.info(f"Starting segmentation of {n_frames} frames")
        logger.info(f"Input images shape: {images.shape}, dtype: {images.dtype}")

        segmentations = []

        for i in range(n_frames):
            logger.info(f"Processing frame {i+1}/{n_frames}")
            frame = images[i]
            logger.debug(f"Frame {i} shape: {frame.shape}, dtype: {frame.dtype}")

            masks, _, _ = self.segment_frame(frame, diameter, channels)
            segmentations.append(masks)

            logger.debug(f"Frame {i} segmentation complete: {masks.max()} cells detected")

        result = np.array(segmentations)
        logger.info(f"Segmentation complete. Output shape: {result.shape}, dtype: {result.dtype}")
        logger.info(f"Total cells across all frames: min={result.min()}, max={result.max()}")

        return result

    def _sanitize_segmentations(self, segmentations: np.ndarray) -> np.ndarray:
        """
        Sanitize segmentation masks to ensure valid cell IDs.

        Parameters
        ----------
        segmentations : np.ndarray
            Segmentation masks with shape (T, H, W)

        Returns
        -------
        np.ndarray
            Sanitized segmentation masks
        """
        logger.info("Sanitizing segmentation masks...")
        sanitized = np.copy(segmentations)

        for frame_idx in range(sanitized.shape[0]):
            frame = sanitized[frame_idx]
            unique_ids = np.unique(frame)
            unique_ids = unique_ids[unique_ids != 0]  # Exclude background

            logger.debug(f"Frame {frame_idx}: {len(unique_ids)} unique cell IDs, range: [{unique_ids.min() if len(unique_ids) > 0 else 0}, {unique_ids.max() if len(unique_ids) > 0 else 0}]")

            # Relabel to ensure continuous IDs starting from 1
            if len(unique_ids) > 0:
                new_frame = np.zeros_like(frame)
                for new_id, old_id in enumerate(unique_ids, start=1):
                    mask = frame == old_id
                    new_frame[mask] = new_id
                sanitized[frame_idx] = new_frame

                logger.debug(f"Frame {frame_idx} relabeled: new ID range [1, {len(unique_ids)}]")

        logger.info(f"Sanitization complete. New range: [{sanitized.min()}, {sanitized.max()}]")
        return sanitized

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
            UlTrack configuration parameters (max_distance, max_gap, min_track_length)

        Returns
        -------
        np.ndarray
            Tracked labels with shape (T, H, W)
        """
        logger.info(f"Starting tracking with segmentation shape: {segmentations.shape}")
        logger.info(f"Segmentation dtype: {segmentations.dtype}, value range: [{segmentations.min()}, {segmentations.max()}]")

        # Validate segmentation data
        if segmentations.shape[0] == 0:
            logger.error("No frames in segmentation array")
            raise ValueError("Segmentation array is empty")

        if segmentations.max() == 0:
            logger.warning("No cells detected in any frame")

        # Sanitize masks to ensure valid cell IDs
        logger.info("Validating and sanitizing segmentation masks...")
        sanitized_segmentations = self._sanitize_segmentations(segmentations)

        # Use default configuration if not provided
        if config_dict is None:
            config_dict = {
                "max_distance": 50,
                "max_gap": 5,
                "min_track_length": 10,
            }

        logger.info(f"UlTrack configuration: {config_dict}")

        # Configure UlTrack using MainConfig with LinkingConfig for tracking parameters
        linking_config = config.LinkingConfig(
            max_distance=config_dict.get("max_distance", 50)
        )
        tracking_config = config.TrackingConfig()
        ultrack_config = config.MainConfig(
            linking_config=linking_config,
            tracking_config=tracking_config,
        )

        logger.info(f"UlTrack max_distance set to: {linking_config.max_distance}")

        # Run tracking
        logger.info("Running UlTrack tracking algorithm...")
        try:
            tracked = track(sanitized_segmentations, ultrack_config)
            logger.info(f"Tracking complete. Output shape: {tracked.shape}, value range: [{tracked.min()}, {tracked.max()}]")
            return tracked
        except Exception as e:
            logger.error(f"Error during tracking: {e}", exc_info=True)
            raise

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
        typer.echo("Segmenting cells...")
        segmentations = self.segment_timelapse(images, diameter, channels)

        typer.echo("Tracking cells...")
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

        typer.echo("Saving intermediate segmentation mask...")
        tifffile.imwrite(output_dir / "segmentations.tiff", segmentations.astype(np.uint16))
        typer.echo("Saving intermediate tracking mask...")
        tifffile.imwrite(output_dir / "tracked.tiff", tracked.astype(np.uint16))

        typer.echo(f"Intermediate results saved to {output_dir}")

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

        typer.echo("Extracting cell trajectories...")
        trajectories = TrackExtractor.extract_trajectories(
            tracked,
            min_track_length=min_track_length,
        )

        typer.echo(f"Extracted {len(trajectories)} cell tracks")

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
