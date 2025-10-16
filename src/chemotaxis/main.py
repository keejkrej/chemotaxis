"""
Command-line interface for the chemotaxis cell tracking program.
"""

from pathlib import Path
from typing import Optional
import typer
import sys
import logging
from .tracker import CellTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = typer.Typer(
    help="Automated cell tracking from microscopy images using Cellpose and UlTrack"
)


@app.command()
def main(
    input: Path = typer.Argument(
        ...,
        help="Path to image file (TIFF) or directory with image sequence (JPEG/PNG/TIFF)",
        exists=True,
    ),
    output: Path = typer.Option(
        Path("./results"),
        "-o",
        "--output",
        help="Output directory for results",
    ),
    model: str = typer.Option(
        "cyto2",
        "-m",
        "--model",
        help="Cellpose model: cyto, cyto2, nuclei",
    ),
    diameter: Optional[float] = typer.Option(
        None,
        "-d",
        "--diameter",
        help="Expected cell diameter in pixels (auto-estimate if not provided)",
    ),
    max_distance: int = typer.Option(
        50,
        "--max-distance",
        help="Maximum distance for cell linking in pixels",
    ),
    max_gap: int = typer.Option(
        5,
        "--max-gap",
        help="Maximum frames to bridge tracking gaps",
    ),
    min_track_length: int = typer.Option(
        10,
        "--min-track-length",
        help="Minimum frames for a valid track",
    ),
    no_gpu: bool = typer.Option(
        False,
        "--no-gpu",
        help="Disable GPU acceleration",
    ),
    no_intermediate: bool = typer.Option(
        False,
        "--no-intermediate",
        help="Skip saving intermediate segmentation/tracking masks",
    ),
):
    """
    Track cells in microscopy images.

    Examples:

        chemotaxis /path/to/images -o ./results --max-distance 100 --max-gap 3

        chemotaxis image.tiff -o ./results -m cyto2 --diameter 30

        chemotaxis data/egf -o results_egf --no-intermediate
    """
    try:
        input_path = input

        # Handle both file and directory inputs
        if input_path.is_dir():
            typer.echo(f"Loading image sequence from directory: {input_path}")
            # Find image files
            supported_formats = ("*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif")
            image_files = []
            for fmt in supported_formats:
                image_files.extend(sorted(input_path.glob(fmt)))
                image_files.extend(sorted(input_path.glob(fmt.upper())))

            if not image_files:
                typer.echo(f"Error: No image files found in {input_path}", err=True)
                raise typer.Exit(code=1)

            typer.echo(f"Found {len(image_files)} image files")

            # Load images
            try:
                import numpy as np
                from PIL import Image

                typer.echo("Loading images...")
                images_list = []
                for img_path in image_files:
                    img = Image.open(img_path)
                    images_list.append(np.array(img))

                images = np.array(images_list)
            except Exception as e:
                typer.echo(f"Error loading images: {e}", err=True)
                raise typer.Exit(code=1)
        else:
            typer.echo(f"Loading image from file: {input_path}")
            tracker = CellTracker(model_type=model, use_gpu=not no_gpu)
            images = tracker.load_images(input_path)

        typer.echo(f"Image shape: {images.shape}")

        # Initialize tracker
        tracker = CellTracker(model_type=model, use_gpu=not no_gpu)

        # Configure tracking
        track_config = {
            "max_distance": max_distance,
            "max_gap": max_gap,
            "min_track_length": min_track_length,
        }

        typer.echo("Starting segmentation and tracking...")
        segmentations, tracked = tracker.process_timelapse(
            images,
            diameter=diameter,
            track_config=track_config,
        )

        # Save results
        output.mkdir(parents=True, exist_ok=True)

        if not no_intermediate:
            typer.echo("Saving intermediate segmentation and tracking masks...")
            tracker.save_intermediate_results(output, segmentations, tracked)

        typer.echo("Extracting and exporting cell tracks...")
        tracker.extract_and_export_tracks(
            tracked,
            output,
            formats=["csv", "json", "summary"],
            min_track_length=min_track_length,
        )

        typer.echo(f"\n✓ Successfully completed analysis")
        typer.echo(f"✓ Results saved to: {output}")
        typer.echo(f"✓ Output files:")
        typer.echo(f"  - tracks.csv: All cell coordinates")
        typer.echo(f"  - tracks.json: Coordinates with statistics")
        typer.echo(f"  - tracks_summary.csv: Summary statistics per track")

    except Exception as e:
        if not isinstance(e, typer.Exit):
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
