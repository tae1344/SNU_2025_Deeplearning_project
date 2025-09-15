"""
Complete Pipeline Runner: Segmentation → Classification

# 기본 SAM 처리
python scripts/run_complete_pipeline.py

# 이미지 수 제한 (테스트용)
python scripts/run_complete_pipeline.py --max-images 100

# 배치 크기 조절
python scripts/run_complete_pipeline.py --batch-size 5

# 특정 데이터셋만 처리
python scripts/run_complete_pipeline.py --splits new_test
python scripts/run_complete_pipeline.py --splits train test
python scripts/run_complete_pipeline.py --splits validation --max-images 50

"""

import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.integrated_segmentation_pipeline import IntegratedSegmentationPipeline


def process_sam_only_dataset(
    pipeline, input_dir: str, output_dir: str, max_images: int = None
) -> dict:
    """Process dataset using SAM only (no YOLO)."""
    from pathlib import Path
    import cv2
    import numpy as np

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "masks").mkdir(parents=True, exist_ok=True)

    # Get all image files (search in subdirectories too)
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        # Search recursively in all subdirectories
        image_files.extend(input_path.rglob(ext))

    # Limit number of images if specified
    if max_images and len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"Limited to {max_images} images for processing")

    print(f"Processing {len(image_files)} images with SAM only...")

    stats = {
        "total": len(image_files),
        "processed": 0,
        "failed": 0,
        "sam_only": 0,
        "fallback": 0,
        "errors": [],
    }

    for img_file in image_files:
        try:
            # Process with SAM
            result = pipeline.run(str(img_file))

            # Convert PIL Image to numpy array for OpenCV
            processed_img = result["processed_image"]
            if hasattr(processed_img, "convert"):  # PIL Image
                processed_img = np.array(processed_img.convert("RGB"))
            elif not isinstance(processed_img, np.ndarray):
                processed_img = np.array(processed_img)

            # 추가: RGB -> BGR 변환
            if processed_img.ndim == 3 and processed_img.shape[2] == 3:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

            # Save processed image
            output_img_path = output_path / "images" / f"{img_file.stem}_processed.jpg"
            cv2.imwrite(str(output_img_path), processed_img)

            # Save mask (ensure values are 0 and 255)
            mask = result["mask"]
            if mask.max() <= 1:  # If mask is 0/1, convert to 0/255
                mask = (mask * 255).astype(np.uint8)

            output_mask_path = output_path / "masks" / f"{img_file.stem}_mask.png"
            cv2.imwrite(str(output_mask_path), mask)

            # Update stats
            stats["processed"] += 1
            stats["sam_only"] += 1

            if stats["processed"] % 10 == 0:
                print(f"Processed {stats['processed']}/{stats['total']} images...")

        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            stats["failed"] += 1
            stats["errors"].append(f"{img_file.name}: {str(e)}")
            continue

    return stats


def run_complete_pipeline(
    base_dir: str,
    process_data: bool = True,
    epochs: int = 80,
    device: str = "mps",
    max_images: int = None,
    batch_size: int = 10,
    splits: list = None,
):
    """
    Run the complete pipeline from Label Studio exports to processed classification data.

    Args:
        base_dir: Project root directory
        process_data: Whether to process dataset with trained model
        epochs: Number of training epochs
        device: Device for training
        max_images: Maximum number of images to process per split
        batch_size: Batch size for processing
        splits: List of splits to process (e.g., ['train', 'test']). If None, process all splits.
    """

    print("=== Complete Segmentation Pipeline ===")
    print(f"Base directory: {base_dir}")
    print(f"Process data: {process_data}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Target splits: {splits if splits else 'all'}")
    print()

    # Process dataset with trained model
    if process_data:
        print("\n Processing dataset with trained model...")
        try:
            # Create integrated pipeline
            # Use SAM-only pipeline (no YOLO)
            from common.segmentation import SegmentationPipeline, PipelineParams

            pipeline = SegmentationPipeline(
                PipelineParams(
                    sam_weights_path="./model/mobile_sam.pt",
                    yolo_seg_weights_path=None,
                    use_yolo_box_prompt=False,
                    use_multi_point_prompt=True,
                )
            )
            pipeline.load_models()

            # Process different splits
            if splits is None:
                splits = ["train", "validation", "test", "new_test"]

            all_stats = {}

            for split in splits:
                input_dir = f"data/{split}/원천데이터"
                output_dir = f"data/processed_{split}_segmented"

                if os.path.exists(input_dir):
                    print(f"Processing {split} split...")
                    # Use SAM-only processing
                    stats = process_sam_only_dataset(
                        pipeline, input_dir, output_dir, max_images
                    )
                    all_stats[split] = stats
                    print(f"✓ {split} processed: {stats['processed']} images")
                else:
                    print(f"⚠ {split} directory not found: {input_dir}")

            print(f"✓ Processing complete!")

        except Exception as e:
            print(f"✗ Error processing dataset: {e}")
            return False

    print("\n=== Pipeline Complete ===")
    print("Next steps:")
    print("1. Review processed images in data/processed_*_segmented/")
    print("2. Update your classification training to use segmented images")
    print("3. Compare performance: original vs segmented")

    return True


def main():
    parser = argparse.ArgumentParser(description="Complete segmentation pipeline")
    parser.add_argument(
        "--base-dir",
        default=os.getcwd(),
        help="Project root directory",
    )
    parser.add_argument(
        "--no-process", action="store_true", help="Skip data processing"
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="Number of training epochs"
    )
    parser.add_argument(
        "--device", default="mps", help="Device for training (mps/cpu/cuda)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process per split",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for processing"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Specific splits to process (e.g., --splits train test new_test)",
    )

    args = parser.parse_args()

    success = run_complete_pipeline(
        base_dir=args.base_dir,
        process_data=not args.no_process,
        epochs=args.epochs,
        device=args.device,
        max_images=args.max_images,
        batch_size=args.batch_size,
        splits=args.splits,
    )

    if success:
        print("\n✓ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
