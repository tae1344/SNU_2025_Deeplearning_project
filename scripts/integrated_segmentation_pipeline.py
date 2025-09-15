"""
Integrated Segmentation Pipeline: YOLO + SAM + Classification
Combines YOLO segmentation, MobileSAM refinement, and classification preprocessing.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

try:
    from ultralytics import YOLO, SAM
    from PIL import Image
except ImportError:
    print("Please install required packages: pip install ultralytics pillow")
    exit(1)

from common.segmentation import SegmentationPipeline, PipelineParams


class IntegratedSegmentationPipeline:
    """
    Complete pipeline: YOLO segmentation → SAM refinement → Classification preprocessing
    """

    def __init__(
        self,
        yolo_model_path: str,
        sam_model_path: str = "./model/mobile_sam.pt",
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.yolo_model_path = yolo_model_path
        self.sam_model_path = sam_model_path
        self.target_size = target_size

        # Load models
        self.yolo_model = YOLO(yolo_model_path)
        self.sam_pipeline = SegmentationPipeline(
            PipelineParams(
                sam_weights_path=sam_model_path,
                yolo_seg_weights_path=None,  # We'll use our trained YOLO
                use_yolo_box_prompt=False,  # We'll provide bbox manually
                use_multi_point_prompt=True,
                target_size=target_size,
            )
        )
        self.sam_pipeline.load_models()

    def process_single_image(
        self, image_path: str, use_sam_refinement: bool = True
    ) -> Dict:
        """
        Process single image through the pipeline.

        Args:
            image_path: Path to input image
            use_sam_refinement: Whether to use SAM for refinement

        Returns:
            Dictionary with processed image, mask, and metadata
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w = img.shape[:2]

        # Step 1: YOLO segmentation
        yolo_results = self.yolo_model.predict(image_path, verbose=False)
        yolo_mask = self._extract_yolo_mask(yolo_results, (h, w))

        if yolo_mask.sum() == 0:
            print(f"Warning: No YOLO mask found for {image_path}")
            # Fallback to center crop
            return self._fallback_center_crop(img, image_path)

        # Step 2: Optional SAM refinement
        if use_sam_refinement:
            try:
                # Get bbox from YOLO mask
                bbox = self._mask_to_bbox(yolo_mask)
                if bbox:
                    # Use SAM with YOLO bbox as prompt
                    sam_result = self.sam_pipeline.run(image_path)
                    refined_mask = sam_result["mask"]

                    # Choose better mask
                    if (
                        refined_mask.sum() > yolo_mask.sum() * 0.5
                    ):  # SAM found something reasonable
                        final_mask = refined_mask
                        method = "yolo+sam"
                    else:
                        final_mask = yolo_mask
                        method = "yolo_only"
                else:
                    final_mask = yolo_mask
                    method = "yolo_only"
            except Exception as e:
                print(f"SAM refinement failed for {image_path}: {e}")
                final_mask = yolo_mask
                method = "yolo_only"
        else:
            final_mask = yolo_mask
            method = "yolo_only"

        # Step 3: ROI crop and resize
        processed_img = self._crop_and_resize(img, final_mask)

        return {
            "processed_image": processed_img,
            "mask": final_mask,
            "method": method,
            "original_size": (h, w),
            "processed_size": self.target_size,
            "mask_area": int(final_mask.sum()),
            "mask_ratio": float(final_mask.sum()) / (h * w),
        }

    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        use_sam_refinement: bool = True,
        save_masks: bool = True,
    ) -> Dict:
        """
        Process entire dataset.

        Args:
            input_dir: Directory containing input images
            output_dir: Output directory for processed images
            use_sam_refinement: Whether to use SAM refinement
            save_masks: Whether to save mask files

        Returns:
            Processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Create output directories
        (output_path / "images").mkdir(parents=True, exist_ok=True)
        if save_masks:
            (output_path / "masks").mkdir(parents=True, exist_ok=True)

        # Get all image files (search recursively in subdirectories)
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(input_path.rglob(ext))

        print(f"Processing {len(image_files)} images...")

        stats = {
            "total": len(image_files),
            "processed": 0,
            "failed": 0,
            "yolo_only": 0,
            "yolo_sam": 0,
            "fallback": 0,
            "errors": [],
        }

        for img_file in image_files:
            try:
                result = self.process_single_image(str(img_file), use_sam_refinement)

                # Save processed image
                output_img_path = (
                    output_path / "images" / f"{img_file.stem}_processed.jpg"
                )
                cv2.imwrite(str(output_img_path), result["processed_image"])

                # Save mask if requested
                if save_masks:
                    output_mask_path = (
                        output_path / "masks" / f"{img_file.stem}_mask.png"
                    )
                    cv2.imwrite(str(output_mask_path), result["mask"])

                # Update stats
                stats["processed"] += 1
                stats[result["method"]] += 1

                if stats["processed"] % 10 == 0:
                    print(f"Processed {stats['processed']}/{stats['total']} images...")

            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                stats["failed"] += 1
                stats["errors"].append(f"{img_file.name}: {str(e)}")
                continue

        # Save processing log
        log_path = output_path / "processing_log.json"
        with open(log_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nProcessing complete!")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")
        print(f"YOLO only: {stats['yolo_only']}")
        print(f"YOLO+SAM: {stats['yolo_sam']}")
        print(f"Fallback: {stats['fallback']}")
        print(f"Log saved to: {log_path}")

        return stats

    def _extract_yolo_mask(self, results, image_shape: Tuple[int, int]) -> np.ndarray:
        """Extract mask from YOLO results."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        try:
            r0 = results[0]
            if hasattr(r0, "masks") and r0.masks is not None:
                masks = r0.masks.data.cpu().numpy()  # (N, H, W)
                for m in masks:
                    binary_mask = (m > 0.5).astype(np.uint8)
                    mask = np.maximum(mask, binary_mask * 255)
        except Exception as e:
            print(f"Error extracting YOLO mask: {e}")

        return mask

    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Convert mask to bounding box."""
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None

        return (
            int(x_indices.min()),
            int(y_indices.min()),
            int(x_indices.max()),
            int(y_indices.max()),
        )

    def _crop_and_resize(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop image based on mask and resize to target size."""
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            # Fallback to center crop
            h, w = img.shape[:2]
            return cv2.resize(img, self.target_size)

        y0, y1 = int(y_indices.min()), int(y_indices.max())
        x0, x1 = int(x_indices.min()), int(x_indices.max())

        # Add padding
        h, w = img.shape[:2]
        pad_x = int(0.1 * (x1 - x0 + 1))
        pad_y = int(0.1 * (y1 - y0 + 1))
        x0 = max(0, x0 - pad_x)
        x1 = min(w - 1, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(h - 1, y1 + pad_y)

        # Make square
        x0, y0, x1, y1 = self._make_square_bbox(x0, y0, x1, y1, w, h)

        # Crop and resize
        cropped = img[y0 : y1 + 1, x0 : x1 + 1]
        resized = cv2.resize(cropped, self.target_size)

        return resized

    def _make_square_bbox(
        self, x0: int, y0: int, x1: int, y1: int, w: int, h: int
    ) -> Tuple[int, int, int, int]:
        """Make bounding box square."""
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        side = max(bw, bh)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        sx0 = max(0, cx - side // 2)
        sy0 = max(0, cy - side // 2)
        sx1 = min(w - 1, sx0 + side - 1)
        sy1 = min(h - 1, sy0 + side - 1)

        sx0 = max(0, sx1 - side + 1)
        sy0 = max(0, sy1 - side + 1)

        return sx0, sy0, sx1, sy1

    def _fallback_center_crop(self, img: np.ndarray, image_path: str) -> Dict:
        """Fallback to center crop when no mask is found."""
        h, w = img.shape[:2]
        center_h, center_w = h // 2, w // 2
        crop_h, crop_w = int(h * 0.6), int(w * 0.6)

        y0 = max(0, center_h - crop_h // 2)
        y1 = min(h, center_h + crop_h // 2)
        x0 = max(0, center_w - crop_w // 2)
        x1 = min(w, center_w + crop_w // 2)

        cropped = img[y0:y1, x0:x1]
        resized = cv2.resize(cropped, self.target_size)

        # Create dummy mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255

        return {
            "processed_image": resized,
            "mask": mask,
            "method": "fallback",
            "original_size": (h, w),
            "processed_size": self.target_size,
            "mask_area": int(mask.sum()),
            "mask_ratio": float(mask.sum()) / (h * w),
        }


def main():
    """Example usage of the integrated pipeline."""
    # Paths
    base_dir = os.getcwd()
    yolo_model_path = (
        "runs/segment/lesion_seg/weights/best.pt"  # Update with actual path
    )
    input_dir = "data/new_test/원천데이터"  # Update with actual path
    output_dir = "data/processed_segmented"

    # Create pipeline
    pipeline = IntegratedSegmentationPipeline(
        yolo_model_path=yolo_model_path,
        sam_model_path="./model/mobile_sam.pt",
        target_size=(224, 224),
    )

    # Process dataset
    stats = pipeline.process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        use_sam_refinement=True,
        save_masks=True,
    )

    print(f"Processing completed with stats: {stats}")


if __name__ == "__main__":
    main()
