from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    # Optional: connected components, hole filling
    from scipy import ndimage as ndi  # type: ignore
except Exception:  # pragma: no cover
    ndi = None  # type: ignore

try:
    # Optional: Ultralytics for MobileSAM/YOLO
    from ultralytics import SAM, YOLO  # type: ignore
except Exception:  # pragma: no cover
    SAM = None  # type: ignore
    YOLO = None  # type: ignore


# ------------------------------ Config Objects ------------------------------ #


@dataclass
class SamParams:
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.85
    crop_n_layers: int = 2
    crop_overlap_ratio: float = 0.5
    box_nms_thresh: float = 0.6
    min_mask_region_area: int = 1200


@dataclass
class PostprocessParams:
    morph_open_kernel: int = 3
    morph_close_kernel: int = 5
    dilate_iterations: int = 0
    keep_largest_component: bool = True
    min_mask_area_ratio: float = 1e-3
    max_mask_area_ratio: float = 0.5
    square_crop: bool = True
    target_size: Tuple[int, int] = (224, 224)


@dataclass
class FallbackParams:
    enable_otsu_fallback: bool = True


@dataclass
class PipelineParams:
    sam_weights_path: str = "./model/mobile_sam.pt"
    yolo_seg_weights_path: Optional[str] = None  # e.g., "yolo11n-seg.pt"
    use_yolo_box_prompt: bool = True
    use_multi_point_prompt: bool = True
    num_positive_points: int = 3
    num_negative_points: int = 3
    background_fill: str = "black"  # placeholder for future use
    sam: SamParams = field(default_factory=SamParams)
    post: PostprocessParams = field(default_factory=PostprocessParams)
    fallback: FallbackParams = field(default_factory=FallbackParams)


# ------------------------------ Main Pipeline ------------------------------ #


class SegmentationPipeline:
    """
    Helper to produce lesion masks using MobileSAM with optional YOLO box prompts,
    multi-prompt sampling, and robust post-processing with an Otsu fallback.

    Usage:
        pipe = SegmentationPipeline(PipelineParams())
        pipe.load_models()  # loads SAM (+ YOLO if provided)
        result = pipe.run(image_path)

    Returns a dict containing:
      - processed_image: PIL.Image
      - mask: np.ndarray (uint8 {0,1})
      - debug: Dict with details
    """

    def __init__(self, params: PipelineParams) -> None:
        self.p = params
        self.sam_model = None
        self.yolo_model = None

    # ------------------------------ Model Loading --------------------------- #
    def load_models(self) -> None:
        if SAM is None or Image is None:
            raise RuntimeError(
                "PIL and ultralytics (SAM) are required for this pipeline."
            )
        self.sam_model = SAM(self.p.sam_weights_path)
        if self.p.yolo_seg_weights_path and YOLO is not None:
            self.yolo_model = YOLO(self.p.yolo_seg_weights_path)

    # --------------------------------- Run --------------------------------- #
    def run(
        self, image_path: str, target_class: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.sam_model is None:
            raise RuntimeError("call load_models() first.")

        pil = Image.open(image_path).convert("RGB")
        w, h = pil.size

        # 1) Optional YOLO step to get coarse bbox for prompting
        yolo_bbox = None
        if self.p.use_yolo_box_prompt and self.yolo_model is not None:
            try:
                y_res = self.yolo_model.predict(image_path, verbose=False)
                yolo_bbox = self._select_best_yolo_bbox(y_res)
            except Exception:
                yolo_bbox = None

        # 2) Build multi-prompts (points and optional box)
        pos_points, neg_points = self._generate_points(w, h)

        # 3) SAM inference: try multiple prompt variants and keep best
        masks_with_scores = []
        prompt_variants = self._build_prompt_variants(pos_points, neg_points, yolo_bbox)
        for pv in prompt_variants:
            try:
                out = self.sam_model.predict(
                    image_path,
                    points=pv.get("points"),
                    labels=pv.get("labels"),
                    bboxes=pv.get("bboxes"),
                    pred_iou_thresh=self.p.sam.pred_iou_thresh,
                    stability_score_thresh=self.p.sam.stability_score_thresh,
                    crop_n_layers=self.p.sam.crop_n_layers,
                    crop_overlap_ratio=self.p.sam.crop_overlap_ratio,
                    box_nms_thresh=self.p.sam.box_nms_thresh,
                    min_mask_region_area=self.p.sam.min_mask_region_area,
                    verbose=False,
                )
                masks_with_scores.extend(self._extract_sam_masks(out))
            except Exception:
                continue

        # 4) Choose best mask candidate
        gray = self._to_grayscale(np.asarray(pil))
        mask = self._select_best_mask(masks_with_scores, gray.shape)

        # 5) Postprocess and validity check
        mask = self._postprocess_mask(mask)
        if not self._is_valid_mask(mask, gray.shape):
            if self.p.fallback.enable_otsu_fallback:
                mask = self._fallback_otsu(gray)
                mask = self._postprocess_mask(mask)

        # 6) ROI crop and resize
        processed = self._apply_roi_crop_and_resize(pil, mask)

        return {
            "processed_image": processed,
            "mask": mask.astype(np.uint8),
            "debug": {
                "yolo_bbox": yolo_bbox,
                "num_candidates": len(masks_with_scores),
                "image_size": (w, h),
            },
        }

    # ------------------------------- Prompts -------------------------------- #
    def _generate_points(
        self, w: int, h: int
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        # Positive: center and near-center grid; Negative: image corners/border
        cx, cy = w // 2, h // 2
        pos = [(cx, cy)]
        if self.p.use_multi_point_prompt:
            offsets = [(-w // 8, 0), (w // 8, 0), (0, -h // 8), (0, h // 8)]
            for dx, dy in offsets[: max(0, self.p.num_positive_points - 1)]:
                x = int(np.clip(cx + dx, 0, w - 1))
                y = int(np.clip(cy + dy, 0, h - 1))
                pos.append((x, y))

        neg = []
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        for i in range(min(self.p.num_negative_points, len(corners))):
            neg.append(corners[i])
        return pos, neg

    def _build_prompt_variants(
        self,
        pos_points: List[Tuple[int, int]],
        neg_points: List[Tuple[int, int]],
        yolo_bbox: Optional[Tuple[int, int, int, int]],
    ) -> List[Dict[str, Any]]:
        variants: List[Dict[str, Any]] = []

        # points-only
        pts = pos_points + neg_points
        if pts:
            labels = [1] * len(pos_points) + [0] * len(neg_points)
            variants.append(
                {"points": [list(p) for p in pts], "labels": labels, "bboxes": None}
            )

        # points + yolo bbox (if available)
        if yolo_bbox is not None:
            variants.append(
                {
                    "points": [list(p) for p in pts] if pts else None,
                    "labels": (
                        [1] * len(pos_points) + [0] * len(neg_points) if pts else None
                    ),
                    "bboxes": [list(yolo_bbox)],
                }
            )

        # bbox-only
        if yolo_bbox is not None:
            variants.append(
                {"points": None, "labels": None, "bboxes": [list(yolo_bbox)]}
            )

        return variants

    # --------------------------------- SAM ---------------------------------- #
    @staticmethod
    def _extract_sam_masks(results: Any) -> List[Tuple[np.ndarray, float]]:
        masks: List[Tuple[np.ndarray, float]] = []
        try:
            res0 = results[0]
            # Ultralytics SAM returns .masks.data (N,H,W) and .boxes.conf or .probs
            if hasattr(res0, "masks") and res0.masks is not None:
                m = res0.masks.data.cpu().numpy()  # (N, H, W) float {0,1}
                scores = None
                if (
                    hasattr(res0, "boxes")
                    and res0.boxes is not None
                    and hasattr(res0.boxes, "conf")
                ):
                    scores = res0.boxes.conf.cpu().numpy()
                for i in range(m.shape[0]):
                    mask = (m[i] > 0.5).astype(np.uint8)
                    score = (
                        float(scores[i])
                        if scores is not None and i < len(scores)
                        else 0.0
                    )
                    masks.append((mask, score))
        except Exception:
            return []
        return masks

    # -------------------------------- YOLO ---------------------------------- #
    @staticmethod
    def _select_best_yolo_bbox(
        yolo_results: Any,
    ) -> Optional[Tuple[int, int, int, int]]:
        try:
            r0 = yolo_results[0]
            if r0.boxes is None or len(r0.boxes) == 0:
                # try masks to derive bbox
                if r0.masks is not None and r0.masks.data is not None:
                    m = r0.masks.data[0].cpu().numpy()
                    ys, xs = np.where(m > 0.5)
                    if ys.size and xs.size:
                        return (
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max()),
                            int(ys.max()),
                        )
                return None

            # choose highest confidence box
            boxes = r0.boxes.xyxy.cpu().numpy()  # (N,4)
            conf = r0.boxes.conf.cpu().numpy()  # (N,)
            idx = int(np.argmax(conf))
            x0, y0, x1, y1 = boxes[idx]
            return int(x0), int(y0), int(x1), int(y1)
        except Exception:
            return None

    # ------------------------------ Postprocess ----------------------------- #
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8)
        if self.p.post.morph_open_kernel and self.p.post.morph_open_kernel > 1:
            mask = self._morph_open(mask, self.p.post.morph_open_kernel)
        if self.p.post.morph_close_kernel and self.p.post.morph_close_kernel > 1:
            mask = self._morph_close(mask, self.p.post.morph_close_kernel)
        if self.p.post.keep_largest_component:
            mask = self._largest_component(mask)
        if self.p.post.dilate_iterations and self.p.post.dilate_iterations > 0:
            mask = self._dilate(mask, iterations=self.p.post.dilate_iterations)
        if ndi is not None:
            mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)
        return mask

    def _apply_roi_crop_and_resize(
        self, pil_img: "Image.Image", mask: np.ndarray
    ) -> "Image.Image":
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            return pil_img.resize(self.p.post.target_size, resample=Image.BILINEAR)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        h, w = mask.shape
        # padding 10%
        pad_x = int(0.1 * (x1 - x0 + 1))
        pad_y = int(0.1 * (y1 - y0 + 1))
        x0 = max(0, x0 - pad_x)
        x1 = min(w - 1, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(h - 1, y1 + pad_y)
        if self.p.post.square_crop:
            x0, y0, x1, y1 = self._make_square_bbox(x0, y0, x1, y1, w, h)
        cropped = pil_img.crop((x0, y0, x1 + 1, y1 + 1))
        return cropped.resize(self.p.post.target_size, resample=Image.BILINEAR)

    def _is_valid_mask(self, mask: np.ndarray, hw: Tuple[int, int]) -> bool:
        h, w = hw
        area = float(mask.sum())
        if area < self.p.post.min_mask_area_ratio * (h * w):
            return False
        if area > self.p.post.max_mask_area_ratio * (h * w):
            return False
        return True

    # ------------------------------- Fallback ------------------------------- #
    @staticmethod
    def _fallback_otsu(gray: np.ndarray) -> np.ndarray:
        hist, bins = np.histogram(gray.ravel(), bins=256, range=(0, 255))
        hist = hist.astype(np.float64)
        centers = (bins[:-1] + bins[1:]) * 0.5
        w1 = np.cumsum(hist)
        w2 = np.cumsum(hist[::-1])[::-1]
        m1 = np.cumsum(hist * centers) / np.maximum(w1, 1e-12)
        m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
        var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
        idx = np.argmax(var)
        thr = float(centers[idx])
        return (gray <= thr).astype(np.uint8)

    # ------------------------------- Utilities ----------------------------- #
    @staticmethod
    def _to_grayscale(np_img: np.ndarray) -> np.ndarray:
        if np_img.ndim == 2:
            return np_img.astype(np.float32)
        if np_img.ndim == 3 and np_img.shape[2] == 1:
            return np_img[:, :, 0].astype(np.float32)
        if np_img.ndim == 3 and np_img.shape[2] >= 3:
            r = np_img[:, :, 0].astype(np.float32)
            g = np_img[:, :, 1].astype(np.float32)
            b = np_img[:, :, 2].astype(np.float32)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        raise ValueError("Unsupported image shape for grayscale conversion")

    @staticmethod
    def _morph_open(mask: np.ndarray, k: int = 3) -> np.ndarray:
        return SegmentationPipeline._dilate(SegmentationPipeline._erode(mask, k), 1, k)

    @staticmethod
    def _morph_close(mask: np.ndarray, k: int = 5) -> np.ndarray:
        return SegmentationPipeline._erode(SegmentationPipeline._dilate(mask, 1, k), k)

    @staticmethod
    def _erode(mask: np.ndarray, k: int = 3) -> np.ndarray:
        pad = k // 2
        kernel = np.ones((k, k), dtype=np.uint8)
        padded = np.pad(mask, pad_width=pad, mode="constant", constant_values=0)
        out = np.zeros_like(mask)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                region = padded[y : y + k, x : x + k]
                out[y, x] = 1 if np.all(region >= kernel) else 0
        return out

    @staticmethod
    def _dilate(mask: np.ndarray, iterations: int = 1, k: int = 3) -> np.ndarray:
        out = mask.copy().astype(np.uint8)
        pad = k // 2
        kernel = np.ones((k, k), dtype=np.uint8)
        for _ in range(max(1, iterations)):
            padded = np.pad(out, pad_width=pad, mode="constant", constant_values=0)
            nxt = np.zeros_like(out)
            for y in range(out.shape[0]):
                for x in range(out.shape[1]):
                    region = padded[y : y + k, x : x + k]
                    nxt[y, x] = 1 if np.any(region & kernel) else 0
            out = nxt
        return out

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        if ndi is None:
            return mask
        labeled, num = ndi.label(mask > 0)
        if num <= 1:
            return mask
        sizes = ndi.sum(mask, labeled, index=range(1, num + 1))
        largest_label = int(np.argmax(sizes)) + 1
        return (labeled == largest_label).astype(np.uint8)

    @staticmethod
    def _make_square_bbox(
        x0: int, y0: int, x1: int, y1: int, w: int, h: int
    ) -> Tuple[int, int, int, int]:
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

    # ------------------------------ Mask Choice ----------------------------- #
    def _select_best_mask(
        self, masks_with_scores: List[Tuple[np.ndarray, float]], hw: Tuple[int, int]
    ) -> np.ndarray:
        if not masks_with_scores:
            return np.zeros(hw, dtype=np.uint8)
        h, w = hw
        min_area = self.p.post.min_mask_area_ratio * (h * w)
        max_area = self.p.post.max_mask_area_ratio * (h * w)

        best_mask = None
        best_score = -1.0
        for m, s in masks_with_scores:
            area = float(m.sum())
            if area < min_area or area > max_area:
                continue
            score = s if s is not None else 0.0
            # small bias towards mid-size masks
            balance = 1.0 - abs((area / (h * w)) - 0.1)
            composite = score + 0.1 * balance
            if composite > best_score:
                best_score = composite
                best_mask = m
        if best_mask is None:
            # fall back to the largest reasonable mask
            sorted_by_area = sorted(
                masks_with_scores, key=lambda t: t[0].sum(), reverse=True
            )
            for m, _ in sorted_by_area:
                area = float(m.sum())
                if min_area <= area <= max_area:
                    return m.astype(np.uint8)
            return sorted_by_area[0][0].astype(np.uint8)
        return best_mask.astype(np.uint8)
