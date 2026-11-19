"""
Visualization utility for grounded relations and detections.

Draws predicted boxes and relation arrows on images.
"""

from typing import List, Optional

import numpy as np
import torch


# ImageNet statistics used during dataset normalization
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denormalize(image: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalization and convert (C, H, W) tensor → (H, W, C) numpy array.
    Output values are clipped to [0, 1].
    """
    img = image.cpu().float() * _STD + _MEAN   # (C, H, W)
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()        # (H, W, C)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    (cx, cy, w, h) → (x_min, y_min, x_max, y_max)
    Input/output shape: (..., 4)
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - 0.5 * w,
                        cy - 0.5 * h,
                        cx + 0.5 * w,
                        cy + 0.5 * h], dim=-1)


def _to_pixels(boxes_norm: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Scale normalized [0, 1] boxes → pixel coordinates.
    boxes_norm: (..., 4) in (x_min, y_min, x_max, y_max) format
    """
    scale = torch.tensor([width, height, width, height],
                         dtype=boxes_norm.dtype)
    return boxes_norm * scale


def draw_relation_boxes(
    image,
    detections: List[dict],
    relations: List[dict],
    output_path: str,
    caption: str = "",
) -> None:
    """
    Draw detection boxes with relation arrows and save visualization.

    Args:
        image:      PIL Image or numpy array.
        detections: [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
        relations:  [{"subject": str, "predicate": str, "object": str,
                      "confidence": float, "subject_box": [...], "object_box": [...]}, ...]
        output_path: Path to save the visualization.
        caption:    Optional caption text to overlay.
    """
    try:
        from PIL import ImageDraw, ImageFont, Image as PILImage
    except ImportError:
        print("[visualize] PIL not available — skipping draw_relation_boxes")
        return

    if isinstance(image, torch.Tensor):
        img_np = _denormalize(image)
        img_np = (img_np * 255).clip(0, 255).astype("uint8")
        pil = PILImage.fromarray(img_np)
    else:
        pil = image.convert("RGB") if hasattr(image, "convert") else PILImage.fromarray(image)

    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    colors = [
        "#FF4444", "#4488FF", "#44BB44", "#FF8800", "#AA44FF",
        "#00CCCC", "#FF44FF", "#888800",
    ]

    # Draw detection boxes
    for i, d in enumerate(detections):
        color = colors[i % len(colors)]
        box = d["box"]
        draw.rectangle(box, outline=color, width=2)
        label_text = f"{d['label']} {d.get('score', 0):.2f}"
        bbox = draw.textbbox((box[0], box[1] - 16), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1] - 16), label_text, fill="white", font=font)

    # Draw relation arrows
    for r in relations:
        subj_box = r.get("subject_box")
        obj_box = r.get("object_box")
        if subj_box and obj_box:
            sx = (subj_box[0] + subj_box[2]) / 2
            sy = (subj_box[1] + subj_box[3]) / 2
            ox = (obj_box[0] + obj_box[2]) / 2
            oy = (obj_box[1] + obj_box[3]) / 2
            draw.line([(sx, sy), (ox, oy)], fill="#FFDD00", width=2)
            mx = (sx + ox) / 2
            my = (sy + oy) / 2 - 12
            rel_text = f"{r['predicate']} ({r.get('confidence', 0):.2f})"
            text_bbox = draw.textbbox((mx, my), rel_text, font=small_font)
            draw.rectangle(text_bbox, fill="#FFDD00")
            draw.text((mx, my), rel_text, fill="black", font=small_font)

    # Caption overlay
    if caption:
        w, h = pil.size
        lines = []
        words = caption.split()
        current = ""
        for word in words:
            test = current + " " + word if current else word
            if draw.textbbox((0, 0), test, font=font)[2] < w - 20:
                current = test
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        y = h - 20 - len(lines) * 18
        for line in lines:
            tb = draw.textbbox((10, y), line, font=font)
            draw.rectangle(tb, fill=(0, 0, 0, 180))
            draw.text((10, y), line, fill="white", font=font)
            y += 18

    pil.save(output_path)
    print(f"[visualize] Saved relation visualization to {output_path}")
