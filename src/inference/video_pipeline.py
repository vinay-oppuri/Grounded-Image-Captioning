"""
video_pipeline.py — Grounded Video Captioning Pipeline.

Extracts frames from an input video at a configurable sample rate,
runs the semantic grounded captioning pipeline on sampled keyframes,
generates timestamped caption transcripts and SRT subtitles, and
optionally renders an annotated output video with bounding boxes,
relation links, and real-time caption subtitles.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.inference.caption_pipeline import run_grounded_pipeline


def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Format seconds into MM:SS.mmm or HH:MM:SS,mmm for subtitles."""
    total_millis = int(round(seconds * 1000))
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60

    if srt_format:
        return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"
    return f"{mins:02d}:{secs:02d}.{millis:03d}"


def export_srt_subtitles(
    transcripts: List[Dict],
    output_srt_path: Union[str, Path],
    default_duration: float = 1.0,
) -> None:
    """Export list of timestamped captions to standard SubRip (.srt) subtitle file."""
    output_srt_path = Path(output_srt_path)
    output_srt_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for idx, entry in enumerate(transcripts, start=1):
        start_sec = entry.get("timestamp_sec", 0.0)
        end_sec = entry.get("end_sec", start_sec + default_duration)
        caption = entry.get("caption", "").strip()
        if not caption:
            continue

        start_str = format_timestamp(start_sec, srt_format=True)
        end_str = format_timestamp(end_sec, srt_format=True)

        lines.append(f"{idx}")
        lines.append(f"{start_str} --> {end_str}")
        lines.append(caption)
        lines.append("")  # Blank line separator

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def overlay_grounded_annotations(
    frame_bgr: np.ndarray,
    detections: List[Dict],
    relations: List[Dict],
    caption: str = "",
    timestamp_str: str = "",
) -> np.ndarray:
    """Render bounding boxes, relation lines, and caption subtitle onto a video frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    try:
        font = ImageFont.truetype("arial.ttf", max(14, int(h * 0.025)))
        small_font = ImageFont.truetype("arial.ttf", max(11, int(h * 0.018)))
        title_font = ImageFont.truetype("arial.ttf", max(16, int(h * 0.03)))
    except Exception:
        font = ImageFont.load_default()
        small_font = font
        title_font = font

    colors = [
        "#FF3333", "#3388FF", "#33CC33", "#FF9900", "#AA33FF",
        "#00DDDD", "#FF33AA", "#FFCC00",
    ]

    # 1. Draw detection bounding boxes
    for i, d in enumerate(detections):
        color = colors[i % len(colors)]
        box = d.get("box", [])
        if len(box) == 4:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{d.get('label', '')} {d.get('score', 0):.2f}"
            tb = draw.textbbox((x1, max(0, y1 - 18)), label, font=small_font)
            draw.rectangle(tb, fill=color)
            draw.text((x1 + 2, max(0, y1 - 18)), label, fill="white", font=small_font)

    # 2. Draw relation connector lines
    for r in relations:
        subj_box = r.get("subject_box")
        obj_box = r.get("object_box")
        if subj_box and obj_box and len(subj_box) == 4 and len(obj_box) == 4:
            sx = (subj_box[0] + subj_box[2]) / 2.0
            sy = (subj_box[1] + subj_box[3]) / 2.0
            ox = (obj_box[0] + obj_box[2]) / 2.0
            oy = (obj_box[1] + obj_box[3]) / 2.0
            draw.line([(sx, sy), (ox, oy)], fill="#FFDD00", width=2)
            mx = (sx + ox) / 2.0
            my = (sy + oy) / 2.0 - 10
            rel_label = f"{r.get('predicate', '')} ({r.get('confidence', 0):.2f})"
            tb = draw.textbbox((mx, my), rel_label, font=small_font)
            draw.rectangle(tb, fill="#FFDD00")
            draw.text((mx + 2, my), rel_label, fill="black", font=small_font)

    # 3. Timestamp badge (top-left)
    if timestamp_str:
        t_box = draw.textbbox((10, 10), timestamp_str, font=title_font)
        draw.rectangle([t_box[0] - 4, t_box[1] - 4, t_box[2] + 4, t_box[3] + 4], fill=(0, 0, 0))
        draw.text((10, 10), timestamp_str, fill="#00FFCC", font=title_font)

    # 4. Caption subtitle banner (bottom)
    if caption:
        words = caption.split()
        lines: List[str] = []
        cur_line = ""
        for word in words:
            test = (cur_line + " " + word).strip()
            tb = draw.textbbox((0, 0), test, font=font)
            if tb[2] - tb[0] < w - 40:
                cur_line = test
            else:
                lines.append(cur_line)
                cur_line = word
        if cur_line:
            lines.append(cur_line)

        line_h = int(h * 0.035) + 4
        banner_h = len(lines) * line_h + 16
        banner_top = h - banner_h - 10
        draw.rectangle([10, banner_top, w - 10, h - 10], fill=(0, 0, 0))

        y = banner_top + 8
        for l in lines:
            tb = draw.textbbox((0, 0), l, font=font)
            line_w = tb[2] - tb[0]
            x = (w - line_w) // 2
            draw.text((x, y), l, fill="white", font=font)
            y += line_h

    annotated_rgb = np.array(pil_img)
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


def run_video_pipeline(
    video_path: Union[str, Path],
    output_dir: Union[str, Path] = "results",
    sample_rate: float = 1.0,
    render_video: bool = True,
    top_k_relations: int = 3,
    temperature: float = 2.0,
    compare_baseline: bool = True,
) -> Dict:
    """Process a video file through the grounded captioning pipeline.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where transcripts, subtitles, and annotated videos are saved.
        sample_rate: Interval in seconds between captioned keyframes (e.g., 1.0 = 1 FPS).
        render_video: If True, writes an annotated output video with bounding boxes & subtitles.
        top_k_relations: Number of relation triplets to predict per frame.
        temperature: Temperature calibration for relation prediction.
        compare_baseline: If True, also generates baseline vanilla BLIP captions for comparison.

    Returns:
        Dict summarizing the video captioning process, transcripts, and generated artifacts.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    print("=" * 70)
    print("  GROUNDED VIDEO CAPTIONING WORKFLOW")
    print(f"  Input video: {video_path.name}")
    print(f"  Resolution:  {width}x{height} @ {fps:.2f} FPS")
    print(f"  Duration:    {duration_sec:.2f}s ({total_frames} frames)")
    print(f"  Sample rate: Every {sample_rate:.2f}s")
    print(f"  Device:      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

    step_frames = max(1, int(round(fps * sample_rate)))

    video_writer: Optional[cv2.VideoWriter] = None
    output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
    if render_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    transcripts: List[Dict] = []
    t_start = time.time()

    current_caption = ""
    current_baseline = ""
    current_detections: List[Dict] = []
    current_relations: List[Dict] = []

    frame_idx = 0
    sampled_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps
        timestamp_str = format_timestamp(timestamp_sec)

        # Check if this frame is a sampled keyframe
        if frame_idx % step_frames == 0 or frame_idx == 0:
            sampled_count += 1
            print(f"\n[Keyframe {sampled_count}] Frame #{frame_idx} (Time: {timestamp_str})")

            # Convert BGR OpenCV frame to RGB PIL Image in memory
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            # Run grounded caption pipeline on frame
            frame_res = run_grounded_pipeline(
                pil_frame,
                top_k_relations=top_k_relations,
                temperature=temperature,
                compare_baseline=compare_baseline,
            )

            current_caption = frame_res.get("caption", "")
            current_baseline = frame_res.get("vanilla_caption", "")
            current_detections = frame_res.get("detections", [])
            current_relations = frame_res.get("relations", [])

            # Next keyframe time defines subtitle window
            next_timestamp_sec = min(duration_sec, timestamp_sec + sample_rate)

            entry = {
                "frame_index": frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "end_sec": round(next_timestamp_sec, 3),
                "timestamp_str": timestamp_str,
                "caption": current_caption,
                "baseline_caption": current_baseline,
                "relations": [
                    f"{r['subject']} {r['predicate']} {r['object']} ({r.get('confidence', 0):.2f})"
                    for r in current_relations
                ],
                "detections": [
                    {"label": d.get("label"), "score": d.get("score"), "box": d.get("box")}
                    for d in current_detections
                ],
                "relation_used": frame_res.get("relation_used", False),
            }
            transcripts.append(entry)

            print(f"  -> Proposed Caption: {current_caption}")
            if current_baseline:
                print(f"  -> Baseline Caption: {current_baseline}")
            if current_relations:
                print(f"  -> Relations:        {len(current_relations)} detected")

        # Overlay annotations onto the frame if video rendering is enabled
        if video_writer is not None:
            annotated_frame = overlay_grounded_annotations(
                frame_bgr,
                detections=current_detections,
                relations=current_relations,
                caption=current_caption,
                timestamp_str=timestamp_str,
            )
            video_writer.write(annotated_frame)

        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()

    elapsed = time.time() - t_start

    # Export transcript JSON
    json_path = output_dir / f"{video_path.stem}_transcript.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2)

    # Export SRT subtitles
    srt_path = output_dir / f"{video_path.stem}_subtitles.srt"
    export_srt_subtitles(transcripts, srt_path, default_duration=sample_rate)

    print("\n" + "=" * 70)
    print(f"  Video processing complete in {elapsed:.2f}s!")
    print(f"  Sampled keyframes:  {len(transcripts)}")
    print(f"  Transcript report:  {json_path}")
    print(f"  Subtitles (.srt):   {srt_path}")
    if render_video:
        print(f"  Annotated video:    {output_video_path}")
    print("=" * 70)

    return {
        "video_path": str(video_path),
        "duration_sec": duration_sec,
        "fps": fps,
        "total_frames": total_frames,
        "sampled_frames_count": len(transcripts),
        "transcript": transcripts,
        "transcript_file": str(json_path),
        "srt_file": str(srt_path),
        "annotated_video": str(output_video_path) if render_video else None,
    }
