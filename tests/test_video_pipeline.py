"""
Unit and integration tests for the Grounded Video Captioning Pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.inference.video_pipeline import (
    format_timestamp,
    export_srt_subtitles,
    overlay_grounded_annotations,
    run_video_pipeline,
)


def test_format_timestamp():
    # Standard format MM:SS.mmm
    assert format_timestamp(0.0) == "00:00.000"
    assert format_timestamp(65.432) == "01:05.432"

    # SRT format HH:MM:SS,mmm
    assert format_timestamp(0.0, srt_format=True) == "00:00:00,000"
    assert format_timestamp(3665.123, srt_format=True) == "01:01:05,123"


def test_export_srt_subtitles(tmp_path: Path):
    srt_file = tmp_path / "test.srt"
    transcripts = [
        {
            "timestamp_sec": 0.0,
            "end_sec": 1.5,
            "caption": "A person riding a bicycle.",
        },
        {
            "timestamp_sec": 1.5,
            "end_sec": 3.0,
            "caption": "The person standing near a bus.",
        },
    ]

    export_srt_subtitles(transcripts, srt_file)
    assert srt_file.exists()

    content = srt_file.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:01,500\nA person riding a bicycle." in content
    assert "2\n00:00:01,500 --> 00:00:03,000\nThe person standing near a bus." in content


def test_overlay_grounded_annotations():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = [
        {"label": "person", "box": [20, 20, 100, 150], "score": 0.95},
        {"label": "bicycle", "box": [80, 100, 200, 200], "score": 0.88},
    ]
    relations = [
        {
            "subject": "person",
            "predicate": "riding",
            "object": "bicycle",
            "confidence": 0.91,
            "subject_box": [20, 20, 100, 150],
            "object_box": [80, 100, 200, 200],
        }
    ]

    annotated = overlay_grounded_annotations(
        frame_bgr=frame,
        detections=detections,
        relations=relations,
        caption="A person is riding a bicycle.",
        timestamp_str="00:01.000",
    )

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    # Ensure drawing changed pixels from pure black
    assert annotated.sum() > 0


def test_run_video_pipeline(tmp_path: Path):
    video_path = tmp_path / "sample_test.mp4"
    output_dir = tmp_path / "results"

    # Create a 6-frame synthetic video at 3 FPS (2 seconds total)
    width, height = 160, 120
    fps = 3.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(6):
        frame = np.full((height, width, 3), fill_value=i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert video_path.exists()

    # Mock run_grounded_pipeline to run test quickly without GPU/model loading overhead
    mock_res = {
        "caption": "A person looking at a bus.",
        "vanilla_caption": "A person and a bus.",
        "detections": [
            {"label": "person", "score": 0.92, "box": [10, 10, 50, 80]},
            {"label": "bus", "score": 0.89, "box": [60, 20, 140, 100]},
        ],
        "relations": [
            {
                "subject": "person",
                "predicate": "looking at",
                "object": "bus",
                "confidence": 0.85,
                "subject_box": [10, 10, 50, 80],
                "object_box": [60, 20, 140, 100],
            }
        ],
        "relation_used": True,
    }

    with patch("src.inference.video_pipeline.run_grounded_pipeline", return_value=mock_res):
        out = run_video_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            sample_rate=1.0,  # 1 FPS -> 2 keyframes (frame 0 and frame 3)
            render_video=True,
        )

    assert out["total_frames"] == 6
    assert out["fps"] == pytest.approx(3.0)
    assert out["sampled_frames_count"] == 2
    assert len(out["transcript"]) == 2

    transcript_path = Path(out["transcript_file"])
    assert transcript_path.exists()
    transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert len(transcript_data) == 2
    assert transcript_data[0]["caption"] == "A person looking at a bus."

    srt_path = Path(out["srt_file"])
    assert srt_path.exists()
    assert "A person looking at a bus." in srt_path.read_text(encoding="utf-8")

    annotated_video = Path(out["annotated_video"])
    assert annotated_video.exists()
    assert annotated_video.stat().st_size > 0
