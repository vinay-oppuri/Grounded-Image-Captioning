"""demo_single_image.py: argument validation, weight loading and result rendering.

Everything here is model-free: the heavy pipeline stages (YOLO / CLIP / BLIP)
are never imported, so these tests run in milliseconds and need no GPU.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import demo_single_image as demo  # noqa: E402
try:
    from src.utils.caption_rerank import RerankWeights  # noqa: E402
except ImportError:
    from utils.caption_rerank import RerankWeights  # noqa: E402


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_missing_image_is_rejected(tmp_path):
    with pytest.raises(demo.DemoError, match="not found"):
        demo.validate_image_path(str(tmp_path / "nope.png"))


def test_unsupported_extension_is_rejected(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(demo.DemoError, match="unsupported image type"):
        demo.validate_image_path(str(p))


def test_supported_extension_is_accepted(tmp_path):
    p = tmp_path / "photo.PNG"
    p.write_bytes(b"\x89PNG")
    assert demo.validate_image_path(str(p)) == p


# ---------------------------------------------------------------------------
# device / output-dir guards
# ---------------------------------------------------------------------------

def test_resolve_device_defaults_to_cpu_when_forced():
    import torch

    assert demo.resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_rejects_bogus_name():
    with pytest.raises(demo.DemoError, match="invalid --device"):
        demo.resolve_device("bogus")


def test_guard_out_dir_refuses_results_caption():
    with pytest.raises(demo.DemoError, match="results_caption"):
        demo.guard_out_dir("results_caption")
    with pytest.raises(demo.DemoError, match="results_caption"):
        demo.guard_out_dir(os.path.join("results_caption", "main"))


def test_guard_out_dir_allows_demo_outputs():
    resolved = demo.guard_out_dir("demo_outputs")
    assert resolved.name == "demo_outputs"


# ---------------------------------------------------------------------------
# checkpoint selection / locked weights
# ---------------------------------------------------------------------------

def test_selected_checkpoint_missing_file():
    ck, reason = demo.load_selected_checkpoint("does/not/exist.json")
    assert ck is None and "not found" in reason


def test_selected_checkpoint_reads_dir(tmp_path):
    ck_dir = tmp_path / "geometry_clip_union_seed43"
    ck_dir.mkdir()
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"selected": {"checkpoint_dir": str(ck_dir)}}), encoding="utf-8")
    ck, reason = demo.load_selected_checkpoint(str(sel))
    assert reason is None and ck == str(ck_dir)


def test_selected_checkpoint_missing_dir(tmp_path):
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"selected": {"checkpoint_dir": str(tmp_path / "gone")}}),
                   encoding="utf-8")
    ck, reason = demo.load_selected_checkpoint(str(sel))
    assert ck is None and "missing" in reason


def test_locked_weights_fallback_when_file_absent():
    warnings = []
    weights = demo.load_locked_weights("no/such/weights.json", warnings)
    assert weights["object_reranked"].as_dict() == {"w_obj": 1.0, "w_hall": 0.0, "w_rel": 0.0}
    assert weights["relation_reranked"].as_dict() == {"w_obj": 1.0, "w_hall": 0.0, "w_rel": 0.5}
    assert warnings and "not found" in warnings[0]


def test_locked_weights_parsed_from_file(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text(json.dumps({"weights": {
        "object_reranked": {"w_obj": 0.5, "w_hall": 0.0, "w_rel": 0.0},
        "relation_reranked": {"w_obj": 0.5, "w_hall": 0.25, "w_rel": 1.0},
    }}), encoding="utf-8")
    warnings = []
    weights = demo.load_locked_weights(str(p), warnings)
    assert weights["object_reranked"].as_dict() == {"w_obj": 0.5, "w_hall": 0.0, "w_rel": 0.0}
    assert weights["relation_reranked"].as_dict() == {"w_obj": 0.5, "w_hall": 0.25, "w_rel": 1.0}
    assert warnings == []


def test_locked_weights_invalid_entries_fall_back(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text(json.dumps({"weights": {"object_reranked": {"w_obj": 1.0,
                                                             "w_hall": 0.0,
                                                             "w_rel": -1.0}}}),
                 encoding="utf-8")
    warnings = []
    weights = demo.load_locked_weights(str(p), warnings)
    assert weights["object_reranked"].as_dict() == {"w_obj": 1.0, "w_hall": 0.0, "w_rel": 0.0}
    assert weights["relation_reranked"].as_dict() == {"w_obj": 1.0, "w_hall": 0.0, "w_rel": 0.5}
    assert any("invalid" in w for w in warnings)


# ---------------------------------------------------------------------------
# result assembly / rendering
# ---------------------------------------------------------------------------

RAW = [{"label": "person", "box": [1.0, 2.0, 30.0, 40.0], "score": 0.9123}]
VERIFIED = [{"label": "person", "box": [1.0, 2.0, 30.0, 40.0], "score": 0.9123,
             "clip_similarity": 0.31, "verification_score": 0.61}]
RELATION = {"subject": "person", "predicate": "riding", "object": "bicycle",
            "confidence": 0.834, "subject_box": [1, 2, 3, 4], "top_k": []}
WEIGHTS = {"object_reranked": RerankWeights(1.0, 0.0, 0.0),
           "relation_reranked": RerankWeights(1.0, 0.0, 0.5)}


def _result(decision, selected=RELATION, warnings=()):
    return demo.build_result(
        image_path="demo_images/presentation.png",
        image_size=(640, 480),
        device="cpu",
        blip_dtype="float32",
        raw_detections=RAW,
        verified_detections=VERIFIED,
        selected_relation=selected,
        decision=decision,
        captions={"baseline": "a photo of a person",
                  "object_reranked": "a photo of a person and a bicycle",
                  "relation_reranked": "a photo of a person riding a bicycle"},
        weights=WEIGHTS,
        warnings=list(warnings),
        created_utc="2026-10-13T00:00:00Z",
    )


def test_build_result_shape():
    result = _result({"use_relation": True, "fallback_reason": None})
    assert result["image"]["size"] == [640, 480]
    assert result["detections"]["raw"][0]["score"] == 0.9123
    assert result["detections"]["verified"][0]["clip_similarity"] == 0.31
    assert result["relation"]["use_relation"] is True
    assert result["relation"]["selected"]["predicate"] == "riding"
    assert set(result["captions"]) == {"baseline", "object_reranked", "relation_reranked"}
    assert result["weights"]["relation_reranked"] == {"w_obj": 1.0, "w_hall": 0.0, "w_rel": 0.5}


def test_render_report_marks_used_relation():
    report = demo.render_report(_result({"use_relation": True, "fallback_reason": None}))
    assert "person riding bicycle" in report
    assert "[USED]" in report
    assert "baseline         : a photo of a person" in report
    assert "relation_reranked: a photo of a person riding a bicycle" in report


def test_render_report_shows_fallback_when_relation_unused():
    result = _result({"use_relation": False, "fallback_reason": "below_confidence_threshold"})
    report = demo.render_report(result)
    assert "below_confidence_threshold" in report
    assert "[USED]" not in report


def test_render_report_without_selected_relation():
    result = _result({"use_relation": False,
                      "fallback_reason": "fewer_than_2_eligible_detections"},
                     selected=None)
    report = demo.render_report(result)
    assert "none (fewer_than_2_eligible_detections)" in report
    assert result["relation"]["selected"] is None


def test_render_report_lists_warnings():
    report = demo.render_report(_result({"use_relation": False, "fallback_reason": "x"},
                                        warnings=["relation stage skipped: missing"]))
    assert "Warnings" in report
    assert "relation stage skipped: missing" in report


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def test_write_result_creates_file(tmp_path):
    result = _result({"use_relation": True, "fallback_reason": None})
    out = demo.write_result(result, str(tmp_path / "demo_outputs"))
    assert out.name == "presentation.demo.json"
    assert json.loads(out.read_text(encoding="utf-8"))["image"]["path"] == "demo_images/presentation.png"


def test_write_result_refuses_results_caption(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "PROJ_ROOT", tmp_path)
    (tmp_path / "results_caption").mkdir()
    result = _result({"use_relation": True, "fallback_reason": None})
    with pytest.raises(demo.DemoError, match="results_caption"):
        demo.write_result(result, str(tmp_path / "results_caption"))
