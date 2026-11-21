"""Caption one arbitrary local image with the frozen grounded-caption pipeline.

This is an orchestration-only demo. Every stage delegates to the exact function
the frozen caption experiment used, so the demo cannot drift from measured
behaviour:

    YOLO11m detection
      -> CLIP semantic verification
      -> geometry+CLIP+union relation model (locked checkpoint)
      -> multi-candidate BLIP generation + uniform LM scoring
      -> deterministic reranking with the LOCKED weights

It prints, for a single image:

  * the verified objects the system believes it saw;
  * the predicted relation (if any) with its confidence and whether it was used;
  * three captions: BLIP's own baseline first candidate, the object-reranked
    arm and the relation-reranked arm.

It never opens an evaluation manifest, never reads human ground truth, and
never writes into results_caption/ (the experiment artifacts are read-only from
here). Results go to demo_outputs/<stem>.demo.json by default.

Usage:
    python demo_single_image.py --image demo_images/presentation.png

Nothing here fabricates output: if a model or checkpoint is unavailable, or the
GPU runs out of memory, the demo says so and degrades to what it can honestly
produce.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

PROJ_ROOT = Path(__file__).resolve().parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

DEFAULT_OUT_DIR = "demo_outputs"
DEFAULT_SELECTION = "results_caption/relation_checkpoint_selection.json"
DEFAULT_WEIGHTS = "results_caption/rerank_weights.json"
DEFAULT_CONF = 0.5
DEFAULT_TOP_K = 10
DEFAULT_NUM_CANDIDATES = 4
BLIP_DTYPE = "float32"

# Built-in fallback for the locked weights, byte-for-byte the values in
# results_caption/rerank_weights.json. Used only if that file cannot be read.
FALLBACK_WEIGHTS = {
    "object_reranked": {"w_obj": 1.0, "w_hall": 0.0, "w_rel": 0.0},
    "relation_reranked": {"w_obj": 1.0, "w_hall": 0.0, "w_rel": 0.5},
}


class DemoError(RuntimeError):
    """A user-facing failure with a message safe to print."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Input / device validation (pure - no models)
# ---------------------------------------------------------------------------

def validate_image_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_file():
        raise DemoError(f"image not found: {raw}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DemoError(
            f"unsupported image type {path.suffix!r} for {raw}; expected one of {supported}")
    return path


def resolve_device(name: Optional[str]):
    import torch

    if name:
        try:
            device = torch.device(name)
        except (RuntimeError, ValueError) as exc:
            raise DemoError(f"invalid --device {name!r}: {exc}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise DemoError("CUDA requested but no CUDA device is available; use --device cpu")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def guard_out_dir(out_dir: str) -> Path:
    """Refuse to write inside results_caption/ (the experiment artifacts)."""
    resolved = Path(out_dir).resolve()
    forbidden = (PROJ_ROOT / "results_caption").resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise DemoError(
            f"refusing to write into {forbidden} (experiment artifacts); "
            "choose a different --out-dir")
    return resolved


def _blip_dtype(name: str):
    import torch

    try:
        return {"float32": torch.float32, "float16": torch.float16}[name]
    except KeyError as exc:  # pragma: no cover - argparse restricts choices
        raise DemoError(f"unknown BLIP dtype {name!r}") from exc


# ---------------------------------------------------------------------------
# Checkpoint / weights loading (pure - no models)
# ---------------------------------------------------------------------------

def load_selected_checkpoint(selection_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (checkpoint_dir, None) or (None, reason) if it cannot be used."""
    path = Path(selection_path)
    if not path.is_file():
        return None, f"selection file {selection_path} not found"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"selection file {selection_path} unreadable: {exc}"
    ck_dir = (data.get("selected") or {}).get("checkpoint_dir")
    if not ck_dir:
        return None, f"selection file {selection_path} records no checkpoint_dir"
    if not Path(ck_dir).is_dir():
        return None, f"relation checkpoint {ck_dir} missing"
    return ck_dir, None


def load_locked_weights(weights_path: str, warnings: List[str]) -> Dict:
    """The two locked reranking arms, falling back to the built-in constants."""
    try:
        from src.utils.caption_rerank import RerankWeights
    except ImportError:
        from utils.caption_rerank import RerankWeights

    fallback = {arm: RerankWeights.from_dict(w) for arm, w in FALLBACK_WEIGHTS.items()}
    path = Path(weights_path)
    if not path.is_file():
        warnings.append(
            f"{weights_path} not found; using built-in locked weights {FALLBACK_WEIGHTS}")
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{weights_path} unreadable ({exc}); using built-in locked weights")
        return fallback

    out: Dict = {}
    for arm in ("object_reranked", "relation_reranked"):
        recorded = (data.get("weights") or {}).get(arm)
        if not recorded:
            warnings.append(f"{weights_path} has no {arm} weights; using built-in {arm}")
            out[arm] = fallback[arm]
            continue
        try:
            out[arm] = RerankWeights.from_dict(recorded)
        except (ValueError, TypeError) as exc:
            warnings.append(f"{weights_path} {arm} weights invalid ({exc}); using built-in")
            out[arm] = fallback[arm]
    return out


# ---------------------------------------------------------------------------
# Result assembly / rendering (pure - no models, fully testable)
# ---------------------------------------------------------------------------

def _clean_detection(det: Mapping) -> Dict:
    box = [round(float(v), 2) for v in det["box"]]
    entry = {"label": str(det["label"]), "box": box, "score": round(float(det["score"]), 4)}
    for key in ("clip_similarity", "verification_score"):
        if det.get(key) is not None:
            entry[key] = round(float(det[key]), 4)
    return entry


def build_result(*, image_path, image_size, device, blip_dtype: str,
                 raw_detections: Sequence[Mapping], verified_detections: Sequence[Mapping],
                 selected_relation: Optional[Mapping], decision: Mapping,
                 captions: Mapping, weights: Mapping, warnings: Sequence[str],
                 created_utc: str) -> Dict:
    relation_out = None
    if selected_relation:
        relation_out = {k: selected_relation[k]
                        for k in ("subject", "predicate", "object", "confidence") if k in selected_relation}
    return {
        "image": {"path": str(image_path).replace("\\", "/"), "size": list(image_size)},
        "device": str(device),
        "blip_dtype": blip_dtype,
        "detections": {
            "raw": [_clean_detection(d) for d in raw_detections],
            "verified": [_clean_detection(d) for d in verified_detections],
        },
        "relation": {
            "selected": relation_out,
            "use_relation": bool(decision.get("use_relation")),
            "fallback_reason": decision.get("fallback_reason"),
        },
        "captions": {k: captions.get(k) for k in ("baseline", "object_reranked", "relation_reranked")},
        "weights": {arm: w.as_dict() for arm, w in weights.items()},
        "warnings": list(warnings),
        "created_utc": created_utc,
    }


def render_report(result: Mapping) -> str:
    lines: List[str] = []
    img = result["image"]
    w, h = img["size"]
    lines.append("Demo single-image caption result")
    lines.append(f"  image      : {img['path']} ({w}x{h})")
    lines.append(f"  device     : {result['device']}")
    lines.append(f"  BLIP dtype : {result['blip_dtype']}")

    raw = result["detections"]["raw"]
    verified = result["detections"]["verified"]
    lines.append("")
    lines.append("Detections")
    lines.append(f"  raw ({len(raw)})")
    lines.append(f"  verified ({len(verified)})")
    for d in verified:
        extra = ""
        if d.get("clip_similarity") is not None:
            extra = f", clip={d['clip_similarity']:.2f}"
            if d.get("verification_score") is not None:
                extra += f", trust={d['verification_score']:.2f}"
        lines.append(f"    - {d['label']} ({d['score']:.2f}{extra})")
    if not verified:
        lines.append("    (none)")

    rel = result["relation"]
    lines.append("")
    lines.append("Predicted relation")
    sel = rel.get("selected")
    if sel:
        state = "USED" if rel["use_relation"] else (rel.get("fallback_reason") or "not used")
        lines.append(f"  {sel['subject']} {sel['predicate']} {sel['object']} "
                     f"(confidence {float(sel['confidence']):.3f})  [{state}]")
    else:
        lines.append(f"  none ({rel.get('fallback_reason') or 'no relation predicted'})")

    lines.append("")
    lines.append("Captions")
    captions = result["captions"]
    for arm in ("baseline", "object_reranked", "relation_reranked"):
        text = captions.get(arm)
        lines.append(f"  {arm:<17}: {text if text is not None else '(unavailable)'}")

    if result.get("warnings"):
        lines.append("")
        lines.append("Warnings")
        for wmsg in result["warnings"]:
            lines.append(f"  ! {wmsg}")
    return "\n".join(lines)


def write_result(result: Mapping, out_dir: str) -> Path:
    root = guard_out_dir(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = Path(result["image"]["path"]).stem
    out_path = root / f"{stem}.demo.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Pipeline (loads models)
# ---------------------------------------------------------------------------

def _detect(image, device, conf: float) -> List[Dict]:
    try:
        from src.utils.yolo_detector import format_detections, load_model
    except ImportError:
        from utils.yolo_detector import format_detections, load_model

    yolo = load_model(str(device))
    raw = yolo(image, verbose=False)
    return format_detections(raw, conf_thres=conf, top_k=DEFAULT_TOP_K)


def _predict_relation(image, verified, ck_dir, device):
    try:
        from src.features.clip_extractor import CLIPExtractor
        from src.utils.caption_relations import (RELATION_CONFIDENCE_THRESHOLD, build_pair_batch,
                                              eligible_detections, load_relation_bundle,
                                              predict_pairs, relation_decision, select_relation)
    except ImportError:
        from relation_prediction.clip_extractor import CLIPExtractor
        from utils.caption_relations import (RELATION_CONFIDENCE_THRESHOLD, build_pair_batch,
                                              eligible_detections, load_relation_bundle,
                                              predict_pairs, relation_decision, select_relation)

    bundle = load_relation_bundle(ck_dir, device=device)
    eligible, _dropped = eligible_detections(verified, bundle.label_vocab, image.size)
    batch = build_pair_batch(image, eligible, bundle, CLIPExtractor(device).encode_crops)
    preds = predict_pairs(bundle, batch)
    selected, _annotated = select_relation(eligible, preds)
    decision = relation_decision(selected, len(eligible), len(preds),
                                 RELATION_CONFIDENCE_THRESHOLD)
    return selected, decision


def run(args) -> Dict:
    import torch
    from PIL import Image

    image_path = validate_image_path(args.image)
    device = resolve_device(args.device)
    warnings: List[str] = []

    image = Image.open(image_path).convert("RGB")

    # 1. Detections (GPU OOM here is fatal - there is nothing honest to show).
    try:
        raw_detections = _detect(image, device, args.conf)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise DemoError(f"CUDA out of memory during detection ({exc}); retry with --device cpu") from exc

    try:
        from src.utils.detection_verifier import verify_detections
    except ImportError:
        from utils.detection_verifier import verify_detections
    verified_detections = verify_detections(copy.deepcopy(raw_detections), image, debug=False)

    # 2. Relation model (optional: if unavailable, degrade to baseline only).
    selected_relation = None
    decision = {"use_relation": False, "fallback_reason": "relation_model_unavailable"}
    ck_dir, ck_reason = load_selected_checkpoint(args.selection)
    if ck_reason:
        warnings.append(f"relation stage skipped: {ck_reason}")
    else:
        try:
            selected_relation, decision = _predict_relation(
                image, verified_detections, ck_dir, device)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            warnings.append(f"relation stage skipped: CUDA OOM ({exc}); retry with --device cpu")
            decision = {"use_relation": False, "fallback_reason": "cuda_oom"}
        except Exception as exc:  # checkpoint/signature mismatch etc.
            warnings.append(f"relation stage skipped: {type(exc).__name__}: {exc}")
            decision = {"use_relation": False, "fallback_reason": "relation_stage_failed"}

    # 3. Captions + reranking with the locked weights.
    weights = load_locked_weights(args.weights, warnings)
    captions = {"baseline": None, "object_reranked": None, "relation_reranked": None}
    try:
        try:
            from src.utils.caption_candidates import generate_and_score
            from src.utils.caption_rerank import evidence_objects, first_candidate, rerank
        except ImportError:
            from utils.caption_candidates import generate_and_score
            from utils.caption_rerank import evidence_objects, first_candidate, rerank

        generated = generate_and_score(image, selected_relation,
                                       num_candidates=args.num_candidates,
                                       dtype=_blip_dtype(args.blip_dtype))
        pool = generated["candidates"]
        evidence = evidence_objects(verified_detections)
        first = first_candidate(pool)
        captions["baseline"] = first["text"] if first else None
        captions["object_reranked"] = rerank(pool, evidence, selected_relation,
                                             weights["object_reranked"])["caption"]
        captions["relation_reranked"] = rerank(pool, evidence, selected_relation,
                                               weights["relation_reranked"])["caption"]
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        warnings.append(f"caption stage skipped: CUDA OOM ({exc}); retry with --device cpu")
    except (ValueError, RuntimeError) as exc:
        warnings.append(f"caption stage skipped: {type(exc).__name__}: {exc}")

    return build_result(
        image_path=image_path,
        image_size=image.size,
        device=device,
        blip_dtype=args.blip_dtype,
        raw_detections=raw_detections,
        verified_detections=verified_detections,
        selected_relation=selected_relation,
        decision=decision,
        captions=captions,
        weights=weights,
        warnings=warnings,
        created_utc=_now(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Caption one local image with the frozen grounded-caption pipeline.")
    ap.add_argument("--image", required=True,
                    help="path to a JPG/PNG/BMP/WEBP image to caption")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available else cpu)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help=f"where to write <stem>.demo.json (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--selection", default=DEFAULT_SELECTION,
                    help="relation checkpoint selection file (read-only)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="locked rerank weights file (read-only)")
    ap.add_argument("--blip-dtype", choices=("float32", "float16"), default=BLIP_DTYPE,
                    help="float32 default: GTX 16xx cards are prone to fp16 NaNs")
    ap.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES,
                    help="beams returned per prefix (must be <= BLIP num_beams=4)")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF,
                    help=f"YOLO confidence threshold (default {DEFAULT_CONF})")
    args = ap.parse_args(argv)

    try:
        if args.num_candidates < 1:
            raise DemoError(f"--num-candidates must be >= 1, got {args.num_candidates}")
        result = run(args)
        out_path = write_result(result, args.out_dir)
    except DemoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_report(result))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    try:
        from src.utils.console import configure_safe_stdio
    except ImportError:
        from utils.console import configure_safe_stdio
    configure_safe_stdio()
    raise SystemExit(main())
