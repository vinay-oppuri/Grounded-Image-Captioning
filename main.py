"""
main.py — Production CLI Entry Point for Grounded Image Captioning.

Workflows supported:
  1. caption          Run grounded caption inference on input image(s) -> results/
  2. train-relations  Train Visual-Semantic Relation MLP on Visual Genome dataset
  3. evaluate         Run benchmark evaluation of relation models on frozen test splits

Usage:
  # Caption all images in ./input/ and write output to ./results/
  python main.py caption

  # Caption a single image
  python main.py caption --input input/sample.jpg --output results/ --visualize

  # Train relation model on Visual Genome
  python main.py train-relations --vg-root ./data/visual_genome --use-visual

  # Evaluate relation model against benchmark
  python main.py evaluate --checkpoint-dir checkpoints/e2_geo
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

# Ensure project root is in sys.path
PROJ_ROOT = Path(__file__).resolve().parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Workflow 1: Grounded Captioning Inference
# ---------------------------------------------------------------------------

def run_caption_workflow(args: argparse.Namespace) -> int:
    """Run object detection, relation prediction, and grounded captioning."""
    from src.inference.caption_pipeline import run_grounded_pipeline, save_debug_visualization
    from src.inference import predict_relations as rel_predict

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: List[Path] = []

    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            images.append(input_path)
        else:
            print(f"[ERROR] Unsupported image format: {input_path}")
            return 1
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        )
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        print("  Place input images in './input/' or specify with --input <path>")
        return 1

    if not images:
        print(f"[INFO] No images found in {input_path}")
        print("  Place .jpg or .png images in './input/' to generate grounded captions.")
        return 0

    print("=" * 70)
    print("  GROUNDED IMAGE CAPTIONING WORKFLOW")
    print(f"  Input:   {input_path} ({len(images)} image(s))")
    print(f"  Output:  {output_dir}")
    print(f"  Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

    results = []
    t_start = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] Processing: {img_path.name}")
        try:
            res = run_grounded_pipeline(
                str(img_path),
                top_k_relations=args.top_k,
                temperature=args.temperature,
                compare_baseline=args.compare,
            )
            results.append(res)

            print("\n" + "=" * 65)
            print(f"  IMAGE CAPTIONING RESULTS: {img_path.name}")
            print("=" * 65)
            print(f"  [Proposed Grounded Model]: {res.get('caption', 'N/A')}")
            if res.get('vanilla_caption'):
                print(f"  [Baseline Vanilla BLIP  ]: {res.get('vanilla_caption')}")

            rels = res.get('relations', [])
            if rels:
                print(f"  [Grounded Relations     ]:")
                for r in rels:
                    conf = r.get('adjusted_confidence', r.get('confidence', 0))
                    print(f"    * {r['subject']} {r['predicate']} {r['object']} (conf: {conf:.2f})")
            else:
                reason = res.get('fallback_reason', 'none')
                print(f"  [Grounded Relations     ]: None ({reason})")
            print("=" * 65)

            # Save visual annotation if requested
            if args.visualize:
                vis_path = output_dir / f"{img_path.stem}_annotated.jpg"
                save_debug_visualization(res, str(vis_path))

        except Exception as e:
            print(f"  [ERROR] Failed processing {img_path.name}: {e}")

    # Save summary report
    report_file = output_dir / "captions_report.json"
    serializable = []
    for r in results:
        serializable.append({
            "image": str(r.get("image_path")),
            "caption": r.get("caption"),
            "baseline_caption": r.get("vanilla_caption", ""),
            "relations": [
                f"{rel['subject']} {rel['predicate']} {rel['object']} ({rel.get('confidence', 0):.2f})"
                for rel in r.get("relations", [])
            ],
            "detections": [
                {"label": d["label"], "score": d.get("score")}
                for d in r.get("detections", [])
            ],
            "relation_used": r.get("relation_used", False),
        })

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  Workflow complete! Processed {len(results)} images in {elapsed:.2f}s")
    print(f"  Report saved to: {report_file}")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# CLI Parser Setup
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Grounded Image Captioning: Production Pipeline & Training Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Workflow command to run")

    # 1. caption
    p_cap = subparsers.add_parser("caption", help="Run grounded image captioning on image(s)")
    p_cap.add_argument("--input", "-i", type=str, default="./input",
                       help="Path to input image file or directory (default: ./input)")
    p_cap.add_argument("--output", "-o", type=str, default="./results",
                       help="Directory to save generated captions and results (default: ./results)")
    p_cap.add_argument("--top-k", type=int, default=3,
                       help="Maximum relation triplets to evaluate per image (default: 3)")
    p_cap.add_argument("--temperature", type=float, default=2.0,
                       help="Temperature for relation prediction calibration (default: 2.0)")
    p_cap.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True,
                       help="Generate baseline vanilla BLIP caption for side-by-side comparison (default: True)")
    p_cap.add_argument("--visualize", action="store_true", default=True,
                       help="Save annotated detection and relationship image to output directory")

    # 2. train-relations
    p_tr = subparsers.add_parser("train-relations", help="Train Visual-Semantic Relation MLP on Visual Genome")
    p_tr.add_argument("--vg-root", type=str, default="./data/visual_genome",
                      help="Root directory of Visual Genome (default: ./data/visual_genome)")
    p_tr.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                      help="Directory to save relation checkpoints (default: ./checkpoints)")
    p_tr.add_argument("--use-visual", action="store_true", default=False,
                      help="Use CLIP visual embeddings from object crops")
    p_tr.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p_tr.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    p_tr.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_tr.add_argument("--seed", type=int, default=42, help="Random seed")

    # 3. evaluate
    p_ev = subparsers.add_parser("evaluate", help="Evaluate relation model checkpoints against frozen test split")
    p_ev.add_argument("--checkpoint-dir", type=str, default="checkpoints/e2_geo",
                      help="Directory containing relation checkpoint and vocabularies")
    p_ev.add_argument("--split-manifest", type=str, default="splits/e0_image_split.json",
                      help="Path to frozen split manifest")
    p_ev.add_argument("--results-dir", type=str, default="results",
                      help="Directory to write evaluation JSON")
    p_ev.add_argument("--output-name", type=str, default="benchmark_eval",
                      help="Name of evaluation output file")
    p_ev.add_argument("--force", action="store_true", default=True,
                      help="Overwrite existing evaluation file")

    return parser


def main() -> None:
    parser = build_parser()

    # Default to "caption" if no command specified but arguments or input exists
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n[TIP] Place images into the 'input/' folder and run 'python main.py caption'")
        return

    args, unknown = parser.parse_known_args()

    if args.command == "caption":
        sys.exit(run_caption_workflow(args))
    elif args.command == "train-relations":
        from src.training.train_relations import main as train_rel_main
        sys.exit(train_rel_main())
    elif args.command == "evaluate":
        from src.evaluation.eval_relations import main as eval_rel_main
        sys.exit(eval_rel_main())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
