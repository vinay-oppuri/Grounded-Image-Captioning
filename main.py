"""
main.py — Production CLI Entry Point for Grounded Image Captioning.

Workflows supported:
  1. caption          Run grounded caption inference on input image(s) -> results/
  2. train-detector   Train HybridDetector (CNN + ViT + GNN) on COCO dataset
  3. train-relations  Train Visual-Semantic Relation MLP on Visual Genome dataset
  4. evaluate         Run benchmark evaluation across models (BLEU, CHAIR, POPE)

Usage:
  # Caption all images in ./input/ and write output to ./results/
  python main.py caption

  # Caption a single image
  python main.py caption --image input/sample.jpg --output results/

  # Train detector on COCO
  python main.py train-detector --coco-root ./data/coco --epochs 30

  # Train relation model on Visual Genome
  python main.py train-relations --vg-root ./data/visual_genome --use-visual

  # Evaluate captioning systems
  python main.py evaluate --mode coco --coco-root ./data/coco
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
# Workflow 2: Train Hybrid Detector
# ---------------------------------------------------------------------------

def run_train_detector_workflow(args: argparse.Namespace) -> int:
    """Train the CNN + ViT + GNN HybridDetector on COCO."""
    from torch.utils.data import DataLoader, Subset
    from src.models.detector import HybridDetector
    from src.dataset.coco_detection import COCODetectionDataset, collate_fn
    from src.training.train_detector import train_detector

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_dir = os.path.join(args.coco_root, "train2017")
    ann_file = os.path.join(args.coco_root, "annotations", "instances_train2017.json")

    for path, label in [(img_dir, "COCO images"), (ann_file, "COCO annotations")]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found at: {path}")
            print("  Please download COCO 2017 to the specified --coco-root directory.")
            return 1

    print("Loading COCO dataset...")
    full_dataset = COCODetectionDataset(image_dir=img_dir, annotation_file=ann_file)
    print(f"  Total images available: {len(full_dataset):,}")

    train_size = min(args.train_size, len(full_dataset))
    val_size = min(args.val_size, len(full_dataset) - train_size)
    rng = random.Random(args.seed)
    indices = list(range(len(full_dataset)))
    rng.shuffle(indices)

    train_ds = Subset(full_dataset, indices[:train_size])
    val_ds = Subset(full_dataset, indices[train_size:train_size + val_size])

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn,
    )

    model = HybridDetector(num_classes=5)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("\nStarting detector training...")
    train_detector(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
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

    # 2. train-detector
    p_td = subparsers.add_parser("train-detector", help="Train HybridDetector on COCO")
    p_td.add_argument("--coco-root", type=str, default="./data/coco",
                      help="Root directory of COCO dataset (default: ./data/coco)")
    p_td.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                      help="Directory to save model checkpoints (default: ./checkpoints)")
    p_td.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    p_td.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    p_td.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    p_td.add_argument("--train-size", type=int, default=16000, help="Number of training samples")
    p_td.add_argument("--val-size", type=int, default=2000, help="Number of validation samples")
    p_td.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p_td.add_argument("--seed", type=int, default=42, help="Random seed")

    # 3. train-relations
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

    # 4. evaluate
    p_ev = subparsers.add_parser("evaluate", help="Run benchmark evaluation (BLEU, CHAIR, POPE)")
    p_ev.add_argument("--mode", choices=["coco", "custom", "hallucination"], default="custom",
                      help="Evaluation data source mode")
    p_ev.add_argument("--image-dir", type=str, default="input",
                      help="Directory containing test images")
    p_ev.add_argument("--output-dir", type=str, default="results",
                      help="Output directory for evaluation results")
    p_ev.add_argument("--coco-root", type=str, default="./data/coco",
                      help="COCO dataset root directory")
    p_ev.add_argument("--num-samples", type=int, default=None,
                      help="Limit number of images to evaluate")

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
    elif args.command == "train-detector":
        sys.exit(run_train_detector_workflow(args))
    elif args.command == "train-relations":
        from src.training.train_relations import main as train_rel_main
        sys.exit(train_rel_main())
    elif args.command == "evaluate":
        from src.evaluation.evaluate import main as eval_main
        sys.exit(eval_main())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
