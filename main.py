"""
main.py — Production CLI Entry Point for Grounded Image & Video Captioning.

Workflows supported:
  1. caption          Run grounded caption inference on input image(s) or video(s) -> results/
  2. train-relations  Train Visual-Semantic Relation MLP on Visual Genome dataset
  3. evaluate         Run benchmark evaluation of relation models on frozen test splits

Usage:
  # Caption all images and videos in ./input/ and write output to ./results/
  python main.py caption

  # Caption a single image
  python main.py caption --input input/sample.jpg --output results/ --visualize

  # Caption a video with 1 frame/sec sampling and render annotated video
  python main.py caption --input input/sample.mp4 --output results/ --sample-rate 1.0 --render-video

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
# Workflow 1: Grounded Captioning Inference (Images & Videos)
# ---------------------------------------------------------------------------

def run_caption_workflow(args: argparse.Namespace) -> int:
    """Run object detection, relation prediction, and grounded captioning on images or videos."""
    from src.inference.caption_pipeline import run_grounded_pipeline, save_debug_visualization
    from src.inference.video_pipeline import run_video_pipeline

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    images: List[Path] = []
    videos: List[Path] = []

    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix in image_extensions:
            images.append(input_path)
        elif suffix in video_extensions:
            videos.append(input_path)
        else:
            print(f"[ERROR] Unsupported media format: {input_path}")
            print(f"  Supported images: {sorted(image_extensions)}")
            print(f"  Supported videos: {sorted(video_extensions)}")
            return 1
    elif input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if p.is_file():
                s = p.suffix.lower()
                if s in image_extensions:
                    images.append(p)
                elif s in video_extensions:
                    videos.append(p)
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        print("  Place input images/videos in './input/' or specify with --input <path>")
        return 1

    if not images and not videos:
        print(f"[INFO] No images or videos found in {input_path}")
        print("  Place .jpg, .png, or .mp4 files in './input/' to generate grounded captions.")
        return 0

    # Process videos
    if videos:
        for idx, vid_path in enumerate(videos, 1):
            print(f"\n[{idx}/{len(videos)}] Processing Video: {vid_path.name}")
            try:
                run_video_pipeline(
                    video_path=vid_path,
                    output_dir=output_dir,
                    sample_rate=args.sample_rate,
                    render_video=args.render_video,
                    top_k_relations=args.top_k,
                    temperature=args.temperature,
                    compare_baseline=args.compare,
                )
            except Exception as e:
                print(f"  [ERROR] Failed processing video {vid_path.name}: {e}")

    # Process images
    if images:
        import demo_single_image as demo
        from src.utils.visualize import draw_relation_boxes

        print("=" * 70)
        print("  GROUNDED IMAGE CAPTIONING WORKFLOW")
        print(f"  Input:   {input_path} ({len(images)} image(s))")
        print(f"  Output:  {output_dir}")
        print(f"  Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("=" * 70)

        results = []
        t_start = time.time()

        for idx, img_path in enumerate(images, 1):
            try:
                class DemoArgs:
                    image = str(img_path)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    selection = "results_caption/relation_checkpoint_selection.json"
                    weights = "results_caption/rerank_weights.json"
                    blip_dtype = "float32"
                    num_candidates = 4
                    conf = 0.5
                    out_dir = str(output_dir)

                demo_res = demo.run(DemoArgs())
                demo_out_path = demo.write_result(demo_res, str(output_dir))
                results.append(demo_res)

                # Render exact format matching expected output
                print(demo.render_report(demo_res))
                print(f"\nwrote {demo_out_path}")

                # Save visual annotation if requested
                if args.visualize:
                    vis_path = output_dir / f"{img_path.stem}_annotated.jpg"
                    img_pil = Image.open(str(img_path)).convert("RGB")
                    sel_rel = demo_res.get("relation", {}).get("selected")
                    rels = [sel_rel] if sel_rel else []
                    cap_text = (
                        demo_res.get("captions", {}).get("relation_reranked")
                        or demo_res.get("captions", {}).get("object_reranked")
                        or demo_res.get("captions", {}).get("baseline")
                        or ""
                    )
                    draw_relation_boxes(
                        img_pil,
                        demo_res.get("detections", {}).get("verified", []),
                        rels,
                        str(vis_path),
                        caption=cap_text,
                    )

            except Exception as e:
                print(f"  [ERROR] Failed processing {img_path.name}: {e}")

        # Save summary report for images
        if results:
            report_file = output_dir / "captions_report.json"
            serializable = []
            for r in results:
                captions_dict = r.get("captions", {})
                sel_rel = r.get("relation", {}).get("selected")
                rel_list = (
                    [f"{sel_rel['subject']} {sel_rel['predicate']} {sel_rel['object']} ({sel_rel.get('confidence', 0):.2f})"]
                    if sel_rel else []
                )
                serializable.append({
                    "image": str(r.get("image", {}).get("path")),
                    "caption": captions_dict.get("relation_reranked") or captions_dict.get("object_reranked"),
                    "baseline_caption": captions_dict.get("baseline", ""),
                    "captions": captions_dict,
                    "relations": rel_list,
                    "detections": [
                        {"label": d["label"], "score": d.get("score")}
                        for d in r.get("detections", {}).get("verified", [])
                    ],
                    "relation_used": r.get("relation", {}).get("use_relation", False),
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
        description="Grounded Image & Video Captioning: Production Pipeline & Training Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Workflow command to run")

    # 1. caption
    p_cap = subparsers.add_parser("caption", help="Run grounded captioning on image(s) or video(s)")
    p_cap.add_argument("--input", "-i", type=str, default="./input",
                       help="Path to input image/video file or directory (default: ./input)")
    p_cap.add_argument("--output", "-o", type=str, default="./results",
                       help="Directory to save generated captions, transcripts, and videos (default: ./results)")
    p_cap.add_argument("--top-k", type=int, default=3,
                       help="Maximum relation triplets to evaluate per frame (default: 3)")
    p_cap.add_argument("--temperature", type=float, default=2.0,
                       help="Temperature for relation prediction calibration (default: 2.0)")
    p_cap.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True,
                       help="Generate baseline vanilla BLIP caption for side-by-side comparison (default: True)")
    p_cap.add_argument("--visualize", action="store_true", default=True,
                       help="Save annotated detection and relationship image to output directory")
    p_cap.add_argument("--sample-rate", type=float, default=1.0,
                       help="Frame sampling interval in seconds for videos (default: 1.0s)")
    p_cap.add_argument("--render-video", action=argparse.BooleanOptionalAction, default=True,
                       help="Render and save annotated output video with bounding boxes & subtitles (default: True)")

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
        print("\n[TIP] Place images/videos into the 'input/' folder and run 'python main.py caption'")
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
