# Grounded Image Captioning

A production-grade pipeline for **relation-grounded image captioning** designed to reduce object hallucination in Vision-Language Models (BLIP) through explicit object detection, visual-semantic relationship reasoning, and evidence gating.

---

## 📁 Repository Structure

```
Grounded-Image-Captioning/
├── main.py                     # Unified production CLI entry point
├── configs/                    # System & training configuration files
│   ├── data.yaml               # Dataset paths, input/output directories
│   ├── model.yaml              # Detector, relation MLP & captioner architectures
│   └── train.yaml              # Hyperparameters (batch size, lr, epochs, seed)
├── input/                      # Input images for captioning workflow
├── results/                    # Generated captions, metadata, and visual outputs
├── checkpoints/                # Model weights & vocabulary mappings
│   ├── e2_geo/                 # Extended 19-D geometry checkpoints
│   ├── relation_mlp.pt         # Trained relation prediction weights
│   ├── label_vocab.json        # COCO object category vocabulary
│   └── pred_vocab.json         # Visual Genome predicate vocabulary
├── data/                       # Dataset storage (Visual Genome / COCO)
├── splits/                     # Frozen image-disjoint split manifests
├── docs/                       # Detailed experimental documentation & runbooks
│   ├── CAPTION_EXPERIMENTS.md  # Hallucination benchmark protocol (CHAIR, POPE)
│   ├── GPU_EXPERIMENTS.md      # Visual-semantic ablation protocol
│   └── report/                 # Project reports & publications
├── scripts/                    # Dataset preparation and utility scripts
│   ├── download_vg.py          # Visual Genome metadata downloader
│   ├── prepare_visual_genome.py# VG image downloader & integrity validator
│   ├── build_clip_cache.py     # Precomputes CLIP embeddings for object/union crops
│   ├── build_caption_eval_set.py# Builds frozen test set for caption evaluation
│   └── check_environment.py    # Environment and GPU readiness check
├── src/                        # Core codebase
│   ├── dataset/                # Dataset loaders & preprocessing
│   │   ├── coco_detection.py   # COCO 2017 detection dataset loader
│   │   └── vg_dataset.py       # Visual Genome relational dataset & feature extractor
│   ├── features/               # Visual feature extraction
│   │   ├── clip_extractor.py   # Frozen CLIP ViT-B/32 crop & union encoder
│   │   ├── clip_cache.py       # Efficient contiguous tensor cache storage
│   │   └── pose_extractor.py   # Pose feature extractor (optional)
│   ├── graph/                  # Graph construction
│   │   └── build_graph.py      # Spatial-semantic k-NN graph builder
│   ├── models/                 # Neural network architectures
│   │   ├── cnn.py              # ResNet backbone
│   │   ├── vit.py              # Vision Transformer backbone
│   │   ├── gnn.py              # Feature refinement Graph Convolutional Network
│   │   ├── detector.py         # HybridDetector (CNN + ViT + GNN)
│   │   ├── relation_mlp.py     # RelationMLP with visual-semantic support
│   │   └── relation_transformer.py # RelationTransformer architecture
│   ├── training/               # Model training modules
│   │   ├── losses.py           # Hungarian bipartite matching loss
│   │   ├── matcher.py          # Bipartite matching implementation
│   │   ├── train_detector.py   # HybridDetector training loop
│   │   └── train_relations.py  # Visual-semantic relation MLP trainer
│   ├── inference/              # Inference & caption pipelines
│   │   ├── predict_relations.py# Calibrated relation inference & selection
│   │   └── caption_pipeline.py # End-to-end grounded captioning pipeline
│   ├── evaluation/             # Metrics & evaluation protocols
│   │   ├── evaluate.py         # Unified multi-system evaluation
│   │   ├── eval_relations.py   # Frozen image-disjoint relation evaluation
│   │   ├── hallucination_eval.py# Hallucination benchmarks (CHAIR, POPE)
│   │   ├── caption_experiment.py# Automated caption ablation experiment runner
│   │   └── visual_experiment.py # GPU visual ablation runner (4 variants x 3 seeds)
│   └── utils/                  # Shared utilities
│       ├── blip_captioner.py   # BLIP caption generator with prefix conditioning
│       ├── clip_scorer.py      # CLIP-based caption reranking & scoring
│       ├── clipscore.py        # Reference-free CLIPScore calculation
│       ├── coco_mentions.py    # Entity detection & mention parser
│       ├── detection_verifier.py# CLIP cross-modal verification for YOLO detections
│       ├── metrics.py          # NLP evaluation metrics (BLEU, METEOR, SPICE)
│       ├── pope.py             # POPE probe generator & evaluator
│       ├── relation_corrector.py# Post-generation semantic relation correction
│       └── yolo_detector.py    # YOLOv11 wrapper & detection formatter
└── tests/                      # Automated test suite (326+ tests)
    └── conftest.py             # Test harness configuration
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/SmaranReddy/Image-Detection-using-GNNs.git
cd Grounded-Image-Captioning

# Setup virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Linux/macOS

# Install PyTorch with CUDA (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Verify Environment

```bash
python scripts/check_environment.py
```

### 3. Run Tests

```bash
python -m pytest tests/ -q
```
*(All 326 tests should pass).*

---

## 💡 Workflows

`main.py` serves as the single unified command-line interface for the entire codebase:

### Workflow 1: Generate Grounded Captions on Images

Place your input images in the `input/` directory:

```bash
# Caption all images in ./input and output results to ./results
python main.py caption

# Caption a single image
python main.py caption --input input/sample.jpg --output results/

# Generate side-by-side comparison with baseline ungrounded BLIP
python main.py caption --input input/sample.jpg --compare
```

Generated outputs in `results/`:
- `captions_report.json`: JSON document containing detected objects, predicted relations, and grounded captions.
- `<image_name>_annotated.jpg`: Annotated visual image showing bounding boxes, entity labels, and predicted interaction arrows.

---

### Workflow 2: Train the Hybrid Object Detector (CNN + ViT + GNN)

Train the multi-backbone detector on COCO:

```bash
python main.py train-detector --coco-root ./data/coco --epochs 30 --batch-size 4
```

---

### Workflow 3: Train the Visual-Semantic Relation Model

Train the relation MLP on Visual Genome annotations:

```bash
# Train geometry-only baseline
python main.py train-relations --vg-root ./data/visual_genome

# Train visual-semantic model (with CLIP crop features)
python main.py train-relations --vg-root ./data/visual_genome --use-visual
```

---

### Workflow 4: Benchmark & Hallucination Evaluation

```bash
# Evaluate on COCO validation images
python main.py evaluate --mode coco --coco-root ./data/coco --num-samples 100

# Run hallucination stress-test (CHAIR and POPE metrics)
python main.py evaluate --mode hallucination --image-dir input/
```

---

## ⚙️ Configuration Files

Configuration parameters are organized in `configs/`:
- `configs/data.yaml`: Dataset roots, split proportions, input/output paths.
- `configs/model.yaml`: Architecture dimensions (YOLO confidence, RelationMLP embedding & hidden dimensions, BLIP models).
- `configs/train.yaml`: Optimizer learning rates, batch sizes, worker counts, and seeds.

---

## 📄 License & Citations

See `docs/report/final_report.pdf` and `docs/CAPTION_EXPERIMENTS.md` for full research documentation and benchmarking results.
