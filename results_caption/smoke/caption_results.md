# Caption experiment: baseline vs relation-grounded BLIP

- images: **5** (frozen test split, eval set `splits/caption_eval_test_250.json`, ids sha256 `541d93c6e2496822`)
- relation checkpoint: `checkpoints_gpu/geometry_clip_union_seed43` (selected by validation top-1: max validation top-1 (training_meta.val_acc); tie -> max val_macro_f1; tie -> lowest seed. Test metrics are not read.)
- relation confidence threshold: **0.5**
- arms: baseline, grounded, objects_only, object_reranked, relation_reranked

## Validity checks

| arm | images | relation used | fallback | prefix echoed | caption differs from baseline (used) | fallback regeneration != baseline |
|---|---|---|---|---|---|---|
| baseline | 5 | 0 | 0 | 5 | 0 | 0 |
| grounded | 5 | 5 | 0 | 5 | 5 | 0 |
| objects_only | 5 | 5 | 0 | 5 | 5 | 0 |
| object_reranked | 5 | 1 | 0 | 5 | 2 | 0 |
| relation_reranked | 5 | 2 | 0 | 5 | 3 | 0 |

All prefixes matched their rule; no relation was dropped before BLIP; no placeholder captions.

## Relation usage

- images with >=1 YOLO detection: 5/5
- mean detections per image: raw 6.00, CLIP-verified 5.60
- images with >=2 eligible detections: 5
- images with a selected relation: 5
- **relation usage rate: 100.00%** (fallback to baseline prefix: 0.00%)
- fallback reasons: {}
- mean confidence of used relations: 0.862
- predicates used: {'on': 2, 'under': 2, 'near': 1}

## Reranking behaviour (descriptive, not a quality metric)

**object_reranked** - weights `{'w_obj': 0.5, 'w_hall': 0.5, 'w_rel': 0.0}` (locked on 0 none-split images, file sha256 `9a864b41cd8abf9b`)

| statistic | value |
|---|---|
| candidates per image (K per prefix, pooled) | min 12, median 12, max 12, mean 12.00 |
| candidate generation success rate | 100.00% |
| candidates dropped (degenerate) | 0  |
| selected score | min -1.179, median -0.487, max 0.006, mean -0.542 |
| margin over runner-up | median 0.0520, mean 0.1266 |
| selected candidate came from prefix | {'baseline': 4, 'relation': 1} |
| **top-ranked != BLIP's first candidate** | 2/5 (40.00%) |
| relation evidence changed the selection | n/a (this arm has no relation term) |
| captions stating the predicted relation | 1/5 (20.00%) |
| mean confidence of stated relations | 0.959 |

**relation_reranked** - weights `{'w_obj': 0.5, 'w_hall': 0.5, 'w_rel': 1.0}` (locked on 0 none-split images, file sha256 `9a864b41cd8abf9b`)

| statistic | value |
|---|---|
| candidates per image (K per prefix, pooled) | min 12, median 12, max 12, mean 12.00 |
| candidate generation success rate | 100.00% |
| candidates dropped (degenerate) | 0  |
| selected score | min -0.584, median -0.288, max 0.492, mean -0.172 |
| margin over runner-up | median 0.0981, mean 0.2085 |
| selected candidate came from prefix | {'relation': 2, 'baseline': 3} |
| **top-ranked != BLIP's first candidate** | 3/5 (60.00%) |
| **relation evidence changed the selection** | 1/5 (20.00%) |
| captions stating the predicted relation | 2/5 (40.00%) |
| mean confidence of stated relations | 0.942 |

## Hallucination metrics - all images

| metric | baseline | grounded | objects_only | object_reranked | relation_reranked |
|---|---|---|---|---|---|
| CHAIR_i (lower better) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| CHAIR_s (lower better) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| hallucinated objects / caption | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mentioned objects / caption | 1.200 | 2.000 | 2.000 | 1.800 | 1.800 |
| object recall | 35.29% | 58.82% | 58.82% | 52.94% | 52.94% |
| caption length (words) | 9.80 | 9.40 | 11.80 | 10.80 | 10.80 |
| POPE-random accuracy | 69.23% | 80.77% | 80.77% | 76.92% | 80.77% |
| POPE-random F1 | 55.56% | 76.19% | 76.19% | 70.00% | 76.19% |
| POPE-random false-positive rate on absent objects | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| POPE-random yes-ratio | 19.23% | 30.77% | 30.77% | 26.92% | 30.77% |
| POPE-popular accuracy | 69.23% | 80.77% | 80.77% | 76.92% | 80.77% |
| POPE-popular F1 | 55.56% | 76.19% | 76.19% | 70.00% | 76.19% |
| POPE-popular false-positive rate on absent objects | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| POPE-popular yes-ratio | 19.23% | 30.77% | 30.77% | 26.92% | 30.77% |
| POPE-adversarial accuracy | 69.23% | 80.77% | 80.77% | 76.92% | 80.77% |
| POPE-adversarial F1 | 55.56% | 76.19% | 76.19% | 70.00% | 76.19% |
| POPE-adversarial false-positive rate on absent objects | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| POPE-adversarial yes-ratio | 19.23% | 30.77% | 30.77% | 26.92% | 30.77% |
| hallucinations introduced by injected prefix | 0 | 0 | 0 | 0 | 0 |
| hallucinations in BLIP continuation | 0 | 0 | 0 | 0 | 0 |

## Hallucination metrics - images where a relation was injected (n=5)

| metric | baseline | grounded | objects_only | object_reranked | relation_reranked |
|---|---|---|---|---|---|
| CHAIR_i (lower better) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| CHAIR_s (lower better) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| hallucinated objects / caption | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mentioned objects / caption | 1.200 | 2.000 | 2.000 | 1.800 | 1.800 |
| object recall | 35.29% | 58.82% | 58.82% | 52.94% | 52.94% |
| caption length (words) | 9.80 | 9.40 | 11.80 | 10.80 | 10.80 |
| POPE-random accuracy | 69.23% | 80.77% | 80.77% | 76.92% | 80.77% |
| POPE-random F1 | 55.56% | 76.19% | 76.19% | 70.00% | 76.19% |
| POPE-random false-positive rate on absent objects | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| POPE-random yes-ratio | 19.23% | 30.77% | 30.77% | 26.92% | 30.77% |
| POPE-popular accuracy | 69.23% | 80.77% | 80.77% | 76.92% | 80.77% |
| POPE-popular F1 | 55.56% | 76.19% | 76.19% | 70.00% | 76.19% |
| POPE-popular false-positive rate on absent objects | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| POPE-popular yes-ratio | 19.23% | 30.77% | 30.77% | 26.92% | 30.77% |
| POPE-adversarial accuracy | 69.23% | 80.77% | 80.77% | 76.92% | 80.77% |
| POPE-adversarial F1 | 55.56% | 76.19% | 76.19% | 70.00% | 76.19% |
| POPE-adversarial false-positive rate on absent objects | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| POPE-adversarial yes-ratio | 19.23% | 30.77% | 30.77% | 26.92% | 30.77% |

## Paired differences (95% bootstrap CI over images)

**grounded_minus_baseline**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +23.53 pt [+5.55, +43.75] |
| POPE-adv F1 | +20.63 pt [+5.79, +37.43] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | -0.01  [-0.08, +0.06] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 0, 'grounded_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**grounded_minus_baseline**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +23.53 pt [+5.55, +43.75] |
| POPE-adv F1 | +20.63 pt [+5.79, +37.43] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | -0.01  [-0.08, +0.06] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 0, 'grounded_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**objects_only_minus_baseline**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +23.53 pt [+5.55, +43.75] |
| POPE-adv F1 | +20.63 pt [+5.79, +37.43] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | -0.01  [-0.09, +0.06] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 0, 'objects_only_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**objects_only_minus_baseline**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +23.53 pt [+5.55, +43.75] |
| POPE-adv F1 | +20.63 pt [+5.79, +37.43] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | -0.01  [-0.09, +0.06] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 0, 'objects_only_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**grounded_minus_objects_only**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +0.00 pt [+0.00, +0.00] |
| POPE-adv F1 | +0.00 pt [+0.00, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | -0.00  [-0.03, +0.02] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 0, 'grounded_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**grounded_minus_objects_only**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +0.00 pt [+0.00, +0.00] |
| POPE-adv F1 | +0.00 pt [+0.00, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | -0.00  [-0.03, +0.02] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 0, 'grounded_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**object_reranked_minus_baseline**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +17.65 pt [+0.00, +38.12] |
| POPE-adv F1 | +14.44 pt [+0.00, +32.50] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.04  [+0.00, +0.08] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 0, 'object_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**object_reranked_minus_baseline**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +17.65 pt [+0.00, +38.12] |
| POPE-adv F1 | +14.44 pt [+0.00, +32.50] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.04  [+0.00, +0.08] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 0, 'object_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**object_reranked_minus_objects_only**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | -5.88 pt [-14.29, +0.00] |
| POPE-adv F1 | -6.19 pt [-18.26, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.05  [+0.00, +0.10] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 0, 'object_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**object_reranked_minus_objects_only**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | -5.88 pt [-14.29, +0.00] |
| POPE-adv F1 | -6.19 pt [-18.26, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.05  [+0.00, +0.10] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 0, 'object_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**object_reranked_minus_grounded**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | -5.88 pt [-14.29, +0.00] |
| POPE-adv F1 | -6.19 pt [-18.26, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.06  [-0.00, +0.11] |

CHAIR_s discordant pairs: {'grounded_hallucinates_only': 0, 'object_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**object_reranked_minus_grounded**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | -5.88 pt [-14.29, +0.00] |
| POPE-adv F1 | -6.19 pt [-18.26, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.06  [-0.00, +0.11] |

CHAIR_s discordant pairs: {'grounded_hallucinates_only': 0, 'object_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**relation_reranked_minus_baseline**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +17.65 pt [+0.00, +38.12] |
| POPE-adv F1 | +20.63 pt [+5.79, +37.43] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.03  [-0.04, +0.08] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 0, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**relation_reranked_minus_baseline**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +17.65 pt [+0.00, +38.12] |
| POPE-adv F1 | +20.63 pt [+5.79, +37.43] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.03  [-0.04, +0.08] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 0, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**relation_reranked_minus_objects_only**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | -5.88 pt [-14.29, +0.00] |
| POPE-adv F1 | +0.00 pt [+0.00, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.04  [+0.00, +0.07] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 0, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**relation_reranked_minus_objects_only**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | -5.88 pt [-14.29, +0.00] |
| POPE-adv F1 | +0.00 pt [+0.00, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.04  [+0.00, +0.07] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 0, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**relation_reranked_minus_grounded**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | -5.88 pt [-14.29, +0.00] |
| POPE-adv F1 | +0.00 pt [+0.00, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.04  [-0.01, +0.09] |

CHAIR_s discordant pairs: {'grounded_hallucinates_only': 0, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**relation_reranked_minus_grounded**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | -5.88 pt [-14.29, +0.00] |
| POPE-adv F1 | +0.00 pt [+0.00, +0.00] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | +0.04  [-0.01, +0.09] |

CHAIR_s discordant pairs: {'grounded_hallucinates_only': 0, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**relation_reranked_minus_object_reranked**, all_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +0.00 pt [+0.00, +0.00] |
| POPE-adv F1 | +6.19 pt [+0.00, +18.26] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | -0.02  [-0.05, +0.00] |

CHAIR_s discordant pairs: {'object_reranked_hallucinates_only': 0, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

**relation_reranked_minus_object_reranked**, relation_used_images (n=5)

| metric | difference |
|---|---|
| CHAIR_i | +0.00 pt [+0.00, +0.00] |
| CHAIR_s | +0.00 pt [+0.00, +0.00] |
| halluc./caption | +0.00 obj [+0.00, +0.00] |
| object recall | +0.00 pt [+0.00, +0.00] |
| POPE-adv F1 | +6.19 pt [+0.00, +18.26] |
| POPE-adv FP rate | +0.00 pt [+0.00, +0.00] |
| CLIPScore | -0.02  [-0.05, +0.00] |

CHAIR_s discordant pairs: {'object_reranked_hallucinates_only': 0, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 1.0} 

## Caption quality (reference-free)

CLIPScore = 2.5 * max(cos(image, 'A photo depicts ' + caption), 0), CLIP ViT-B/32. Image-text alignment, NOT a hallucination metric and not reference-based.

| arm | CLIPScore |
|---|---|
| baseline | 0.7131 |
| grounded | 0.7015 |
| objects_only | 0.7036 |
| object_reranked | 0.7575 |
| relation_reranked | 0.7396 |

## EXPLORATORY threshold sweep (not the primary result)

| threshold | usage | CHAIR_i | CHAIR_s | POPE-adv F1 |
|---|---|---|---|---|
| 0.0 | 100.00% | 0.00% | 0.00% | 76.19% |
| 0.3 | 100.00% | 0.00% | 0.00% | 76.19% |
| 0.5 | 100.00% | 0.00% | 0.00% | 76.19% |
| 0.7 | 100.00% | 0.00% | 0.00% | 76.19% |
| 0.9 | 60.00% | 0.00% | 0.00% | 70.00% |

## Notes

- CHAIR/POPE ground truth: human Visual Genome annotations mapped to COCO-80; not exhaustive, so absolute hallucination rates are upper bounds for every arm.
- Fallback images are scored with the baseline arm's caption (same prefix).
- The threshold sweep reuses the grounded arm's relation captions and is exploratory; the primary result is the pre-declared threshold.
- CLIPScore is reference-free image-text alignment (Hessel et al., 2021), not a reference-based caption quality metric; no reference captions exist for these images, so BLEU/CIDEr/SPICE are not reported.

No conclusion is written automatically. Read the paired differences against the pre-registered rules in CAPTION_EXPERIMENTS.md.
