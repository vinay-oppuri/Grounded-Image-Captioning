# Caption experiment: baseline vs relation-grounded BLIP

- images: **250** (frozen test split, eval set `splits/caption_eval_test_250.json`, ids sha256 `541d93c6e2496822`)
- relation checkpoint: `checkpoints_gpu/geometry_clip_union_seed43` (selected by validation top-1: max validation top-1 (training_meta.val_acc); tie -> max val_macro_f1; tie -> lowest seed. Test metrics are not read.)
- relation confidence threshold: **0.5**
- arms: baseline, grounded, objects_only, object_reranked, relation_reranked

## Validity checks

| arm | images | relation used | fallback | prefix echoed | caption differs from baseline (used) | fallback regeneration != baseline |
|---|---|---|---|---|---|---|
| baseline | 250 | 0 | 0 | 250 | 0 | 0 |
| grounded | 250 | 227 | 23 | 250 | 223 | 0 |
| objects_only | 250 | 227 | 23 | 250 | 227 | 0 |
| object_reranked | 250 | 81 | 0 | 250 | 157 | 0 |
| relation_reranked | 250 | 130 | 0 | 250 | 167 | 0 |

All prefixes matched their rule; no relation was dropped before BLIP; no placeholder captions.

## Relation usage

- images with >=1 YOLO detection: 250/250
- mean detections per image: raw 4.76, CLIP-verified 4.70
- images with >=2 eligible detections: 235
- images with a selected relation: 235
- **relation usage rate: 90.80%** (fallback to baseline prefix: 9.20%)
- fallback reasons: {'fewer_than_2_eligible_detections': 15, 'below_confidence_threshold': 8}
- mean confidence of used relations: 0.813
- predicates used: {'near': 59, 'on': 52, 'holding': 39, 'behind': 31, 'under': 26, 'in': 9, 'wearing': 7, 'riding': 3, 'carrying': 1}

## Reranking behaviour (descriptive, not a quality metric)

**object_reranked** - weights `{'w_obj': 1.0, 'w_hall': 0.0, 'w_rel': 0.0}` (locked on 200 val-split images, file sha256 `8a200b119a864d88`)

| statistic | value |
|---|---|
| candidates per image (K per prefix, pooled) | min 4, median 12.0, max 12, mean 11.52 |
| candidate generation success rate | 100.00% |
| candidates dropped (degenerate) | 0  |
| selected score | min -1.599, median 0.544, max 2.345, mean 0.360 |
| margin over runner-up | median 0.0429, mean 0.1145 |
| selected candidate came from prefix | {'baseline': 130, 'relation': 61, 'objects_only': 59} |
| **top-ranked != BLIP's first candidate** | 157/250 (62.80%) |
| relation evidence changed the selection | n/a (this arm has no relation term) |
| captions stating the predicted relation | 81/250 (32.40%) |
| mean confidence of stated relations | 0.847 |

**relation_reranked** - weights `{'w_obj': 1.0, 'w_hall': 0.0, 'w_rel': 0.5}` (locked on 200 val-split images, file sha256 `8a200b119a864d88`)

| statistic | value |
|---|---|
| candidates per image (K per prefix, pooled) | min 4, median 12.0, max 12, mean 11.52 |
| candidate generation success rate | 100.00% |
| candidates dropped (degenerate) | 0  |
| selected score | min -1.599, median 0.629, max 2.345, mean 0.533 |
| margin over runner-up | median 0.0431, mean 0.1266 |
| selected candidate came from prefix | {'relation': 106, 'baseline': 120, 'objects_only': 24} |
| **top-ranked != BLIP's first candidate** | 167/250 (66.80%) |
| **relation evidence changed the selection** | 49/250 (19.60%) |
| captions stating the predicted relation | 130/250 (52.00%) |
| mean confidence of stated relations | 0.822 |

## Hallucination metrics - all images

| metric | baseline | grounded | objects_only | object_reranked | relation_reranked |
|---|---|---|---|---|---|
| CHAIR_i (lower better) | 4.11% | 8.30% | 8.35% | 6.78% | 6.36% |
| CHAIR_s (lower better) | 5.20% | 15.60% | 15.60% | 12.40% | 11.60% |
| hallucinated objects / caption | 0.056 | 0.160 | 0.160 | 0.140 | 0.132 |
| mentioned objects / caption | 1.364 | 1.928 | 1.916 | 2.064 | 2.076 |
| object recall | 40.67% | 54.98% | 54.60% | 59.83% | 60.45% |
| caption length (words) | 8.36 | 9.50 | 10.22 | 10.42 | 10.46 |
| POPE-random accuracy | 71.82% | 79.33% | 79.17% | 81.32% | 81.55% |
| POPE-random F1 | 60.85% | 74.04% | 73.85% | 77.15% | 77.50% |
| POPE-random false-positive rate on absent objects | 0.15% | 0.31% | 0.46% | 0.46% | 0.46% |
| POPE-random yes-ratio | 21.98% | 29.63% | 29.63% | 31.78% | 32.01% |
| POPE-popular accuracy | 71.75% | 78.79% | 78.71% | 80.78% | 81.09% |
| POPE-popular F1 | 60.79% | 73.54% | 73.42% | 76.65% | 77.07% |
| POPE-popular false-positive rate on absent objects | 0.31% | 1.38% | 1.38% | 1.53% | 1.38% |
| POPE-popular yes-ratio | 22.05% | 30.17% | 30.09% | 32.31% | 32.47% |
| POPE-adversarial accuracy | 71.67% | 78.56% | 78.56% | 80.70% | 81.01% |
| POPE-adversarial F1 | 60.72% | 73.33% | 73.28% | 76.58% | 76.99% |
| POPE-adversarial false-positive rate on absent objects | 0.46% | 1.84% | 1.68% | 1.68% | 1.53% |
| POPE-adversarial yes-ratio | 22.13% | 30.40% | 30.25% | 32.39% | 32.54% |
| hallucinations introduced by injected prefix | 0 | 32 | 32 | 22 | 22 |
| hallucinations in BLIP continuation | 14 | 8 | 8 | 13 | 11 |

## Hallucination metrics - images where a relation was injected (n=227)

| metric | baseline | grounded | objects_only | object_reranked | relation_reranked |
|---|---|---|---|---|---|
| CHAIR_i (lower better) | 4.15% | 8.59% | 8.65% | 7.02% | 6.57% |
| CHAIR_s (lower better) | 5.29% | 16.74% | 16.74% | 13.22% | 12.33% |
| hallucinated objects / caption | 0.057 | 0.172 | 0.172 | 0.150 | 0.141 |
| mentioned objects / caption | 1.379 | 2.000 | 1.987 | 2.132 | 2.145 |
| object recall | 40.11% | 55.48% | 55.08% | 60.16% | 60.83% |
| caption length (words) | 8.42 | 9.67 | 10.46 | 10.57 | 10.59 |
| POPE-random accuracy | 71.54% | 79.72% | 79.55% | 81.72% | 81.97% |
| POPE-random F1 | 60.30% | 74.66% | 74.45% | 77.77% | 78.14% |
| POPE-random false-positive rate on absent objects | 0.17% | 0.33% | 0.50% | 0.50% | 0.50% |
| POPE-random yes-ratio | 21.70% | 30.05% | 30.05% | 32.22% | 32.47% |
| POPE-popular accuracy | 71.45% | 79.13% | 79.05% | 81.14% | 81.47% |
| POPE-popular F1 | 60.23% | 74.12% | 73.99% | 77.22% | 77.67% |
| POPE-popular false-positive rate on absent objects | 0.33% | 1.50% | 1.50% | 1.67% | 1.50% |
| POPE-popular yes-ratio | 21.79% | 30.63% | 30.55% | 32.80% | 32.97% |
| POPE-adversarial accuracy | 71.37% | 78.88% | 78.88% | 81.05% | 81.39% |
| POPE-adversarial F1 | 60.16% | 73.89% | 73.84% | 77.14% | 77.59% |
| POPE-adversarial false-positive rate on absent objects | 0.50% | 2.00% | 1.84% | 1.84% | 1.67% |
| POPE-adversarial yes-ratio | 21.87% | 30.88% | 30.72% | 32.89% | 33.06% |

## Paired differences (95% bootstrap CI over images)

**grounded_minus_baseline**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | +4.19 pt [+1.49, +6.89] |
| CHAIR_s | +10.40 pt [+6.00, +14.80] |
| halluc./caption | +0.10 obj [+0.06, +0.15] |
| object recall | +14.30 pt [+11.46, +17.21] |
| POPE-adv F1 | +12.61 pt [+9.49, +15.77] |
| POPE-adv FP rate | +1.38 pt [+0.46, +2.42] |
| CLIPScore | -0.04  [-0.05, -0.03] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 4, 'grounded_hallucinates_only': 30, 'mcnemar_exact_p': 6.16488978266716e-06} 

**grounded_minus_baseline**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | +4.44 pt [+1.59, +7.25] |
| CHAIR_s | +11.45 pt [+6.61, +16.30] |
| halluc./caption | +0.11 obj [+0.06, +0.17] |
| object recall | +15.37 pt [+12.27, +18.48] |
| POPE-adv F1 | +13.73 pt [+10.38, +17.16] |
| POPE-adv FP rate | +1.50 pt [+0.50, +2.64] |
| CLIPScore | -0.04  [-0.05, -0.03] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 4, 'grounded_hallucinates_only': 30, 'mcnemar_exact_p': 6.16488978266716e-06} 

**objects_only_minus_baseline**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | +4.25 pt [+1.49, +6.97] |
| CHAIR_s | +10.40 pt [+6.00, +15.20] |
| halluc./caption | +0.10 obj [+0.06, +0.15] |
| object recall | +13.93 pt [+11.11, +16.71] |
| POPE-adv F1 | +12.56 pt [+9.46, +15.68] |
| POPE-adv FP rate | +1.23 pt [+0.31, +2.17] |
| CLIPScore | -0.03  [-0.04, -0.02] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 5, 'objects_only_hallucinates_only': 31, 'mcnemar_exact_p': 1.2913485988974571e-05} 

**objects_only_minus_baseline**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | +4.49 pt [+1.56, +7.39] |
| CHAIR_s | +11.45 pt [+6.61, +16.30] |
| halluc./caption | +0.11 obj [+0.06, +0.17] |
| object recall | +14.97 pt [+11.97, +18.05] |
| POPE-adv F1 | +13.67 pt [+10.43, +17.02] |
| POPE-adv FP rate | +1.34 pt [+0.34, +2.36] |
| CLIPScore | -0.04  [-0.05, -0.03] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 5, 'objects_only_hallucinates_only': 31, 'mcnemar_exact_p': 1.2913485988974571e-05} 

**grounded_minus_objects_only**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | -0.05 pt [-1.02, +0.89] |
| CHAIR_s | +0.00 pt [-2.00, +2.00] |
| halluc./caption | +0.00 obj [-0.02, +0.02] |
| object recall | +0.37 pt [-0.74, +1.52] |
| POPE-adv F1 | +0.05 pt [-0.99, +1.08] |
| POPE-adv FP rate | +0.15 pt [-0.31, +0.75] |
| CLIPScore | -0.00  [-0.01, +0.00] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 3, 'grounded_hallucinates_only': 3, 'mcnemar_exact_p': 1.0} 

**grounded_minus_objects_only**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | -0.06 pt [-1.05, +0.96] |
| CHAIR_s | +0.00 pt [-2.20, +2.20] |
| halluc./caption | +0.00 obj [-0.02, +0.02] |
| object recall | +0.40 pt [-0.78, +1.63] |
| POPE-adv F1 | +0.05 pt [-1.08, +1.17] |
| POPE-adv FP rate | +0.17 pt [-0.34, +0.82] |
| CLIPScore | -0.00  [-0.01, +0.00] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 3, 'grounded_hallucinates_only': 3, 'mcnemar_exact_p': 1.0} 

**object_reranked_minus_baseline**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | +2.68 pt [+0.44, +4.92] |
| CHAIR_s | +7.20 pt [+3.20, +11.20] |
| halluc./caption | +0.08 obj [+0.04, +0.13] |
| object recall | +19.15 pt [+16.41, +21.98] |
| POPE-adv F1 | +15.86 pt [+12.96, +18.86] |
| POPE-adv FP rate | +1.23 pt [+0.30, +2.29] |
| CLIPScore | -0.00  [-0.01, +0.01] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 4, 'object_reranked_hallucinates_only': 22, 'mcnemar_exact_p': 0.0005335211753845215} 

**object_reranked_minus_baseline**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | +2.87 pt [+0.53, +5.21] |
| CHAIR_s | +7.93 pt [+3.52, +12.33] |
| halluc./caption | +0.09 obj [+0.05, +0.14] |
| object recall | +20.05 pt [+17.17, +23.04] |
| POPE-adv F1 | +16.98 pt [+13.89, +20.29] |
| POPE-adv FP rate | +1.34 pt [+0.33, +2.48] |
| CLIPScore | -0.00  [-0.01, +0.01] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 4, 'object_reranked_hallucinates_only': 22, 'mcnemar_exact_p': 0.0005335211753845215} 

**object_reranked_minus_objects_only**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | -1.57 pt [-3.37, +0.13] |
| CHAIR_s | -3.20 pt [-6.40, +0.00] |
| halluc./caption | -0.02 obj [-0.06, +0.02] |
| object recall | +5.22 pt [+3.42, +7.02] |
| POPE-adv F1 | +3.30 pt [+1.64, +4.99] |
| POPE-adv FP rate | +0.00 pt [-0.61, +0.61] |
| CLIPScore | +0.03  [+0.02, +0.04] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 13, 'object_reranked_hallucinates_only': 5, 'mcnemar_exact_p': 0.09625244140625} 

**object_reranked_minus_objects_only**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | -1.62 pt [-3.53, +0.20] |
| CHAIR_s | -3.52 pt [-7.05, +0.00] |
| halluc./caption | -0.02 obj [-0.06, +0.02] |
| object recall | +5.08 pt [+3.17, +6.99] |
| POPE-adv F1 | +3.30 pt [+1.58, +5.10] |
| POPE-adv FP rate | +0.00 pt [-0.67, +0.67] |
| CLIPScore | +0.03  [+0.03, +0.04] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 13, 'object_reranked_hallucinates_only': 5, 'mcnemar_exact_p': 0.09625244140625} 

**object_reranked_minus_grounded**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | -1.52 pt [-3.23, +0.14] |
| CHAIR_s | -3.20 pt [-6.40, +0.00] |
| halluc./caption | -0.02 obj [-0.06, +0.01] |
| object recall | +4.85 pt [+3.01, +6.71] |
| POPE-adv F1 | +3.25 pt [+1.57, +4.95] |
| POPE-adv FP rate | -0.15 pt [-0.75, +0.31] |
| CLIPScore | +0.04  [+0.03, +0.04] |

CHAIR_s discordant pairs: {'grounded_hallucinates_only': 12, 'object_reranked_hallucinates_only': 4, 'mcnemar_exact_p': 0.076812744140625} 

**object_reranked_minus_grounded**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | -1.57 pt [-3.40, +0.20] |
| CHAIR_s | -3.52 pt [-7.05, -0.44] |
| halluc./caption | -0.02 obj [-0.06, +0.02] |
| object recall | +4.68 pt [+2.75, +6.64] |
| POPE-adv F1 | +3.25 pt [+1.47, +5.07] |
| POPE-adv FP rate | -0.17 pt [-0.83, +0.34] |
| CLIPScore | +0.04  [+0.03, +0.05] |

CHAIR_s discordant pairs: {'grounded_hallucinates_only': 12, 'object_reranked_hallucinates_only': 4, 'mcnemar_exact_p': 0.076812744140625} 

**relation_reranked_minus_baseline**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | +2.25 pt [-0.03, +4.50] |
| CHAIR_s | +6.40 pt [+2.40, +10.40] |
| halluc./caption | +0.08 obj [+0.03, +0.12] |
| object recall | +19.78 pt [+17.07, +22.59] |
| POPE-adv F1 | +16.27 pt [+13.39, +19.32] |
| POPE-adv FP rate | +1.07 pt [+0.15, +2.12] |
| CLIPScore | -0.00  [-0.01, +0.01] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 5, 'relation_reranked_hallucinates_only': 21, 'mcnemar_exact_p': 0.002493917942047119} 

**relation_reranked_minus_baseline**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | +2.42 pt [+0.03, +4.79] |
| CHAIR_s | +7.05 pt [+2.64, +11.45] |
| halluc./caption | +0.08 obj [+0.04, +0.13] |
| object recall | +20.72 pt [+17.90, +23.72] |
| POPE-adv F1 | +17.43 pt [+14.30, +20.75] |
| POPE-adv FP rate | +1.17 pt [+0.17, +2.22] |
| CLIPScore | -0.00  [-0.01, +0.01] |

CHAIR_s discordant pairs: {'baseline_hallucinates_only': 5, 'relation_reranked_hallucinates_only': 21, 'mcnemar_exact_p': 0.002493917942047119} 

**relation_reranked_minus_objects_only**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | -1.99 pt [-3.70, -0.39] |
| CHAIR_s | -4.00 pt [-7.20, -1.20] |
| halluc./caption | -0.03 obj [-0.06, +0.00] |
| object recall | +5.85 pt [+4.06, +7.67] |
| POPE-adv F1 | +3.71 pt [+2.17, +5.29] |
| POPE-adv FP rate | -0.15 pt [-0.75, +0.31] |
| CLIPScore | +0.03  [+0.02, +0.04] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 13, 'relation_reranked_hallucinates_only': 3, 'mcnemar_exact_p': 0.021270751953125} 

**relation_reranked_minus_objects_only**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | -2.08 pt [-3.95, -0.37] |
| CHAIR_s | -4.41 pt [-7.93, -1.32] |
| halluc./caption | -0.03 obj [-0.07, +0.00] |
| object recall | +5.75 pt [+3.87, +7.65] |
| POPE-adv F1 | +3.75 pt [+2.12, +5.42] |
| POPE-adv FP rate | -0.17 pt [-0.82, +0.34] |
| CLIPScore | +0.03  [+0.03, +0.04] |

CHAIR_s discordant pairs: {'objects_only_hallucinates_only': 13, 'relation_reranked_hallucinates_only': 3, 'mcnemar_exact_p': 0.021270751953125} 

**relation_reranked_minus_grounded**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | -1.94 pt [-3.60, -0.38] |
| CHAIR_s | -4.00 pt [-6.80, -1.20] |
| halluc./caption | -0.03 obj [-0.06, +0.00] |
| object recall | +5.47 pt [+3.74, +7.23] |
| POPE-adv F1 | +3.66 pt [+2.14, +5.23] |
| POPE-adv FP rate | -0.31 pt [-0.77, +0.00] |
| CLIPScore | +0.04  [+0.03, +0.04] |

CHAIR_s discordant pairs: {'grounded_hallucinates_only': 12, 'relation_reranked_hallucinates_only': 2, 'mcnemar_exact_p': 0.012939453125} 

**relation_reranked_minus_grounded**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | -2.02 pt [-3.71, -0.36] |
| CHAIR_s | -4.41 pt [-7.49, -1.32] |
| halluc./caption | -0.03 obj [-0.07, +0.00] |
| object recall | +5.35 pt [+3.50, +7.21] |
| POPE-adv F1 | +3.70 pt [+2.08, +5.38] |
| POPE-adv FP rate | -0.33 pt [-0.84, +0.00] |
| CLIPScore | +0.04  [+0.03, +0.05] |

CHAIR_s discordant pairs: {'grounded_hallucinates_only': 12, 'relation_reranked_hallucinates_only': 2, 'mcnemar_exact_p': 0.012939453125} 

**relation_reranked_minus_object_reranked**, all_images (n=250)

| metric | difference |
|---|---|
| CHAIR_i | -0.42 pt [-1.01, -0.03] |
| CHAIR_s | -0.80 pt [-2.00, +0.00] |
| halluc./caption | -0.01 obj [-0.02, +0.00] |
| object recall | +0.62 pt [+0.13, +1.23] |
| POPE-adv F1 | +0.41 pt [-0.05, +0.97] |
| POPE-adv FP rate | -0.15 pt [-0.47, +0.00] |
| CLIPScore | +0.00  [-0.00, +0.00] |

CHAIR_s discordant pairs: {'object_reranked_hallucinates_only': 2, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 0.5} 

**relation_reranked_minus_object_reranked**, relation_used_images (n=227)

| metric | difference |
|---|---|
| CHAIR_i | -0.45 pt [-1.07, -0.03] |
| CHAIR_s | -0.88 pt [-2.20, +0.00] |
| halluc./caption | -0.01 obj [-0.02, +0.00] |
| object recall | +0.67 pt [+0.13, +1.32] |
| POPE-adv F1 | +0.45 pt [-0.05, +1.05] |
| POPE-adv FP rate | -0.17 pt [-0.51, +0.00] |
| CLIPScore | -0.00  [-0.00, +0.00] |

CHAIR_s discordant pairs: {'object_reranked_hallucinates_only': 2, 'relation_reranked_hallucinates_only': 0, 'mcnemar_exact_p': 0.5} 

## Caption quality (reference-free)

CLIPScore = 2.5 * max(cos(image, 'A photo depicts ' + caption), 0), CLIP ViT-B/32. Image-text alignment, NOT a hallucination metric and not reference-based.

| arm | CLIPScore |
|---|---|
| baseline | 0.7115 |
| grounded | 0.6733 |
| objects_only | 0.6776 |
| object_reranked | 0.7090 |
| relation_reranked | 0.7091 |

## EXPLORATORY threshold sweep (not the primary result)

| threshold | usage | CHAIR_i | CHAIR_s | POPE-adv F1 |
|---|---|---|---|---|
| 0.0 | 94.00% | 8.23% | 15.60% | 73.69% |
| 0.3 | 94.00% | 8.23% | 15.60% | 73.69% |
| 0.5 | 90.80% | 8.30% | 15.60% | 73.33% |
| 0.7 | 72.80% | 8.39% | 14.80% | 70.24% |
| 0.9 | 27.60% | 5.31% | 7.60% | 65.02% |

## Notes

- CHAIR/POPE ground truth: human Visual Genome annotations mapped to COCO-80; not exhaustive, so absolute hallucination rates are upper bounds for every arm.
- Fallback images are scored with the baseline arm's caption (same prefix).
- The threshold sweep reuses the grounded arm's relation captions and is exploratory; the primary result is the pre-declared threshold.
- CLIPScore is reference-free image-text alignment (Hessel et al., 2021), not a reference-based caption quality metric; no reference captions exist for these images, so BLEU/CIDEr/SPICE are not reported.

No conclusion is written automatically. Read the paired differences against the pre-registered rules in CAPTION_EXPERIMENTS.md.
