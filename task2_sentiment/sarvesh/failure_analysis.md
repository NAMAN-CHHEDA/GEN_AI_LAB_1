# Task 2 — Failure / Error Analysis (Sarvesh)

Extracted from `results.md` for the lab-required `failure_analysis.md` file.
Source candidates: `outputs/bilstm_error_review_candidates.csv`.

### Manual error analysis of 20 BiLSTM failures

Errors were sampled from the held-out test set into four required buckets (5 each): confident false positives (`p≥0.90`, true=0), confident false negatives (`p≤0.10`, true=1), near-threshold errors (`|p−0.5|≤0.05`), and long-review slice failures (`token_len>144`). Source file: `outputs/bilstm_error_review_candidates.csv`.

#### Representative examples

**Confident FP (true=0, pred=1, p≈0.9999)**  
Review text is strongly positive (“Wow love the place… Great place…”), yet gold label is negative.  
**Error type:** likely label noise / mismatched gold label.  
**Testable fix:** manual label audit on ultra-confident disagreements; optionally down-weight contested labels.

**Confident FP (true=0, pred=1, p≈0.9997)**  
“I love cafe rio! … However I've been to this location 3 different times… DO NOT EAT AT THIS LOCATION!”  
**Error type:** brand-level praise overridden by location-specific complaint (contrast not resolved).  
**Testable fix:** aspect/location conditioning; explicit “this location” contrast features.

**Confident FN (true=1, pred=0, p≈3e-5)**  
Long negative history about old owners, then “Now its much better… nothing but positive things.”  
**Error type:** recency/contrast failure; early negative tokens dominate.  
**Testable fix:** hierarchical segment pooling with recency weighting.

**Near-threshold (true=0, pred=1, p≈0.50005)**  
“Still not the most wonderful experience, but hey - it's a chain store.”  
**Error type:** lukewarm/ambiguous sentiment near decision boundary.  
**Testable fix:** validation-tuned threshold or reject option for `0.45<p<0.55`.

**Long-slice FN (true=1, pred=0, len=197, p≈0.024)**  
Long hotel narrative with late overall positive conclusion after service complaints.  
**Error type:** truncation/dilution of late polarity under `MAX_LEN=160`.  
**Testable fix:** increase `MAX_LEN` or chunk-then-attend over segments.

#### Full 20-case coding table

| # | Bucket | Error type | Testable fix |
|---|---|---|---|
| 1 | Confident FP | Label noise / gold-label mismatch | Audit ultra-confident disagreements; soft-label contested cases |
| 2 | Confident FP | Contrastive “but…” reversal ignored | Discourse-connector features; clause-level polarity aggregation |
| 3 | Confident FP | Comparison to preferred alternative | Comparative/entity-aspect features |
| 4 | Confident FP | Praise mixed with price/value objection | Multi-aspect heads (quality vs value) |
| 5 | Confident FP | Brand love vs this-location complaint | Location-aspect conditioning |
| 6 | Confident FN | Mild positive buried in negative comparisons | Down-weight contrastive negatives; longer context |
| 7 | Confident FN | Positive update after negative history | Recency-weighted hierarchical pooling |
| 8 | Confident FN | Overall positive, service complaints dominate | Aspect aggregation before final decision |
| 9 | Confident FN | Late positive conclusion diluted/truncated | Larger MAX_LEN; chunk + attend |
| 10 | Confident FN | Hype/disappointment with residual positivity | Contrast-aware aggregation |
| 11 | Near-threshold | Lukewarm/ambiguous sentiment | Threshold tuning; abstain band |
| 12 | Near-threshold | Irony/sarcasm | Irony cues; ensemble with TextCNN |
| 13 | Near-threshold | Mixed social + venue signals | Temperature scaling for calibration |
| 14 | Near-threshold | Functional positive, quality mediocre | Aspect labels (amenity vs quality) |
| 15 | Near-threshold | Negative content with joking tone | Better calibration; probability ensemble |
| 16 | Slice (long) | Truncation of late sentiment | MAX_LEN↑ or hierarchical BiLSTM |
| 17 | Slice (long) | Long narrative, polarity late | Attention over LSTM states |
| 18 | Slice (long) | Rebuttal of bad reviews flips polarity | Sentence-level voting |
| 19 | Slice (long) | Recovery story (bad → good) | Segment sentiment then aggregate |
| 20 | Slice (long) | Logistics-heavy weak-lexicon review | Longer context; domain cue features |

---
