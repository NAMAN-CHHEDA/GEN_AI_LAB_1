# Task 2: Yelp Polarity Sentiment Classification

**Role:** Applied NLP Researcher, Text Classification Team  
**Author:** Sarvesh  
**Hardware (reported runs):** NVIDIA GeForce RTX 4090  
**Constraint:** No pretrained embeddings and no pretrained language models — all embeddings were learned from scratch with `nn.Embedding`.

**Submitted notebooks**
- `01_baseline_meanpool.ipynb`
- `02_textcnn_experimental1.ipynb`
- `03_bilstm_experimental2.ipynb`

**Supporting artifacts:** `outputs/` (metrics, predictions, bootstrap CIs, slice results, McNemar, error-review candidates)

---

## 2.1 Data Preprocessing (10 Marks)

### Dataset

We use the **Yelp Polarity** dataset (`fancyzhx/yelp_polarity`) for binary sentiment classification:
- Label `0` = negative
- Label `1` = positive

| Split | Examples | Negatives | Positives |
|---|---:|---:|---:|
| Official train (raw) | 560,000 | 280,000 | 280,000 |
| Training (90% stratified) | 504,000 | 252,000 | 252,000 |
| Validation (10% stratified) | 56,000 | 28,000 | 28,000 |
| Official test | 38,000 | 19,000 | 19,000 |

### Class distribution and balance

Train, validation, and test splits are **exactly balanced** (50/50). Because class priors are equal, accuracy and macro-F1 are directly comparable and no class-weighting or resampling was required.

### Review length distribution

Raw training reviews are long-tailed in word count (many short reviews, a smaller tail of very long reviews). After preprocessing/tokenization, the training length distribution has approximately:
- median near the short/medium range
- **90th percentile ≈ 144 tokens**
- selected sequence length **MAX_LEN = 160** to cover ~90% of reviews with limited truncation

This choice reduces padding cost while acknowledging that the long-review slice remains harder.

### Missing values and malformed entries

We audited missing `text`/`label` fields and empty strings after whitespace stripping. The Yelp Polarity release used here was clean; any empty/missing rows would be dropped before vocabulary construction and training. No additional malformed-entry repair was required.

### Text preprocessing pipeline

Each review was normalized with:

1. Lowercasing
2. Negation normalization (`won't` → `will not`, `can't` → `can not`, `n't` → ` not`)
3. Removal of escaped/newline characters
4. URL and HTML tag removal
5. Removal of punctuation, digits, and special characters (keep alphabetic tokens)
6. Stopword removal, while **preserving** sentiment-critical negations: `no`, `not`, `nor`, `never`
7. Lemmatization with WordNet
8. Whitespace tokenization

**Worked example**
- Input: `The service wasn't good.`
- Output tokens: `['service', 'not', 'good']`

### Vocabulary, tokenization IDs, and from-scratch embeddings

- Vocabulary is built from the **training split only** to prevent leakage
- Vocabulary size = **30,000** including `<PAD>` (index 0) and `<UNK>` (index 1)
- Reviews are mapped to integer IDs, truncated/padded to **MAX_LEN = 160**
- Embeddings are a randomly initialized `nn.Embedding(30000, 128)` trained end-to-end (no Word2Vec/GloVe/FastText and no pretrained LMs)

---

## 2.2 Model Training and Evaluation (15 Marks)

### Shared experimental setup

All three models use the same preprocessing, vocabulary, `MAX_LEN`, train/val/test splits, batch size, optimizer, and epoch budget:

| Setting | Value |
|---|---|
| Embedding dim | 128 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | 5 |
| Loss | `BCEWithLogitsLoss` (single logit) |
| Decision threshold | 0.5 on sigmoid probability |
| Checkpoint rule | best validation macro-F1 |

### Model lineup and justification

| Model | Architecture | What changed vs previous | Why this choice |
|---|---|---|---|
| **Baseline** | Mean-pooled embeddings → Linear | Reference model | Learns token polarity but ignores order; fair weak neural baseline |
| **Experimental 1: TextCNN** | Conv1d kernels `{3,4,5}`, 128 filters, max-over-time pool | Local n-gram structure | Captures phrases like “not good” that mean-pooling dilutes |
| **Experimental 2: BiLSTM** | 1-layer bidirectional LSTM, hidden 128 | Sequential left/right context | Handles delayed negation and contrast better than fixed windows |

---

### Model 1 — Baseline (Mean-Pooled Embedding Classifier)

**Design**
Masked mean pooling over non-padding token embeddings, dropout 0.3, linear classifier to one logit.

**Systems metrics**
- Parameters: 3,840,129
- Training time: 111.91 s
- Average examples/sec: 22,921
- Peak GPU memory: 97.96 MB
- GPU: NVIDIA GeForce RTX 4090

**Test metrics**

| Metric | Value |
|---|---:|
| Accuracy | 0.9342 |
| Precision macro / micro / weighted | 0.9342 / 0.9342 / 0.9342 |
| Recall macro / micro / weighted | 0.9342 / 0.9342 / 0.9342 |
| F1 macro / micro / weighted | 0.9342 / 0.9342 / 0.9342 |
| ROC-AUC | 0.9805 |
| PR-AUC | 0.9800 |
| MCC | 0.8684 |
| Brier score | 0.0499 |
| ECE | 0.0067 |
| Confusion matrix | `[[17738, 1262], [1239, 17761]]` |
| 95% CI accuracy | [0.9318, 0.9366] |
| 95% CI macro-F1 | [0.9318, 0.9366] |
| 95% CI MCC | [0.8635, 0.8733] |

---

### Model 2 — Experimental 1 (TextCNN)

**Design**
Embedding → three Conv1d branches (kernel sizes 3/4/5, 128 filters each) → ReLU → max-over-time pooling → concatenate → dropout 0.5 → linear logit.

**Systems metrics**
- Parameters: 4,037,377
- Training time: 109.09 s
- Average examples/sec: 23,208
- Peak GPU memory: 143.73 MB
- GPU: NVIDIA GeForce RTX 4090

**Test metrics**

| Metric | Value |
|---|---:|
| Accuracy | 0.9395 |
| Precision macro / micro / weighted | 0.9399 / 0.9395 / 0.9399 |
| Recall macro / micro / weighted | 0.9395 / 0.9395 / 0.9395 |
| F1 macro / micro / weighted | 0.9395 / 0.9395 / 0.9395 |
| ROC-AUC | 0.9850 |
| PR-AUC | 0.9852 |
| MCC | 0.8794 |
| Brier score | 0.0456 |
| ECE | 0.0178 |
| Confusion matrix | `[[17577, 1423], [875, 18125]]` |
| 95% CI accuracy | [0.9371, 0.9418] |
| 95% CI macro-F1 | [0.9371, 0.9418] |
| 95% CI MCC | [0.8746, 0.8840] |

---

### Model 3 — Experimental 2 (BiLSTM)

**Design**
Embedding → packed bidirectional LSTM (1 layer, hidden 128) → concat final forward/backward states → dropout 0.5 → linear logit.

**Systems metrics**
- Parameters: 4,104,449
- Training time: 583.63 s
- Average examples/sec: 4,324
- Peak GPU memory: 224.54 MB
- GPU: NVIDIA GeForce RTX 4090

**Test metrics**

| Metric | Value |
|---|---:|
| Accuracy | 0.9486 |
| Precision macro / micro / weighted | 0.9487 / 0.9486 / 0.9487 |
| Recall macro / micro / weighted | 0.9486 / 0.9486 / 0.9486 |
| F1 macro / micro / weighted | 0.9486 / 0.9486 / 0.9486 |
| ROC-AUC | 0.9884 |
| PR-AUC | 0.9885 |
| MCC | 0.8974 |
| Brier score | 0.0401 |
| ECE | 0.0211 |
| Confusion matrix | `[[18157, 843], [1109, 17891]]` |
| 95% CI accuracy | [0.9464, 0.9508] |
| 95% CI macro-F1 | [0.9464, 0.9508] |
| 95% CI MCC | [0.8929, 0.9017] |

---

### Side-by-side comparison (all required metrics)

| Metric | Baseline | TextCNN | BiLSTM |
|---|---:|---:|---:|
| Accuracy | 0.9342 | 0.9395 | **0.9486** |
| Precision (macro) | 0.9342 | 0.9399 | **0.9487** |
| Precision (micro) | 0.9342 | 0.9395 | **0.9486** |
| Precision (weighted) | 0.9342 | 0.9399 | **0.9487** |
| Recall (macro) | 0.9342 | 0.9395 | **0.9486** |
| Recall (micro) | 0.9342 | 0.9395 | **0.9486** |
| Recall (weighted) | 0.9342 | 0.9395 | **0.9486** |
| F1 (macro) | 0.9342 | 0.9395 | **0.9486** |
| F1 (micro) | 0.9342 | 0.9395 | **0.9486** |
| F1 (weighted) | 0.9342 | 0.9395 | **0.9486** |
| ROC-AUC | 0.9805 | 0.9850 | **0.9884** |
| PR-AUC | 0.9800 | 0.9852 | **0.9885** |
| MCC | 0.8684 | 0.8794 | **0.8974** |
| Brier (↓ better) | 0.0499 | 0.0456 | **0.0401** |
| ECE (↓ better) | **0.0067** | 0.0178 | 0.0211 |
| Parameter count | 3,840,129 | 4,037,377 | 4,104,449 |
| Training time (s) | 111.91 | **109.09** | 583.63 |
| Examples/sec | 22,921 | **23,208** | 4,324 |
| Peak GPU memory (MB) | **97.96** | 143.73 | 224.54 |
| GPU | RTX 4090 | RTX 4090 | RTX 4090 |

### Paired McNemar tests (baseline vs each experimental)

| Comparison | n01 (exp right, base wrong) | n10 (base right, exp wrong) | Statistic | p-value | Decision (α=0.05) |
|---|---:|---:|---:|---:|---|
| Baseline vs TextCNN | 1057 | 854 | 21.35 | 3.82×10⁻⁶ | TextCNN significantly better |
| Baseline vs BiLSTM | 1212 | 663 | 160.16 | 1.04×10⁻³⁶ | BiLSTM significantly better |

BiLSTM’s discordant win margin (1212 vs 663) is larger than TextCNN’s (1057 vs 854), matching the larger accuracy gap.

### Robustness by data slice (macro-F1 / error rate)

| Slice | n | Baseline | TextCNN | BiLSTM |
|---|---:|---|---|---|
| Short (≤50 tokens) | 19,274 | 0.9361 / 0.0628 | 0.9429 / 0.0561 | **0.9524 / 0.0469** |
| Medium (51–144) | 15,046 | 0.9339 / 0.0655 | 0.9399 / 0.0598 | **0.9493 / 0.0502** |
| Long (>144) | 3,680 | 0.9099 / 0.0826 | 0.9076 / 0.0861 | **0.9123 / 0.0796** |
| Contains negation | 28,293 | 0.9269 / 0.0707 | 0.9348 / 0.0633 | **0.9451 / 0.0529** |
| No explicit negation | 9,707 | 0.9276 / 0.0516 | 0.9260 / 0.0521 | **0.9347 / 0.0470** |

All models degrade on long reviews. BiLSTM is best on every slice; long reviews remain the weakest regime.

---

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

## 2.3 Comparative Analysis and Observations (5 Marks)

### Comparison across my three models

Ranking by test accuracy / macro-F1 / MCC:

1. **BiLSTM — best overall** (Acc 0.9486, F1 0.9486, MCC 0.8974)
2. **TextCNN** (Acc 0.9395, F1 0.9395, MCC 0.8794)
3. **Baseline MeanPool** (Acc 0.9342, F1 0.9342, MCC 0.8684)

Evidence quality:
- McNemar p-values show both experimental models beat baseline far below α=0.05
- BiLSTM accuracy CI `[0.9464, 0.9508]` does not overlap baseline CI `[0.9318, 0.9366]`

Efficiency trade-off:
- TextCNN is the best speed/accuracy compromise (~109 s, +0.53 accuracy points over baseline)
- BiLSTM is slower (~584 s) and uses more memory, but wins discrimination (ROC/PR-AUC) and Brier score
- Baseline remains best calibrated (ECE 0.0067)

### Strengths

- Competitive from-scratch performance without pretrained LMs
- Controlled bake-off: identical data pipeline across models
- BiLSTM improves every robustness slice, especially short/medium reviews and negation

### Weaknesses and limitations

- Long reviews are the shared failure mode (truncation + mixed sentiment)
- Assignment forbids pretrained embeddings/LMs, capping semantic transfer
- BiLSTM can overfit as train loss keeps falling while validation loss rises after mid-training
- Confident errors often involve contrast, aspect conflict, or possible label noise
- As models become more accurate/confident, ECE worsens relative to baseline

### Future work

1. Raise `MAX_LEN` or use hierarchical chunking for long documents
2. Early stopping + LR schedule + stronger regularization for BiLSTM
3. Probability ensemble of TextCNN + BiLSTM
4. Temperature scaling / threshold tuning for calibration and near-boundary cases
5. Contrast- and aspect-aware features for brand-vs-location failures

---

## Conclusion

Under a strict no-pretrained-embedding setting, a mean-pooled embedding baseline already reaches **93.42%** test accuracy. TextCNN improves this to **93.95%** by modeling local n-grams, and BiLSTM further reaches **94.86%** by modeling bidirectional context. Both experimental gains are statistically significant by paired McNemar tests against baseline, with BiLSTM providing the larger improvement. Remaining errors concentrate in long, mixed, and contrastive reviews, motivating longer-context and aspect/contrast modeling as the next research steps.

---

## Appendix A — File map

| Content | Path |
|---|---|
| Baseline notebook | `01_baseline_meanpool.ipynb` |
| TextCNN notebook | `02_textcnn_experimental1.ipynb` |
| BiLSTM notebook | `03_bilstm_experimental2.ipynb` |
| Baseline metrics | `outputs/baseline_metrics.json` |
| TextCNN metrics | `outputs/textcnn_experimental1_metrics.json` |
| BiLSTM metrics | `outputs/bilstm_experimental2_metrics.json` |
| McNemar results | `outputs/mcnemar_results.json` |
| Slice robustness | `outputs/*_slice_results.csv` |
| Error-review candidates | `outputs/bilstm_error_review_candidates.csv` |
| Predictions | `outputs/*_predictions.csv` |

## Appendix B — One-line metric interpretation cheat sheet

- **Accuracy / F1 / MCC:** overall correctness; MCC is balanced and robust for binary tasks
- **ROC-AUC / PR-AUC:** ranking quality of predicted probabilities
- **Brier / ECE:** probability calibration (lower is better)
- **Bootstrap CI:** uncertainty of point estimates
- **McNemar:** whether paired prediction differences vs baseline are significant
- **Slice metrics:** where the model is fragile (here: long reviews)
