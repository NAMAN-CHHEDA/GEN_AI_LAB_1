# DATA266 Laboratory 1: Collaborative Report on Language Modeling, Sentiment Classification, and Unpaired Image-to-Image Translation

**San José State University — Fall 2026**


|                 |                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------- |
| **Course**      | DATA266                                                                                      |
| **Assignment**  | Laboratory 1 (Team Report)                                                                   |
| **Team number** | Lab Pair 10                                                                                  |
| **Repository**  | [https://github.com/NAMAN-CHHEDA/GEN_AI_LAB_1](https://github.com/NAMAN-CHHEDA/GEN_AI_LAB_1) |


**Authors**


| Name               | SJSU ID   |
| ------------------ | --------- |
| Sarvesh Reshimwale | 019129357 |
| Naman Chheda       | 019158893 |


---

## Abstract

This report presents the joint outcome of Laboratory 1, in which each team member independently implemented and evaluated models for three tasks: (i) a character-level GPT-style language model trained on TinyStories without prebuilt Transformer modules; (ii) binary sentiment classification on Yelp Polarity without pretrained embeddings or pretrained language models; and (iii) unpaired Monet↔Photo translation with CycleGAN, including class competition submission and human preference auditing. Architectures and hyperparameters were chosen independently so that members’ systems are not near-duplicates. We report the required quantitative metrics for each task, summarize failure or error analyses, and provide a cross-member discussion of strengths, limitations, and future work. Supporting artifacts (code, checkpoints, logs, predictions, and metric tables) are organized in the shared repository under named member folders.

---



## 1. Introduction

Modern applied-AI workflows separate individual experimentation from team-level synthesis. Laboratory 1 adopts that pattern: each member designs, trains, and documents personal models, after which the team produces a single comparative report. The technical objectives span three modalities.

**Task 1** revisits the TinyStories setting of Eldan and Li (2023) as a controlled testbed for causal self-attention (Vaswani et al., 2017). Members implement multi-head attention, positional embeddings, and autoregressive decoding from scratch.

**Task 2** studies binary sentiment under a low-resource constraint—no pretrained word vectors or language models—using the Yelp Polarity domain. Each member trains a baseline and two experimental classifiers and reviews error modes manually.

**Task 3** applies CycleGAN (Zhu et al., 2017) to unpaired style transfer between Monet paintings and photographs, with evaluation via distributional metrics, cycle reconstruction, perceptual similarity, human ratings, and the class Kaggle leaderboard.

The remainder of this document states team ownership (Section 2), describes each task with comparative tables and analysis (Sections 3–5), lists evidence paths (Section 6), and closes with references (Section 7).

---



## 2. Team Ownership and Reproducibility

Each member is solely responsible for the architecture, hyperparameters, training runs, metrics tables, and failure or error analyses in their named directories:

- `task1_llm/{sarvesh,naman}/`
- `task2_sentiment/{sarvesh,member_naman}/`
- `task3_gan/{sarvesh,naman}/`

**Sarvesh Reshimwale (019129357)** contributed a character-level GPT on TinyStories; MeanPool, TextCNN, and BiLSTM sentiment models; and a CycleGAN Monet↔Photo system whose competition submission reports local FID 94.494, MiFID 0.4108, Kaggle score 47.4524, and leaderboard rank 11 (recorded 6 October 2026).

**Naman Chheda (019158893)** contributed a smaller character-level GPT with layer and dropout ablations; n-gram bag, Transformer, and BiGRU sentiment models with calibration and slice analyses; and an independent CycleGAN implementation under `task3_gan/naman/` (a 60,000-step baseline, run01, and a 150,000-step final model, run02) whose competition submission reports local FID 99.501, MiFID 0.4121, and Kaggle score 49.9565, with member-specific results and artifacts documented in that folder.

Reproduction entry points are documented in the repository root `README.md` (smoke scripts and notebook sequences per member and task). Raw training logs and environment manifests are retained under `reproducibility/` and the corresponding member directories. No credentials or API keys are included in the repository.

---



## 3. Task 1 — Character-Level Language Modeling on TinyStories



### 3.1 Methods

Both members implemented causal Transformer blocks comprising multi-head self-attention with causal masking, layer normalization, position-wise feed-forward networks, and residual connections, together with learned token and positional embeddings and a linear language-modeling head. Training minimized next-character cross-entropy with learning-rate warm-up and cosine decay. Prebuilt Transformer or attention library modules were not used.

Table 1 summarizes architectural and optimization choices. The two systems differ in width, depth, activation, regularization, context length, batch size, and data-split seed.

**Table 1.** Task 1 architecture and hyperparameters.


| Item                | Sarvesh Reshimwale                                                    | Naman Chheda                                             |
| ------------------- | --------------------------------------------------------------------- | -------------------------------------------------------- |
| Normalization / FFN | Pre-LN; ReLU FFN                                                      | Pre-LN; GELU FFN                                         |
| Model width d       | 192                                                                   | 128                                                      |
| Layers              | 5                                                                     | 4 (ablations: 2, 4, 6)                                   |
| Attention heads     | 6                                                                     | 4                                                        |
| FFN width           | 768                                                                   | 512                                                      |
| Dropout             | 0.1                                                                   | 0.2 (ablation also 0.0)                                  |
| Context length      | 160                                                                   | 128                                                      |
| Batch size          | 48                                                                    | 64                                                       |
| Epochs              | 12                                                                    | 15                                                       |
| Optimizer           | AdamW (\beta_1,\beta_2)=(0.9,0.999), weight decay 0.05, grad clip 1.0 | AdamW (0.9,0.95), weight decay 0.1, grad clip 1.0        |
| Learning rate       | Peak 8\times10^{-4}, warmup 800 steps, cosine to 8\times10^{-5}       | Peak 1\times10^{-3}, 5% warmup, cosine to 0.1\times peak |
| Split / seed        | Story-level shuffle, seed 3141                                        | Contiguous sequence split, seed 8893                     |
| Vocabulary size     | 85                                                                    | 83                                                       |
| Parameters          | 2,267,904                                                             | 830,976                                                  |
| Hardware            | NVIDIA GeForce RTX 4090; mixed precision (fp16)                       | NVIDIA GeForce RTX 4090; mixed precision (fp16)          |




### 3.2 Quantitative Results

**Table 2.** Task 1 evaluation metrics.


| Metric                              | Sarvesh         | Naman           |
| ----------------------------------- | --------------- | --------------- |
| Training cross-entropy (eval)       | 0.7040          | 0.8100          |
| Validation cross-entropy            | 0.7392          | 0.8301          |
| Perplexity                          | 2.094           | 2.294           |
| Bits per character                  | 1.066           | 1.198           |
| Generalization gap (val − train)    | 0.0352          | 0.0201          |
| Top-1 next-character accuracy (val) | 0.767           | 0.739           |
| Repeated 4-gram rate (greedy)       | 0.277           | 0.599           |
| Non-finite optimization steps       | 8               | 6               |
| Parameter count                     | 2,267,904       | 830,976         |
| Training throughput (tokens/s)      | \approx 573,000 | \approx 650,000 |
| Peak GPU memory                     | \approx 599 MB  | \approx 324 MB  |
| Wall-clock training time            | \approx 370 s   | \approx 303 s   |


Generation diversity (Distinct-1/2/3) and additional generation statistics are reported in each member’s `metrics_report.csv` and sample files. Loss curves are provided under each member’s `outputs/` directory.

Naman’s ablation study indicates that increasing depth from two to four layers substantially reduces validation loss (approximately 0.966 to 0.830), with a further improvement at six layers (approximately 0.779). Removing dropout lowers validation loss but widens the train–validation gap, consistent with a bias–variance trade-off.

### 3.3 Failure Analysis

Qualitative inspection of generated text reveals recurring failure modes for both systems.

**Sarvesh.** (1) Under greedy decoding, generations exhibit phrase-level loops (elevated repeated 4-gram rate), which diminish under temperature sampling. (2) Temperature sampling introduces orthographic inventions and locally ungrammatical spans. (3) Discourse-level contradictions appear (e.g., affective inconsistency within a short narrative). Detailed snippets are recorded in `task1_llm/sarvesh/failure_analysis.md`.

**Naman.** (1) Strong lexical repetition under greedy decoding (repeated 4-gram rate \approx 0.599). (2) Fragmented or non-word tokens at higher temperature. (3) Logical contradictions analogous to those observed by Sarvesh. Details appear in `task1_llm/naman/failure_analysis.md`.

### 3.4 Discussion

The larger model attains lower validation cross-entropy and perplexity, whereas the smaller model is more memory-efficient, faster per run, and exhibits a tighter generalization gap. Both confirm that causal masking and from-scratch attention are sufficient for coherent short children’s-story fragments, yet decoding strategy dominates surface quality: greedy search favors repetition, while stochastic decoding trades repetition for orthographic noise. Character-level tokenization remains a structural limitation for long-range semantics relative to subword models, which were outside the scope of this assignment. Future work includes nucleus sampling with repetition penalties, modestly longer contexts, and a shared evaluation harness for Distinct-n and repeated-ngram statistics.

---



## 4. Task 2 — Binary Sentiment Classification (Yelp Polarity)



### 4.1 Methods

Each member trained three classifiers with embeddings learned from scratch (no pretrained embeddings or language models): one baseline and two experimental variants. Text was lowercased and cleaned; tokenization and sequence truncation followed member-specific preprocessing. Evaluation covered accuracy, precision/recall/F1 (macro, micro, weighted), confusion matrices, ROC-AUC, PR-AUC, MCC, Brier score, expected calibration error, bootstrap confidence intervals where computed, McNemar tests against the baseline, and slice-wise robustness, in addition to parameter count, throughput, and memory.

**Table 3.** Task 2 model families and training setup.


| Item                             | Sarvesh                                                     | Naman                                                         |
| -------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
| Baseline                         | Mean pooling over learned embeddings                        | N-gram bag classifier                                         |
| Experimental model 1             | TextCNN                                                     | Transformer encoder classifier                                |
| Experimental model 2             | Bidirectional LSTM                                          | Bidirectional GRU                                             |
| Embedding dimension / vocabulary | 128 / 30,000                                                | Learned embeddings / 50,000                                   |
| Maximum length                   | 160 tokens                                                  | 320 tokens (head–tail packing)                                |
| Optimization                     | Adam (1\times10^{-3}), batch 128, 5 epochs, BCE with logits | AdamW with cosine schedule and EMA; early stopping patience 2 |
| Hardware                         | NVIDIA GeForce RTX 4090                                     | NVIDIA GeForce RTX 4090                                       |




### 4.2 Quantitative Results

**Table 4.** Sarvesh — test-set metrics by model.


| Metric                | MeanPool       | TextCNN           | BiLSTM             |
| --------------------- | -------------- | ----------------- | ------------------ |
| Accuracy              | 0.9342         | 0.9395            | 0.9486             |
| Macro-F1              | 0.9342         | 0.9395            | 0.9486             |
| ROC-AUC               | 0.9805         | 0.9850            | 0.9884             |
| MCC                   | 0.8684         | 0.8794            | 0.8974             |
| Brier score           | 0.0499         | 0.0456            | 0.0401             |
| ECE                   | 0.0067         | 0.0178            | 0.0211             |
| Parameters            | 3.84\times10^6 | 4.04\times10^6    | 4.10\times10^6     |
| Training time (s)     | 111.9          | 109.1             | 583.6              |
| Peak memory (MB)      | 98             | 144               | 225                |
| McNemar p vs MeanPool | —              | 3.82\times10^{-6} | 1.04\times10^{-36} |


**Table 5.** Naman — test-set metrics by model.


| Metric                | N-gram bag              | Transformer                 | BiGRU                    |
| --------------------- | ----------------------- | --------------------------- | ------------------------ |
| Accuracy [95% CI]     | 0.9598 [0.9577, 0.9616] | 0.9428 [0.9406, 0.9451]     | 0.9624 [0.9604, 0.9643]  |
| Macro-F1              | 0.9598                  | 0.9428                      | 0.9624                   |
| ROC-AUC               | 0.9907                  | 0.9867                      | 0.9940                   |
| MCC                   | 0.9196                  | 0.8857                      | 0.9248                   |
| Brier / ECE           | 0.0314 / 0.0059         | 0.0431 / 0.0128             | 0.0278 / 0.0030          |
| Parameters            | 6.56\times10^7          | 6.84\times10^6              | 3.90\times10^7           |
| Training time (s)     | 147                     | 121                         | 311                      |
| Peak memory (MB)      | 2018                    | 702                         | 1490                     |
| McNemar vs n-gram bag | —                       | inferior (p\approx10^{-58}) | superior (p\approx0.002) |


Complete metric suites, including confusion matrices, PR-AUC, and slice-wise macro-F1, are stored in each member’s `metrics_report.csv` and associated output files.

### 4.3 Error Analysis

Manual review of twenty errors per primary experimental model indicates that annotation noise and mixed sentiment are major sources of disagreement with gold labels. Sarvesh’s BiLSTM errors include high-confidence false positives on reviews that read as positive despite a negative label, contrastive documents that praise a brand while condemning a location, and false negatives associated with temporal shifts (“now improved”) or truncation beyond 160 tokens; long documents form the weakest length slice. Naman’s BiGRU review attributes errors to label noise, mixed polarity, contrast, sarcasm, negation, and short-text slices. Full case lists with proposed mitigations appear in the respective `failure_analysis.md` files.

### 4.4 Discussion

Under the no-pretrained-embedding constraint, both members obtain strong discrimination (ROC-AUC >0.98). Sequence models (BiLSTM, BiGRU) improve over bag-of-embeddings or n-gram baselines on ranking and correlation metrics, though calibration (ECE) is not uniformly improved. Naman’s Transformer underperforms his n-gram baseline on this setup, illustrating that increased architectural complexity does not guarantee gains when lexical cues dominate. Limitations include label noise in Yelp Polarity, sensitivity to truncation length, and imperfect calibration. Future work includes unified slice definitions across members, longer contexts for truncated reviews, and targeted handling of contrastive and sarcastic constructions.

---



## 5. Task 3 — Unpaired Image-to-Image Translation (CycleGAN)



### 5.1 Methods

Each member implemented a CycleGAN with two generators and two discriminators trained using adversarial losses and cycle-consistency constraints on unpaired Monet and photographic images. Identity regularization and other training details were set independently. Translations were produced in both directions (Monet→Photo and Photo→Monet). Evaluation comprises Fréchet and Kernel Inception Distances, generative precision/recall, cycle \ell_1, LPIPS, content cosine similarity, loss curves, stability diagnostics, human ratings with inter-rater agreement, computational cost, and the class competition score/rank.

**Table 6.** Task 3 configurations (member-level detail).


| Item                        | Sarvesh                                                                                 | Naman                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Generators / discriminators | ResNet generators; PatchGAN discriminators with spectral normalization                  | Generators with 9 residual blocks (ngf = 64) and resize-convolution upsampling; discriminators with ndf = 64 (member implementation under `task3_gan/naman/`) |
| Resolution                  | 256\times256                                                                            | 256\times256 (random crop from 286\times286, horizontal flip)                           |
| Data                        | 300 Monet, 7,038 photos (unpaired)                                                      | Shared Monet/Photo domains under `task3_gan/data/` (300 Monet, 7,038 photos); run02 trained on all images, no held-out set |
| Batch size / epochs         | 12 / 150                                                                                | 1 / 150,000 iterations                                                                  |
| Loss weights                | \lambda_{\mathrm{cycle}}=10, \lambda_{\mathrm{identity}}=0.5                            | \lambda_{\mathrm{cycle}}=10 (steps 0-40,000), 5 (40,000-100,000), 10 (100,000-150,000); \lambda_{\mathrm{identity}}=2.5 (steps 0-100,000), 5 (100,000-150,000) |
| Additional training choices | Discriminator learning rate 0.5\times generator; AMP (bfloat16); test-time augmentation | Adam (lr 2\times10^{-4}, \beta=(0.5,0.999)), constant until step 50,000 then linear decay; DiffAugment (color, translation) on discriminator inputs only; image pool of 50; EMA generator weights (decay 0.999); seed 42; run stopped and resumed at steps 40,000 and 100,000 (manifests in `reproducibility/manifests_run02/`) |
| Parameters                  | 28,275,336                                                                              | Reported in member metrics                                                              |
| Hardware / cost             | RTX 4090; \approx 755.8 min; peak VRAM \approx 12.9 GB                                  | \approx 297.7 min (17,863 s); peak GPU memory \approx 19.6 GB; 0 non-finite steps       |
| Outputs                     | `pred_A2B` (300), `pred_B2A` (7,038)                                                    | `task3_gan/naman/outputs/`: `pred_A2B` (300), `pred_B2A` (300)                          |
| Checkpoints                 | `G_A2B_final.pth`, `G_B2A_final.pth`                                                    | `task3_gan/naman/checkpoints_run02/ema_step125000.pt` (EMA generator weights; submitted model) |




### 5.2 Quantitative Results 

**Table 7.** Class competition submission metrics and leaderboard outcome


| Quantity                     | Sarvesh                                | Naman                                     |
| ---------------------------- | -------------------------------------- | ----------------------------------------- |
| Mean FID (both directions)   | 94.494                                 | 99.501                                    |
| Mean MiFID (both directions) | 0.4108                                 | 0.4121                                    |
| Photo → Monet FID / MiFID    | 95.309 / 0.4056                        | 97.667 / 0.4082                           |
| Monet → Photo FID / MiFID    | 93.679 / 0.4159                        | 101.335 / 0.4161                          |
| Kaggle score                 | 47.4524                                | 49.9565                                   |
| Leaderboard rank             | 11                                     | 8 (team entry, morning of 6 October 2026) |



Source file: `task3_gan/sarvesh/submission.csv`. Training curves and epoch histories are available at `outputs/loss_curves_v3.png` and `logs/training_history_v3.csv`. Naman’s source file is `task3_gan/naman/submission.csv` (run02, step 125,000); his Kaggle score improved across submissions from 54.4377 (run01, step 60,000) to 51.0272 (run02, step 100,000) to 49.9565 (run02, step 125,000).

**Table 8.** Supplementary distributional and perceptual metrics (Sarvesh; see `full_metrics_report.csv`).


| Metric                    | Monet → Photo | Photo → Monet |
| ------------------------- | ------------- | ------------- |
| FID                       | 199.27        | 220.99        |
| KID                       | 0.0301        | 0.0291        |
| Generative precision      | 0.633         | 0.533         |
| Generative recall         | 0.533         | 0.600         |
| Cycle \ell_1              | 0.146         | 0.113         |
| Translation LPIPS         | 0.282         | 0.360         |
| Cycle LPIPS               | 0.370         | 0.270         |
| Content cosine similarity | 0.690         | 0.645         |


Parameter count, training time, throughput, and peak memory associated with these probes are recorded alongside the metric CSV and member `results.md`.

Naman (run02, step 125,000; class grader protocol on the first 300 sorted images per direction): Monet → Photo FID 101.335 / MiFID 0.4161; Photo → Monet FID 97.667 / MiFID 0.4082. Naman’s corresponding metric tables, curves, predictions, and competition records are maintained in `task3_gan/naman/` (`results.md`, metric CSVs, and `outputs/`).

### 5.3 Human Audit

A blinded audit of thirty fixed samples rated style, content preservation, and artifacts on a 1–5 scale with two raters.

**Table 9.** Human audit summary (Sarvesh protocol).


| Axis      | Mean (Rater 1) | Mean (Rater 2) | Exact agreement | Agreement within \pm1 | Cohen’s \kappa |
| --------- | -------------- | -------------- | --------------- | --------------------- | -------------- |
| Style     | 3.97           | 3.23           | 26.7%           | 73.3%                 | −0.059         |
| Content   | 4.10           | 4.17           | 50.0%           | 90.0%                 | 0.185          |
| Artifacts | 4.33           | 3.30           | 26.7%           | 56.7%                 | 0.073          |
| Overall   | 4.13           | 3.57           | 34.4%           | —                     | —              |


Panels and rater files reside in `task3_gan/sarvesh/outputs/human_audit/`. Content ratings show the highest agreement; style and artifact axes exhibit greater rater disagreement, indicating that subjective style quality is less stable than content judgments for this sample set.

### 5.4 Failure Modes and Discussion

Observed failure modes include incomplete style transfer on some photographs, geometric distortion of scene layout, and localized textural artifacts. Cycle \ell_1 and cycle LPIPS indicate that reconstructions are imperfect but nontrivial, supporting that cycle constraints are active rather than vacuous. Relative to an earlier internal configuration with a much larger identity weight, reducing \lambda_{\mathrm{identity}} to 0.5 and adopting spectral normalization coincided with improved competition FID (approximately 104 to 94.5 under the local evaluation script).

Limitations include the well-known difficulty of FID on small artistic domains, residual content drift, and modest inter-rater reliability on style. Submissions used for the class competition are model inference outputs from the trained generators. Future work includes aligned qualitative panels across members on identical audit identifiers, unified scripts for KID/LPIPS/precision-recall, and further schedule or regularization sweeps.

---



## 6. Artifact Index


| Content                                     | Location                                                               |
| ------------------------------------------- | ---------------------------------------------------------------------- |
| Task 1 metrics and curves (Sarvesh)         | `task1_llm/sarvesh/metrics_report.csv`, `outputs/`                     |
| Task 1 metrics, ablations, curves (Naman)   | `task1_llm/naman/metrics_report.csv`, `outputs/`                       |
| Task 2 metrics and logs (Sarvesh)           | `task2_sentiment/sarvesh/metrics_report.csv`, `outputs/`, `logs/`      |
| Task 2 metrics and plots (Naman)            | `task2_sentiment/member_naman/metrics_report.csv`, `outputs/gpu/eval/` |
| Task 3 submission and predictions (Sarvesh) | `task3_gan/sarvesh/submission.csv`, `outputs/pred_*`                   |
| Task 3 extended metrics and audit (Sarvesh) | `full_metrics_report.csv`, `outputs/human_audit/`                      |
| Task 3 member package (Naman)               | `task3_gan/naman/`                                                     |
| Reproducibility manifests and raw logs      | `reproducibility/manifests/`, `reproducibility/raw_logs/`              |


---



## 7. References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*.
2. Eldan, R., & Li, Y. (2023). *TinyStories: How small can language models be and still speak coherent English?* arXiv preprint.
3. Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
4. Course materials and datasets: TinyStories; Yelp Polarity sentiment resources; class Kaggle competition for CycleGAN style transfer (DATA266, Fall 2026).

---

*End of report.*