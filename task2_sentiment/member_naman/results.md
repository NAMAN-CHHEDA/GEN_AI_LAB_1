# Task 2 results: gpu

## Overview
<!-- AUTO:overview -->
- Run: `gpu`, seed 42, dataset `fancyzhx/yelp_polarity`.
- Models trained from scratch (no pretrained embeddings or language models): ngram_bag, transformer, bigru.
- Baseline: `ngram_bag`; experimental: `transformer`, `bigru`.
<!-- /AUTO:overview -->

## Data and preprocessing
<!-- AUTO:data -->
- Train reviews: 504,000; validation: 56,000 (carved from the train split); test: 38,000 (original Hugging Face order).
- Vocabulary: 50,000 ids (max 50,000, min frequency 3), built from train tokens only.
- max_len 320 (head 25% + tail, applied at batching time for Transformer and BiGRU; the n-gram bag is not truncated).
- Cleaning steps used:
  1. lowercase; decode literal escapes (\n, \", \uXXXX); normalise curly apostrophes
  2. remove URLs, HTML tags and entities
  3. expand contractions (won't -> will not, can't -> can not, n't -> not); drop 're 's 'm 'll 've 'd
  4. keep only a-z, 0-9 and whitespace (digits kept)
  5. remove 37 neutral function words (negations, intensifiers and pronouns are kept)
  6. WordNet lemmatisation (exceptions: las, mrs, us, vegas)

| split | reviews | negative | positive | positive share | OOV tokens % | mean tokens | cut at max_len 320 (%) |
|---|---|---|---|---|---|---|---|
| train | 504000 | 251808 | 252192 | 0.500 | 0.432 | 93.04 | 2.71 |
| val | 56000 | 28192 | 27808 | 0.497 | 0.494 | 92.98 | 2.65 |
| test | 38000 | 19000 | 19000 | 0.500 | 0.489 | 92.7 | 2.65 |
<!-- /AUTO:data -->

## Model 1 (n-gram bag)
<!-- AUTO:model_ngram_bag -->
- Parameters (at this run's sizes): 65,600,033; input kind: `bag`.
- Training: AdamW, cosine schedule with 5% warm-up, batch size 128, EMA decay 0.999, patience 2.

| hyperparameter | value |
|---|---|
| embed_dim | 32.0 |
| bigram_buckets | 2000000.0 |
| dropout | 0.4 |
| lr | 0.002 |
| weight_decay | 0.0 |

Layers:
```
NGramBag(
  (bag): EmbeddingBag(2050000, 32, mode='mean')
  (dropout): Dropout(p=0.4, inplace=False)
  (out): Linear(in_features=32, out_features=1, bias=True)
)
```
<!-- /AUTO:model_ngram_bag -->

I chose a bag of unigrams and hashed bigrams because it is the simplest strong baseline: it ignores word order except for local pairs like "not good". I used a small 32-dimension embedding because the hashed bigram table (2M buckets) already holds most of the capacity. I expected it to land around 95%, since word and bigram counts carry most of the sentiment signal in Yelp reviews. It reached 95.98% test accuracy, trained in about 2.5 minutes and was by far the fastest at inference. It beat the transformer, so a simple model can be hard to beat here.

## Model 2 (Transformer)
<!-- AUTO:model_transformer -->
- Parameters (at this run's sizes): 6,838,017; input kind: `padded`.
- Training: AdamW, cosine schedule with 5% warm-up, batch size 128, EMA decay 0.999, patience 2.

| hyperparameter | value |
|---|---|
| d_model | 128 |
| layers | 2 |
| heads | 4 |
| ffn | 512 |
| dropout | 0.2 |
| token_dropout | 0.1 |
| pool | mean_max |
| lr | 0.0005 |

Layers:
```
TransformerClassifier(
  (embed): Embedding(50000, 128, padding_idx=0)
  (pos_embed): Embedding(320, 128)
  (embed_dropout): Dropout(p=0.2, inplace=False)
  (encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-1): 2 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=512, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
        (linear2): Linear(in_features=512, out_features=128, bias=True)
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
        (dropout1): Dropout(p=0.2, inplace=False)
        (dropout2): Dropout(p=0.2, inplace=False)
      )
    )
  )
  (final_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
  (head_dropout): Dropout(p=0.3, inplace=False)
  (out): Linear(in_features=256, out_features=1, bias=True)
)
```
<!-- /AUTO:model_transformer -->

I chose a small 2-layer transformer encoder (4 heads, trained from scratch) to test whether self-attention over the whole review helps compared to counting words. I used d_model 128 so it stays small enough to train quickly without pretraining. I expected it to be competitive with the baseline or slightly better on long reviews. It reached 94.28%, below both other models, and best validation came at epoch 2. Without pretrained weights, a small transformer has to learn language from 504K reviews alone, and that was not enough to beat word counts.

## Model 3 (bigram BiGRU)
<!-- AUTO:model_bigru -->
- Parameters (at this run's sizes): 38,977,601; input kind: `padded`.
- Training: AdamW, cosine schedule with 5% warm-up, batch size 128, EMA decay 0.999, patience 2.

| hyperparameter | value |
|---|---|
| word_dim | 128 |
| bigram_dim | 64 |
| bigram_buckets | 500000 |
| hidden | 128 |
| layers | 2 |
| rnn_dropout | 0.25 |
| emb_dropout | 0.25 |
| head_dropout | 0.4 |
| pooling | max_mean_attn |
| lr | 0.001 |

Layers:
```
BiGRUClassifier(
  (word_embed): Embedding(50000, 128, padding_idx=0)
  (bigram_embed): Embedding(500001, 64, padding_idx=0)
  (emb_dropout): Dropout(p=0.25, inplace=False)
  (gru): GRU(192, 128, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
  (attn_proj): Linear(in_features=256, out_features=128, bias=True)
  (attn_vector): Linear(in_features=128, out_features=1, bias=False)
  (head_dropout): Dropout(p=0.4, inplace=False)
  (out): Linear(in_features=768, out_features=1, bias=True)
)
```
<!-- /AUTO:model_bigru -->

I chose a 2-layer bidirectional GRU because it reads the review in order, which should help with negation and "but" clauses. I used 128-dimension word embeddings plus a 64-dimension hashed bigram embedding, and combined max, mean and attention pooling so it can use both the strongest and the overall sentiment signal. I expected it to be the best model. It reached 96.24%, the highest of the three, with the best calibration (ECE 0.003). It overfit quickly: the best validation result was at epoch 1, and early stopping ended training after epoch 3.

## Results
### Main test metrics
<!-- AUTO:results_main -->
| model | accuracy [95% CI] | macro-F1 [95% CI] | ROC-AUC | PR-AUC | MCC [95% CI] | Brier | ECE | McNemar p (vs baseline) | params | train time (s) | inference ex/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ngram_bag | 0.9598 [0.9577, 0.9616] | 0.9598 [0.9577, 0.9616] | 0.9907 | 0.9906 | 0.9196 [0.9154, 0.9232] | 0.0314 | 0.0059 | baseline | 65,600,033 | 147 | 197676 |
| transformer | 0.9428 [0.9406, 0.9451] | 0.9428 [0.9406, 0.9451] | 0.9867 | 0.9872 | 0.8857 [0.8812, 0.8902] | 0.0431 | 0.0128 | 1.23e-58 (b=1134, c=491) | 6,838,017 | 121 | 63449 |
| bigru | 0.9624 [0.9604, 0.9643] | 0.9624 [0.9604, 0.9643] | 0.9940 | 0.9942 | 0.9248 [0.9208, 0.9285] | 0.0278 | 0.0030 | 0.00198 (b=473, c=574) | 38,977,601 | 311 | 31904 |
<!-- /AUTO:results_main -->

### Metrics per slice
<!-- AUTO:results_slices -->
| model | slice | n | macro-F1 | error rate |
|---|---|---|---|---|
| ngram_bag | short | 9122 | 0.9550 | 0.0432 |
| ngram_bag | medium | 21412 | 0.9601 | 0.0399 |
| ngram_bag | long | 7466 | 0.9600 | 0.0375 |
| ngram_bag | has_negation | 27859 | 0.9574 | 0.0411 |
| ngram_bag | has_contrast | 21277 | 0.9545 | 0.0449 |
| transformer | short | 9122 | 0.9415 | 0.0562 |
| transformer | medium | 21412 | 0.9418 | 0.0582 |
| transformer | long | 7466 | 0.9409 | 0.0553 |
| transformer | has_negation | 27859 | 0.9362 | 0.0615 |
| transformer | has_contrast | 21277 | 0.9337 | 0.0655 |
| bigru | short | 9122 | 0.9558 | 0.0424 |
| bigru | medium | 21412 | 0.9639 | 0.0361 |
| bigru | long | 7466 | 0.9619 | 0.0358 |
| bigru | has_negation | 27859 | 0.9616 | 0.0371 |
| bigru | has_contrast | 21277 | 0.9584 | 0.0412 |
<!-- /AUTO:results_slices -->

### Training and efficiency
<!-- AUTO:results_training -->
| model | parameters | epochs run | best epoch | best val loss | best val acc | training time (s) | train examples/s | peak memory (MB) | memory note |
|---|---|---|---|---|---|---|---|---|---|
| ngram_bag | 65,600,033 | 4 | 2 | 0.1224 | 0.9582 | 147 | 13791 | 2018 | CUDA max_memory_allocated |
| transformer | 6,838,017 | 4 | 2 | 0.1612 | 0.9389 | 121 | 16898 | 702 | CUDA max_memory_allocated |
| bigru | 38,977,601 | 3 | 1 | 0.1016 | 0.9618 | 311 | 4943 | 1490 | CUDA max_memory_allocated |

| model | inference_examples_per_sec |
|---|---|
| ngram_bag | 197676.0 |
| transformer | 63449.0 |
| bigru | 31904.0 |
<!-- /AUTO:results_training -->

## Comparison and observations
<!-- AUTO:comparison_facts -->
- Highest test accuracy: `bigru` (0.9624, 95% CI [0.9604, 0.9643]).
- `transformer` vs baseline: accuracy difference -0.0169; McNemar b=1134, c=491, exact p=1.23e-58.
- `bigru` vs baseline: accuracy difference +0.0027; McNemar b=473, c=574, exact p=0.00198.
<!-- /AUTO:comparison_facts -->

The BiGRU is best (96.24%), the n-gram bag is close (95.98%), and the transformer is lowest (94.28%). The BiGRU beats the n-gram bag by only 0.27 points, but a paired McNemar test shows the difference is real (p = 0.002), and the transformer is significantly worse than the baseline. I expected the transformer to do better than it did. Accuracy is similar across the short, medium and long slices, so none of the models fails on review length; reviews with a contrast word ("but", "however") are the hardest slice for all three. In the BiGRU error review, many errors look like label noise (a few reviews are clearly positive or negative but labeled the opposite way), and most of the rest are mixed-sentiment or contrast reviews. Sarcasm, negation scope and very short reviews appear only a few times.

## Hardware disclosure
<!-- AUTO:hardware -->
- Training device per model: ngram_bag: NVIDIA GeForce RTX 4090, transformer: NVIDIA GeForce RTX 4090, bigru: NVIDIA GeForce RTX 4090
- Inference device per model: ngram_bag: NVIDIA GeForce RTX 4090, transformer: NVIDIA GeForce RTX 4090, bigru: NVIDIA GeForce RTX 4090
- torch 2.14.1+cu126, Python 3.12.10, Windows 11
- This machine: 32 logical CPUs, 128 GB RAM; CUDA available: True (NVIDIA GeForce RTX 4090)
- Mixed precision setting: True (used only on CUDA).
- Seed 42; cuDNN deterministic mode on.
<!-- /AUTO:hardware -->

## Limitations and future work
<!-- AUTO:limitations_facts -->
- Test reviews used in this run: all.
- Train reviews used: all remaining; epochs: 8.
- Single seed per model (no repeated runs), so differences smaller than the bootstrap intervals are not conclusive.
<!-- /AUTO:limitations_facts -->

Limitations: each model was trained once with one seed, so I have no estimate of run-to-run variation; the transformer was small and lightly tuned, so its lower score does not mean transformers are worse in general; some labels look noisy, which limits the achievable accuracy; 2.7% of reviews are cut at 320 tokens; and GPU training may not be bitwise reproducible. The test set was used once and only for the final evaluation. Future work: add trigram features or give more weight to the final sentences to handle contrast and past-vs-present reviews; run several seeds and report the spread; tune the transformer (more layers, longer training); estimate the label-noise rate on a hand-checked sample; and try character n-grams for typos and rare words.