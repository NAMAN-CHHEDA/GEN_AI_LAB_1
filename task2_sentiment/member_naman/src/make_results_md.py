"""Write results.md: facts are filled in automatically between AUTO markers, my own words go elsewhere.

Run:  python src/make_results_md.py --config local

Text between <!-- AUTO:name --> and <!-- /AUTO:name --> is replaced on every run. Everything else in an
existing results.md (my explanations) is never touched.
"""

import argparse
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import LEMMA_EXCEPTIONS, STOPWORDS
from evaluate import SLICE_NAMES
from models import build_model
from predict import MODEL_NAMES
from utils import load_config, load_json, project_root

MODEL_TITLES = {"ngram_bag": "Model 1 (n-gram bag)", "transformer": "Model 2 (Transformer)",
                "bigru": "Model 3 (bigram BiGRU)"}
TODO_MODEL = ("TODO (my words): why this architecture, why this embedding size, what I expected, what happened.")

# The document skeleton: headings, AUTO blocks (replaced on every run) and my own TODO placeholders.
TEMPLATE = """# Task 2 results: {run_name}

## Overview
<!-- AUTO:overview -->
<!-- /AUTO:overview -->

## Data and preprocessing
<!-- AUTO:data -->
<!-- /AUTO:data -->

## Model 1 (n-gram bag)
<!-- AUTO:model_ngram_bag -->
<!-- /AUTO:model_ngram_bag -->

{todo}

## Model 2 (Transformer)
<!-- AUTO:model_transformer -->
<!-- /AUTO:model_transformer -->

{todo}

## Model 3 (bigram BiGRU)
<!-- AUTO:model_bigru -->
<!-- /AUTO:model_bigru -->

{todo}

## Results
### Main test metrics
<!-- AUTO:results_main -->
<!-- /AUTO:results_main -->

### Metrics per slice
<!-- AUTO:results_slices -->
<!-- /AUTO:results_slices -->

### Training and efficiency
<!-- AUTO:results_training -->
<!-- /AUTO:results_training -->

## Comparison and observations
<!-- AUTO:comparison_facts -->
<!-- /AUTO:comparison_facts -->

TODO (my words): why the models differ, what the slices and the error review show, what I expected and what happened.

## Hardware disclosure
<!-- AUTO:hardware -->
<!-- /AUTO:hardware -->

## Limitations and future work
<!-- AUTO:limitations_facts -->
<!-- /AUTO:limitations_facts -->

TODO (my words): limitations and future work.
"""


def table(df):
    """DataFrame -> markdown table (no extra dependency)."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def load_summaries(cfg):
    out = cfg["paths"]["output_dir"]
    return {n: load_json(out / n / "summary.json") for n in MODEL_NAMES if (out / n / "summary.json").exists()}


def overview_block(cfg, summaries):
    return "\n".join([f"- Run: `{cfg['run_name']}`, seed {cfg['seed']}, dataset `{cfg['data']['dataset']}`.",
                      f"- Models trained from scratch (no pretrained embeddings or language models): {', '.join(summaries) or 'none yet'}.",
                      "- Baseline: `ngram_bag`; experimental: `transformer`, `bigru`."])


def data_block(cfg):
    info = load_json(cfg["paths"]["processed_dir"] / "split_info.json")
    d = cfg["data"]
    with np.load(cfg["paths"]["processed_dir"] / "arrays.npz") as npz:
        cut = {s: 100 * float((npz[f"{s}_n_tokens"] > d["max_len"]).mean()) for s in ("train", "val", "test")}
    rows = [{"split": s, "reviews": info["counts"][s], "negative": info["class_counts"][s]["label_0"],
             "positive": info["class_counts"][s]["label_1"],
             "positive share": f"{info['class_counts'][s]['label_1'] / info['counts'][s]:.3f}",
             "OOV tokens %": info["oov_token_percent"][s], "mean tokens": info["mean_tokens_per_review"][s],
             f"cut at max_len {d['max_len']} (%)": f"{cut[s]:.2f}"} for s in ("train", "val", "test")]
    steps = [
        "lowercase; decode literal escapes (\\n, \\\", \\uXXXX); normalise curly apostrophes",
        "remove URLs, HTML tags and entities",
        "expand contractions (won't -> will not, can't -> can not, n't -> not); drop 're 's 'm 'll 've 'd",
        f"keep only a-z{', 0-9' if d['keep_digits'] else ''} and whitespace (digits {'kept' if d['keep_digits'] else 'removed'})",
        f"remove {len(STOPWORDS)} neutral function words (negations, intensifiers and pronouns are kept)",
        (f"WordNet lemmatisation (exceptions: {', '.join(sorted(LEMMA_EXCEPTIONS))})" if d["lemmatize"] else "no lemmatisation"),
    ]
    return "\n".join(
        [f"- Train reviews: {info['counts']['train']:,}; validation: {info['counts']['val']:,} (carved from the train split); "
         f"test: {info['counts']['test']:,} (original Hugging Face order).",
         f"- Vocabulary: {info['vocab_size']:,} ids (max {d['vocab_max']:,}, min frequency {d['min_freq']}), built from train tokens only.",
         f"- max_len {d['max_len']} (head {d['head_fraction']:.0%} + tail, applied at batching time for Transformer and BiGRU; the n-gram bag is not truncated).",
         "- Cleaning steps used:"] + [f"  {i + 1}. {s}" for i, s in enumerate(steps)] + ["", table(pd.DataFrame(rows))])


def model_block(cfg, name):
    vocab_size = len(load_json(cfg["paths"]["processed_dir"] / "vocab.json"))
    model = build_model(name, cfg, vocab_size)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hyper = pd.DataFrame([{"hyperparameter": k, "value": v} for k, v in cfg["models"][name].items()])
    return "\n".join([f"- Parameters (at this run's sizes): {params:,}; input kind: `{model.input_kind}`.",
                      f"- Training: AdamW, cosine schedule with {cfg['train']['lr_warmup_ratio']:.0%} warm-up, "
                      f"batch size {cfg['train']['batch_size']}, EMA decay {cfg['train']['ema_decay']}, patience {cfg['train']['patience']}.",
                      "", table(hyper), "", "Layers:", "```", str(model), "```"])


def results_blocks(cfg):
    out = cfg["paths"]["output_dir"]
    if not (out / "metrics_report.csv").exists():
        message = "Not available yet: run predict.py and evaluate.py."
        return message, message
    report = pd.read_csv(out / "metrics_report.csv")
    slices = []
    for _, r in report.iterrows():
        for s in SLICE_NAMES:
            slices.append({"model": r["model"], "slice": s, "n": int(r[f"slice_{s}_n"]),
                           "macro-F1": f"{r[f'slice_{s}_macro_f1']:.4f}", "error rate": f"{r[f'slice_{s}_error_rate']:.4f}"})
    return (out / "metrics_table.md").read_text(encoding="utf-8").strip(), table(pd.DataFrame(slices))


def training_block(cfg, summaries, report):
    if not summaries:
        return "Not available yet: no model has finished training."
    rows = [{"model": n, "parameters": f"{s['parameters']:,}", "epochs run": s["epochs_run"], "best epoch": s["best_epoch"],
             "best val loss": f"{s['best_val_loss']:.4f}", "best val acc": f"{s['best_val_accuracy']:.4f}",
             "training time (s)": f"{s['total_training_time_s']:.0f}", "train examples/s": f"{s['train_examples_per_sec']:.0f}",
             "peak memory (MB)": f"{s['peak_memory_mb']:.0f}", "memory note": s["peak_memory_note"]} for n, s in summaries.items()]
    text = table(pd.DataFrame(rows))
    if report is not None:
        text += "\n\n" + table(report[["model", "inference_examples_per_sec"]].round(0))
    return text


def comparison_block(report):
    if report is None:
        return "Not available yet."
    baseline = report[report["role"] == "baseline"].iloc[0]
    best = report.loc[report["accuracy"].idxmax()]
    lines = [f"- Highest test accuracy: `{best['model']}` ({best['accuracy']:.4f}, 95% CI [{best['accuracy_ci_low']:.4f}, {best['accuracy_ci_high']:.4f}])."]
    for _, r in report[report["role"] == "experimental"].iterrows():
        lines.append(f"- `{r['model']}` vs baseline: accuracy difference {r['accuracy'] - baseline['accuracy']:+.4f}; "
                     f"McNemar b={int(r['mcnemar_b'])}, c={int(r['mcnemar_c'])}, exact p={r['mcnemar_p_exact']:.3g}.")
    return "\n".join(lines)


def hardware_block(cfg, summaries, report):
    lines = [f"- Training device per model: " + (", ".join(f"{n}: {s['device']}" for n, s in summaries.items()) or "n/a"),
             f"- Inference device per model: " + (", ".join(f"{r['model']}: {r['inference_device']}" for _, r in report.iterrows()) if report is not None else "n/a"),
             f"- torch {torch.__version__}, Python {platform.python_version()}, {platform.system()} {platform.release()}",
             f"- This machine: {psutil.cpu_count(logical=True)} logical CPUs, {psutil.virtual_memory().total / 2**30:.0f} GB RAM; "
             f"CUDA available: {torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""),
             f"- Mixed precision setting: {cfg['train']['amp']} (used only on CUDA).",
             f"- Seed {cfg['seed']}; cuDNN deterministic mode on."]
    return "\n".join(lines)


def limitations_block(cfg):
    d = cfg["data"]
    lines = [f"- Test reviews used in this run: {'all' if d['test_samples'] is None else str(d['test_samples']) + ' (a subset)'}.",
             f"- Train reviews used: {'all remaining' if d['train_samples'] is None else d['train_samples']}; epochs: {cfg['train']['epochs']}.",
             "- Single seed per model (no repeated runs), so differences smaller than the bootstrap intervals are not conclusive."]
    return "\n".join(lines)


def build_blocks(cfg):
    summaries = load_summaries(cfg)
    report_path = cfg["paths"]["output_dir"] / "metrics_report.csv"
    report = pd.read_csv(report_path) if report_path.exists() else None
    main_table, slice_table = results_blocks(cfg)
    blocks = {"overview": overview_block(cfg, summaries), "data": data_block(cfg),
              "results_main": main_table, "results_slices": slice_table,
              "results_training": training_block(cfg, summaries, report), "comparison_facts": comparison_block(report),
              "hardware": hardware_block(cfg, summaries, report), "limitations_facts": limitations_block(cfg)}
    for name in MODEL_NAMES:
        blocks[f"model_{name}"] = model_block(cfg, name)
    return blocks


def marker_pattern(name):
    return re.compile(rf"(<!-- AUTO:{name} -->\n).*?\n?(<!-- /AUTO:{name} -->)", re.DOTALL)


def update_results_file(path, blocks, run_name):
    """Create the file from the template, or replace only the AUTO blocks of an existing one."""
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else TEMPLATE.format(run_name=run_name, todo=TODO_MODEL)
    missing = []
    for name, content in blocks.items():
        pattern = marker_pattern(name)
        if pattern.search(text):
            text = pattern.sub(lambda m: m.group(1) + content + "\n" + m.group(2), text)
        else:
            missing.append(name)
    path.write_text(text, encoding="utf-8")
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write results.md with automatic facts and TODO placeholders")
    parser.add_argument("--config", default="local")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    blocks = build_blocks(cfg)
    targets = [cfg["paths"]["output_dir"] / "results.md"]
    if cfg["run_name"] == "gpu":
        targets.append(project_root() / "results.md")
    for target in targets:
        missing = update_results_file(target, blocks, cfg["run_name"])
        print(f"wrote {target.relative_to(project_root()).as_posix()}" + (f" (no marker found for: {missing})" if missing else ""))


if __name__ == "__main__":
    main()
