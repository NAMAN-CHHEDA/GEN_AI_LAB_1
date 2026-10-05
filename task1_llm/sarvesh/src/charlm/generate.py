"""Text generation helpers and sample dumping."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .data import decode_ids
from .metrics import encode_prompt
from .model import CharGPT


def run_generation_suite(
    model: CharGPT,
    cfg: dict,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, float], dict]:
    gcfg = cfg["generation"]
    run_name = cfg["run_name"]
    sample_dir = output_dir / f"{run_name}_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, list[str]] = {}
    speeds: dict[str, float] = {}
    detailed: dict[str, list[dict]] = {}

    max_new = int(gcfg["max_new_tokens"])
    prompts = list(gcfg["prompts"])

    for setting in gcfg["settings"]:
        name = setting["name"]
        texts: list[str] = []
        detail_rows: list[dict] = []
        greedy = bool(setting.get("greedy", False))

        for i, prompt in enumerate(prompts):
            local_seed = seed + i * 17
            torch.manual_seed(local_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(local_seed)

            # Generator device must match the probs tensor device (CUDA vs CPU).
            gen = None
            if not greedy:
                gen = torch.Generator(device=device)
                gen.manual_seed(local_seed)

            idx = encode_prompt(prompt, char_to_idx, device)
            out = model.generate(
                idx,
                max_new_tokens=max_new,
                temperature=float(setting.get("temperature", 1.0)),
                top_k=setting.get("top_k"),
                top_p=setting.get("top_p"),
                greedy=greedy,
                generator=gen,
            )
            full = decode_ids(out[0].tolist(), idx_to_char)
            texts.append(full)
            detail_rows.append({"prompt": prompt, "text": full})

        model.eval()
        speed_prompt = encode_prompt(prompts[0], char_to_idx, device)
        t0 = time.perf_counter()
        _ = model.generate(
            speed_prompt,
            max_new_tokens=max_new,
            temperature=float(setting.get("temperature", 1.0)),
            top_k=setting.get("top_k"),
            top_p=setting.get("top_p"),
            greedy=greedy,
        )
        speeds[name] = max_new / max(1e-8, time.perf_counter() - t0)

        generated[name] = texts
        detailed[name] = detail_rows
        (sample_dir / f"{name}.txt").write_text(
            "\n\n---\n\n".join(f"PROMPT: {r['prompt']}\n{r['text']}" for r in detail_rows),
            encoding="utf-8",
        )
        print(f"generated setting: {name} | gen tok/s ~ {speeds[name]:.0f}")

    out_json = {
        "run_name": run_name,
        "max_new_tokens": max_new,
        "settings": detailed,
        "speeds_tokens_per_sec": speeds,
    }
    with open(output_dir / f"{run_name}_samples.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    return generated, speeds, out_json
