# src/generator.py
import argparse, json, os, time
from dataclasses import dataclass, asdict
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .metrics import compute_metrics
from .prompts import build_prompt as _build_prompt_params  # underlying builder


@dataclass
class GenConfig:
    max_new_tokens: int = 120
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    seed: int = 42


def build_prompt(kind: str, params: Dict[str, Any]) -> str:
    """Wrapper to stay compatible with experiments.py (kind, params)."""
    merged = dict(params or {})
    merged["type"] = kind
    return _build_prompt_params(merged)


def generate_text(model_name: str, prompt: str, cfg: GenConfig) -> str:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tok(prompt, return_tensors="pt")

    # Seed for reproducibility on CPU
    torch.manual_seed(cfg.seed)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Game text generator")
    ap.add_argument("--type", required=True, choices=["lore", "quest", "item", "names"])
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--outdir", default="outputs/samples")

    # generation params
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=42)

    # content params
    ap.add_argument("--place", default="Ebonridge Keep")
    ap.add_argument("--tone", default="somber")
    ap.add_argument("--level", type=int, default=8)
    ap.add_argument("--biome", default="Crystal Marsh")
    ap.add_argument("--name", default="Aetherglass Dagger")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--faction", default="Verdant Wardens")

    args = ap.parse_args()

    content_params = {
        "place": args.place,
        "tone": args.tone,
        "level": args.level,
        "biome": args.biome,
        "name": args.name,
        "n": args.n,
        "faction": args.faction,
    }

    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    prompt = build_prompt(args.type, content_params)
    t0 = time.time()
    text = generate_text(args.model, prompt, cfg)
    dt = round(time.time() - t0, 3)

    metrics = compute_metrics(text)
    record = {
        "type": args.type,
        "content_params": content_params,
        "gen_params": asdict(cfg),
        "model": args.model,
        "runtime_sec": dt,
        "prompt": prompt,
        "text": text.strip(),
        "metrics": metrics,
    }

    _ensure_dir(args.outdir)
    out_path = os.path.join(args.outdir, f"{args.type}_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"Saved {out_path}")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

