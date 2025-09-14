import argparse, json, os, random, time
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .prompts import TEMPLATES, SYSTEM_STYLE

DEFAULT_MODEL = "distilgpt2"

@dataclass
class GenConfig:
    max_new_tokens: int = 120
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    seed: int = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_prompt(kind: str, params: Dict[str, Any]) -> str:
    if kind not in TEMPLATES:
        raise ValueError(f"Unknown type {kind}. Choose from {list(TEMPLATES.keys())}")
    try:
        user = TEMPLATES[kind].format(**params)
    except KeyError as e:
        raise ValueError(f"Missing parameter for template: {e}")
    return f"{SYSTEM_STYLE}\n\n{user}\n\nOutput:"

def generate_text(model_name: str, prompt: str, cfg: GenConfig) -> str:
    set_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tok(prompt, return_tensors="pt")
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

def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    p = argparse.ArgumentParser(description="Game text generator")
    p.add_argument("--type", required=True, choices=list(TEMPLATES.keys()))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--place", default="Ebonridge Keep")
    p.add_argument("--tone", default="somber")
    p.add_argument("--level", type=int, default=8)
    p.add_argument("--biome", default="Crystal Marsh")
    p.add_argument("--name", default="Aetherglass Dagger")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--faction", default="Verdant Wardens")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = GenConfig(args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.repetition_penalty, args.seed)
    content_params = {"place": args.place, "tone": args.tone, "level": args.level, "biome": args.biome,
                      "name": args.name, "n": args.n, "faction": args.faction}

    prompt = build_prompt(args.type, content_params)
    t0 = time.time()
    text = generate_text(args.model, prompt, cfg)
    dt = round(time.time() - t0, 2)

    record = {
        "type": args.type,
        "content_params": content_params,
        "gen_params": asdict(cfg),
        "model": args.model,
        "runtime_sec": dt,
        "prompt": prompt,
        "text": text.strip(),
    }
    stamp = int(time.time())
    out_path = f"outputs/samples/{args.type}_{stamp}.json"
    save_json(record, out_path)
    print(f"Saved {out_path}\n")
    print(record["text"][:800])

if __name__ == "__main__":
    main()

