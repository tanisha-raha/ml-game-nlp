# src/experiments.py
import argparse, csv, os, time
from dataclasses import asdict
from .generator import GenConfig, build_prompt, generate_text
from .metrics import compute_metrics

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run_one(kind: str, content_params: dict, model: str, cfg: GenConfig):
    prompt = build_prompt(kind, content_params)
    t0 = time.time()
    text = generate_text(model, prompt, cfg)
    dt = round(time.time() - t0, 3)
    mets = compute_metrics(text)
    rec = {
        "type": kind,
        "model": model,
        **content_params,
        **asdict(cfg),
        "runtime_sec": dt,
        **mets,
        "text": text.strip().replace("\n", " "),
    }
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["lore","quest","item","names"])
    ap.add_argument("--outdir", default="outputs/experiments")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--temperatures", default="0.7,0.9,1.1")
    ap.add_argument("--top_ps", default="0.85,0.9")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    # content defaults
    ap.add_argument("--place", default="Ebonridge Keep")
    ap.add_argument("--tone", default="somber")
    ap.add_argument("--level", type=int, default=8)
    ap.add_argument("--biome", default="Crystal Marsh")
    ap.add_argument("--name", default="Aetherglass Dagger")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--faction", default="Verdant Wardens")
    args = ap.parse_args()

    content_params = {
        "place": args.place, "tone": args.tone, "level": args.level, "biome": args.biome,
        "name": args.name, "n": args.n, "faction": args.faction,
    }

    ensure_dir(args.outdir)
    csv_path = os.path.join(args.outdir, f"{args.type}_grid.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type","model","place","tone","level","biome","name","n","faction",
            "max_new_tokens","temperature","top_p","top_k","repetition_penalty","seed",
            "runtime_sec","char_len","word_len","distinct_1","distinct_2","repetition_2gram","toxicity_hits","text"
        ])
        writer.writeheader()
        for seed in [int(s) for s in args.seeds.split(",")]:
            for temp in [float(x) for x in args.temperatures.split(",")]:
                for top_p in [float(x) for x in args.top_ps.split(",")]:
                    cfg = GenConfig(
                        max_new_tokens=args.max_new_tokens,
                        temperature=temp,
                        top_p=top_p,
                        top_k=50,
                        repetition_penalty=1.1,
                        seed=seed,
                    )
                    rec = run_one(args.type, content_params, args.model, cfg)
                    writer.writerow(rec)
    print(f"Saved grid to {csv_path}")

if __name__ == "__main__":
    main()

