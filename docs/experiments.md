# Parameter Experiments (NLP for Games)

**Task:** Lore generation about *Ebonridge Keep* (somber tone)  
**Model:** distilgpt2 (CPU)  
**Params swept:** temperature = [0.7, 0.9, 1.1], top_p = [0.85, 0.9], seed = [1, 2]  
**Max new tokens:** 120

## Observations
- **Temperature ↑** → more variety and unusual phrasing; repetition risk up slightly (see `repetition_2gram` column).
- **top_p = 0.85 vs 0.9** → 0.85 feels safer/cleaner; 0.9 gives bolder word choices.
- **Seed changes** structure and details even with fixed params.
- **Runtime** stayed low on CPU after model load (see `runtime_sec`).

## Sample rows (see `outputs/experiments/lore_grid.csv`)
Include 2–4 short snippets here that contrast temp/top_p.

## Next ideas
- Sweep `repetition_penalty` (1.1 vs 1.3).
- Try longer `max_new_tokens` for quests.
- Swap in a slightly larger model for richer vocabulary if time permits.

