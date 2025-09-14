# src/prompts.py
"""
Prompt builder that supports either:
  - build_prompt(kind, params)  [via wrapper in generator.py]
  - build_prompt(params_dict_with_type)  [this function]
"""

def build_prompt(params):
    """
    params: dict expected to contain a 'type' key plus fields used below.
    """
    t = (params or {}).get("type", "lore")

    if t == "lore":
        place = params.get("place", "a forgotten citadel")
        tone = params.get("tone", "mysterious")
        return (
            f"Write a short piece of fantasy lore about {place}. "
            f"The tone should be {tone}. Keep it vivid and coherent."
        )

    elif t == "quest":
        level = params.get("level", 8)
        biome = params.get("biome", "ancient ruins")
        return (
            "Design a concise quest hook. "
            f"Player level: {level}. Location/biome: {biome}. "
            "Include an objective, an obstacle, and a reward in one or two sentences."
        )

    elif t == "item":
        name = params.get("name", "Aetherglass Dagger")
        return (
            f"Describe a rare fantasy item named '{name}'. "
            "Include rarity, effect, drawback, and a short usage tip (3–5 sentences)."
        )

    elif t == "names":
        n = params.get("n", 8)
        faction = params.get("faction", "Verdant Wardens")
        return (
            f"Generate {n} fantasy character names for the {faction} faction. "
            "Return one name per line, no numbering."
        )

    # fallback
    return "Write a short fantasy-themed text snippet."

