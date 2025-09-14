# src/metrics.py
from collections import Counter
from typing import Dict, List, Union

# minimal wordlist just to show the mechanism
PROFANITY = {"nsfwword1", "nsfwword2"}

def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in text.split()]

def distinct_n(words: List[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    total = len(words) - n + 1
    return len(ngrams) / total if total > 0 else 0.0

def repetition_score(words: List[str], n: int = 2) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    total = max(1, len(ngrams))
    return repeats / total

def simple_toxicity_flags(words: List[str]) -> int:
    return sum(1 for w in words if w in PROFANITY)

def compute_metrics(text: str) -> Dict[str, Union[float, int]]:
    words = tokenize_words(text)
    return {
        "char_len": len(text),
        "word_len": len(words),
        "distinct_1": round(distinct_n(words, 1), 4),
        "distinct_2": round(distinct_n(words, 2), 4),
        "repetition_2gram": round(repetition_score(words, 2), 4),
        "toxicity_hits": simple_toxicity_flags(words),
    }

