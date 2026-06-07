"""
Self-supervised PPMI co-occurrence embeddings for sequence similarity.

Learns a vector for each unique symbol from a training sequence by counting
how often symbols appear near each other.  Two symbols are similar if they
appear next to the same neighbours — the distributional hypothesis applied at
the symbol level.

Context similarity is the cosine similarity of mean-pooled symbol vectors,
optionally weighted toward the most recent positions.

No external dependencies.  Works for any discrete sequence: text, DNA,
categorical events, discretised numbers.
"""

import math
from typing import Any, Callable, Sequence


def learn_embeddings(seq: list, window: int = 3) -> dict[Any, list[float]]:
    """
    Build L2-normalised PPMI co-occurrence embeddings from a training sequence.

    window  — context radius: count co-occurrences within ±window positions.
              Wider windows capture longer-range dependencies at the cost of
              blurring local structure.

    Returns {symbol: vector} where each vector has dimension = |vocab|.
    For small vocabularies (text ≤ 60 chars, DNA = 4, weather ≤ 10) this is
    already compact; no SVD needed.
    """
    vocab = sorted(set(seq), key=str)
    idx   = {c: i for i, c in enumerate(vocab)}
    n     = len(vocab)

    cooc = [[0.0] * n for _ in range(n)]
    for i, c in enumerate(seq):
        ci = idx[c]
        for j in range(max(0, i - window), min(len(seq), i + window + 1)):
            if i != j:
                cooc[ci][idx[seq[j]]] += 1.0

    total = sum(cooc[i][j] for i in range(n) for j in range(n)) or 1.0
    row_s = [sum(cooc[i]) or 1.0 for i in range(n)]
    col_s = [sum(cooc[i][j] for i in range(n)) or 1.0 for j in range(n)]

    ppmi: list[list[float]] = []
    for i in range(n):
        row = []
        for j in range(n):
            if cooc[i][j] > 0:
                val = math.log(cooc[i][j] * total / (row_s[i] * col_s[j]))
                row.append(max(0.0, val))
            else:
                row.append(0.0)
        ppmi.append(row)

    embeddings: dict[Any, list[float]] = {}
    for i, c in enumerate(vocab):
        vec  = ppmi[i]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        embeddings[c] = [v / norm for v in vec]

    return embeddings


def make_embedding_sim(
    embeddings: dict[Any, list[float]],
    recency: bool = True,
) -> Callable[[Sequence, Sequence], float]:
    """
    Return a similarity function over context sequences.

    Each context is embedded by mean-pooling its symbol vectors.
    When recency=True, more recent positions (end of context) get linearly
    higher weight — matching the intuition that the most recent symbols are
    the most predictive.

    Similarity = cosine(embed(ctx_a), embed(ctx_b)), remapped from [-1,1]
    to [0,1].  Unknown symbols contribute a zero vector (no attraction or
    repulsion — honest ignorance).
    """
    dim      = len(next(iter(embeddings.values()))) if embeddings else 0
    zero_vec = [0.0] * dim

    def _embed(ctx: Sequence) -> list[float]:
        n = len(ctx)
        if n == 0:
            return zero_vec[:]
        result  = [0.0] * dim
        total_w = 0.0
        for i, c in enumerate(ctx):
            w   = (i + 1) / n if recency else 1.0
            vec = embeddings.get(c, zero_vec)
            for d in range(dim):
                result[d] += w * vec[d]
            total_w += w
        if total_w > 0:
            result = [v / total_w for v in result]
        return result

    def sim(ctx_a: Sequence, ctx_b: Sequence) -> float:
        va  = _embed(ctx_a)
        vb  = _embed(ctx_b)
        dot = sum(a * b for a, b in zip(va, vb))
        na  = math.sqrt(sum(a * a for a in va))
        nb  = math.sqrt(sum(b * b for b in vb))
        if na < 1e-10 or nb < 1e-10:
            return 1.0 if list(ctx_a) == list(ctx_b) else 0.0
        return (dot / (na * nb) + 1.0) / 2.0

    return sim
