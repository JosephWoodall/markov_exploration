import random
from collections import defaultdict
from typing import Any, Callable, Sequence

from predictor import UniversalPredictor


class PredictorForest:
    """
    N UniversalPredictor instances that diverge through independent feedback dropout.

    Each tree observes the same sequence in the same order, but each independently
    skips learning from some steps at random. This breaks symmetry: different nodes
    get created first, different coupling links form, different credibilities develop.
    The topologies grow differently from identical starting conditions.

    At prediction time each tree votes; votes are weighted by confidence. The winner
    is whichever outcome accumulates the most confidence-weighted support.

    The public interface matches UniversalPredictor exactly — it is a drop-in
    replacement anywhere a single predictor is used.
    """

    def __init__(
        self,
        context_length: int,
        similarity_fn: Callable[[Sequence, Sequence], float] | None = None,
        learning_rate: float = 0.1,
        coupling_lr: float = 0.3,
        feedback_strength: float = 0.3,
        vigilance: float = 0.7,
        min_context_length: int = 1,
        coupling_ema: bool = True,
        n_trees: int = 5,
        dropout: float = 0.2,
        seed: int = 42,
    ):
        self.n       = n_trees
        self.dropout = dropout

        master_rng  = random.Random(seed)
        self._rngs  = [random.Random(master_rng.randint(0, 2**32)) for _ in range(n_trees)]

        self.trees = [
            UniversalPredictor(
                context_length,
                similarity_fn,
                learning_rate=learning_rate,
                coupling_lr=coupling_lr,
                feedback_strength=feedback_strength,
                vigilance=vigilance,
                min_context_length=min_context_length,
                coupling_ema=coupling_ema,
            )
            for _ in range(n_trees)
        ]

    # ── public interface ──────────────────────────────────────────────────────

    def observe(self, value: Any) -> None:
        for tree in self.trees:
            tree.observe(value)

    def predict(self) -> tuple[Any, float]:
        votes: dict[Any, float] = defaultdict(float)
        total_conf = 0.0

        for tree in self.trees:
            pred, conf = tree.predict()
            if pred is not None and conf > 0:
                votes[pred] += conf
                total_conf  += conf

        if not votes:
            return None, 0.0

        best       = max(votes, key=votes.get)
        confidence = votes[best] / total_conf
        return best, confidence

    def feedback(self, actual: Any) -> None:
        # Each tree independently decides whether to learn from this step.
        # Skipped steps leave _last_context stale but harmlessly unused.
        for tree, rng in zip(self.trees, self._rngs):
            if rng.random() >= self.dropout:
                tree.feedback(actual)

    # ── diagnostics (aggregated across trees) ────────────────────────────────

    def node_stats(self) -> dict:
        all_stats = [tree.node_stats() for tree in self.trees]
        result = {}
        for key in ('total_nodes', 'observed', 'exploration', 'correction',
                    'coupling_links', 'allocator_trials'):
            result[key] = sum(s[key] for s in all_stats)
        for key in ('mean_coupling', 'lambda', 'optimizer_rolling_acc'):
            result[key] = sum(s[key] for s in all_stats) / self.n
        result['max_coupling']    = max(s['max_coupling']    for s in all_stats)
        result['optimizer_budget'] = int(sum(s['optimizer_budget'] for s in all_stats) / self.n)
        return result

    def similarity_quality(self) -> float:
        return sum(t.similarity_quality() for t in self.trees) / self.n

    def convergence_state(self) -> dict:
        states   = [tree.convergence_state() for tree in self.trees]
        qualities = [s['quality_now'] for s in states]
        median_q  = sorted(qualities)[self.n // 2]
        idx       = min(range(self.n), key=lambda i: abs(qualities[i] - median_q))
        return states[idx]

    def lookahead_quality(self, n_steps: int) -> float:
        return sum(t.lookahead_quality(n_steps) for t in self.trees) / self.n
