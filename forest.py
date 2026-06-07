import math
import random
from collections import defaultdict
from typing import Any, Callable, Sequence

from predictor import UniversalPredictor


class PredictorForest:
    """
    A forest of UniversalPredictor instances that start identical and diverge
    through four mechanisms:

      1. Heterogeneous k       — tree i uses context_length + i, so each tree
                                 specialises in a different temporal scale.
      2. Feedback dropout      — each tree independently skips some learning steps,
                                 causing different topologies to grow from the same data.
      3. Staggered offsets     — tree i starts learning after i * stagger steps,
                                 so each tree builds its early topology from a
                                 different slice of the sequence.
      4. Inter-tree credibility — each tree earns a persistent weight based on its
                                 track record. Trees that have been right recently
                                 speak louder in the vote.

    At prediction time two voting modes are available:

      mixture  — confidence-weighted sum of distributions (soft, inclusive)
      product  — weighted geometric mean of distributions (selective: an outcome
                 wins only when most trees agree on it; disagreement collapses to
                 uncertainty, which is correct behaviour on random data)

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
        voting: str = 'product',        # 'product' | 'mixture'
        heterogeneous_k: bool = True,   # tree i uses context_length + i
        stagger: int = 0,               # tree i defers learning for i * stagger steps
        tree_lr: float = 0.1,           # learning rate for inter-tree credibility
    ):
        self.n        = n_trees
        self.dropout  = dropout
        self.voting   = voting
        self.tree_lr  = tree_lr

        master_rng = random.Random(seed)
        self._rngs  = [random.Random(master_rng.randint(0, 2**32)) for _ in range(n_trees)]

        k_values = (
            [context_length + i for i in range(n_trees)]
            if heterogeneous_k
            else [context_length] * n_trees
        )

        self.trees = [
            UniversalPredictor(
                k_values[i],
                similarity_fn,
                learning_rate=learning_rate,
                coupling_lr=coupling_lr,
                feedback_strength=feedback_strength,
                vigilance=vigilance,
                min_context_length=min_context_length,
                coupling_ema=coupling_ema,
            )
            for i in range(n_trees)
        ]

        self._offsets    = [i * stagger for i in range(n_trees)]
        self._steps      = [0] * n_trees
        self._tree_creds = [1.0] * n_trees          # inter-tree credibility weights
        self._last_preds: list[Any] = [None] * n_trees

    # ── helpers ───────────────────────────────────────────────────────────────

    def _tree_dist(self, tree: UniversalPredictor) -> dict[Any, float]:
        """Reconstruct normalised successor distribution from last predict() call."""
        contrib = tree._last_contributions
        if not contrib:
            return {}
        total = sum(w for w, _ in contrib.values()) or 1e-12
        d: dict[Any, float] = defaultdict(float)
        for w, succ in contrib.values():
            d[succ] += w / total
        return dict(d)

    def _mixture_vote(
        self,
        dists: list[dict[Any, float]],
        confs: list[float],
    ) -> tuple[Any, float]:
        """Confidence × credibility weighted sum across all trees."""
        votes: dict[Any, float] = defaultdict(float)
        total_w = sum(
            confs[i] * self._tree_creds[i]
            for i in range(self.n)
            if confs[i] > 0
        )
        if total_w < 1e-12:
            return None, 0.0
        for i, (d, c) in enumerate(zip(dists, confs)):
            if c > 0 and d:
                w = c * self._tree_creds[i] / total_w
                for v, p in d.items():
                    votes[v] += w * p
        if not votes:
            return None, 0.0
        best = max(votes, key=votes.get)
        return best, float(votes[best])

    def _product_vote(
        self,
        dists: list[dict[Any, float]],
        confs: list[float],
    ) -> tuple[Any, float]:
        """
        Weighted geometric mean of distributions.

        An outcome wins only when most trees agree on it.  Disagreement
        collapses to near-uniform → correctly low confidence on random data.
        Falls back to mixture if the product degenerates.
        """
        active = [
            (dists[i], self._tree_creds[i])
            for i in range(self.n)
            if confs[i] > 0 and dists[i]
        ]
        if not active:
            return None, 0.0

        vocab      = set().union(*(d.keys() for d, _ in active))
        total_cred = sum(cred for _, cred in active) or 1e-12

        product: dict[Any, float] = {}
        for v in vocab:
            log_p = sum(
                (cred / total_cred) * math.log(max(d.get(v, 1e-10), 1e-10))
                for d, cred in active
            )
            product[v] = math.exp(log_p)

        total = sum(product.values())
        if total < 1e-12:
            return self._mixture_vote(dists, confs)

        normed = {v: p / total for v, p in product.items()}
        best   = max(normed, key=normed.get)
        return best, float(normed[best])

    # ── public interface ──────────────────────────────────────────────────────

    def observe(self, value: Any) -> None:
        for tree in self.trees:
            tree.observe(value)

    def predict(self) -> tuple[Any, float]:
        dists: list[dict[Any, float]] = []
        confs: list[float]            = []

        for i, tree in enumerate(self.trees):
            pred, conf = tree.predict()
            self._last_preds[i] = pred
            dists.append(self._tree_dist(tree))
            confs.append(conf)

        if self.voting == 'product':
            return self._product_vote(dists, confs)
        return self._mixture_vote(dists, confs)

    def feedback(self, actual: Any) -> None:
        for i, (tree, rng) in enumerate(zip(self.trees, self._rngs)):
            self._steps[i] += 1

            if self._steps[i] <= self._offsets[i]:
                continue                   # staggered offset: not yet active

            if rng.random() < self.dropout:
                continue                   # dropout: skip this learning step

            tree.feedback(actual)

            # Inter-tree credibility update
            if self._last_preds[i] is not None:
                correct = self._last_preds[i] == actual
                factor  = 1.0 + self.tree_lr if correct else 1.0 - self.tree_lr
                self._tree_creds[i] = max(0.1, self._tree_creds[i] * factor)

        # Normalise to prevent runaway growth
        max_cred = max(self._tree_creds)
        if max_cred > 5.0:
            scale = 5.0 / max_cred
            self._tree_creds = [c * scale for c in self._tree_creds]

    # ── diagnostics ───────────────────────────────────────────────────────────

    def node_stats(self) -> dict:
        all_stats = [tree.node_stats() for tree in self.trees]
        result: dict = {}
        for key in ('total_nodes', 'observed', 'exploration',
                    'correction', 'coupling_links', 'allocator_trials'):
            result[key] = sum(s[key] for s in all_stats)
        for key in ('mean_coupling', 'lambda', 'optimizer_rolling_acc'):
            result[key] = sum(s[key] for s in all_stats) / self.n
        result['max_coupling']     = max(s['max_coupling']     for s in all_stats)
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
