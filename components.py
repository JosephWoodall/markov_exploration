"""
Optimizer  — learns the right activation budget (adaptive k) per prediction.
Allocator  — given that budget, selects which nodes fill it using four criteria:
             similarity × credibility (base), diversity, and coupling bonus.
"""

import math
from collections import defaultdict
from typing import Any


class Optimizer:
    """
    Learns the minimum budget that achieves good predictions.

    Grows when constrained AND wrong  (budget was too small to see the answer).
    Shrinks when unconstrained AND right (budget had slack we didn't need).
    Also scales the immediate budget for a given query by its difficulty:
    harder queries (low max_sim, low model quality) get a larger slice.
    """

    def __init__(self, min_k: int = 3, max_k: int = 20, initial_k: int = 10):
        self.min_k   = min_k
        self.max_k   = max_k
        self.budget  = initial_k    # learned ceiling
        self._acc_ema = 0.5         # rolling accuracy (EMA)
        self._alpha   = 0.08

    def get_budget(self, max_sim: float, quality: float) -> int:
        """
        Scale this query's budget by difficulty.
        Difficult query (low sim, low quality) gets the full learned ceiling.
        Easy query gets closer to min_k.
        """
        sim_uncertainty     = 1.0 - max(0.0, min(1.0, max_sim))
        quality_uncertainty = 1.0 - max(0.0, min(1.0, quality / 2.0))
        difficulty = 0.65 * sim_uncertainty + 0.35 * quality_uncertainty
        raw = self.min_k + difficulty * (self.budget - self.min_k)
        return max(self.min_k, min(self.budget, int(round(raw))))

    def update(self, correct: bool, n_candidates: int, budget_used: int) -> None:
        self._acc_ema = self._alpha * float(correct) + (1 - self._alpha) * self._acc_ema
        constrained   = n_candidates > budget_used

        if not correct and constrained:
            # budget too small — couldn't reach potentially useful nodes
            self.budget = min(self.max_k, self.budget + 1)
        elif correct and not constrained and self.budget > self.min_k:
            # budget had slack and succeeded — can afford to be leaner
            self.budget = max(self.min_k, self.budget - 1)

    @property
    def rolling_accuracy(self) -> float:
        return self._acc_ema


class Allocator:
    """
    Greedy node selection using four criteria weighted together:

      base       = similarity × credibility   (who is close and trustworthy?)
      diversity  = penalise repeating an already-covered outcome (new perspectives)
      coupling   = bonus for nodes coupled to already-selected ones (signal amplification)

    After each prediction, updates a per-node selection success rate so the
    allocator gradually learns which nodes are worth selecting beyond raw similarity.
    """

    def __init__(
        self,
        diversity_weight: float = 0.4,
        coupling_weight:  float = 0.3,
    ):
        self.dw = diversity_weight
        self.cw = coupling_weight
        self._node_hits:   dict[int, int] = defaultdict(int)   # times selected + correct
        self._node_trials: dict[int, int] = defaultdict(int)   # times selected

    def select(
        self,
        candidates: dict[int, tuple[float, Any]],
        coupling:   dict[tuple[int, int], float],
        budget:     int,
    ) -> dict[int, tuple[float, Any]]:
        """
        Return at most `budget` nodes from candidates.
        candidates: {node_idx: (base_weight, successor)}
        """
        if not candidates:
            return {}
        budget    = min(budget, len(candidates))
        remaining = dict(candidates)
        selected: dict[int, tuple[float, Any]] = {}

        for _ in range(budget):
            if not remaining:
                break

            # Outcome coverage of already-selected set
            sel_total = sum(w for w, _ in selected.values()) or 1e-12
            coverage: dict[Any, float] = defaultdict(float)
            for w, succ in selected.values():
                coverage[succ] += w / sel_total

            best_idx, best_score = None, -1e12

            for idx, (base_w, succ) in remaining.items():
                # Diversity penalty — proportional to how saturated this outcome is
                div_penalty = self.dw * coverage.get(succ, 0.0) * base_w

                # Coupling bonus — effect of already-selected nodes on this candidate
                coup_bonus = self.cw * sum(
                    coupling.get((si, idx), 0.0)
                    for si in selected
                )

                score = base_w - div_penalty + coup_bonus
                if score > best_score:
                    best_score = score
                    best_idx   = idx

            if best_idx is not None:
                selected[best_idx] = remaining.pop(best_idx)

        return selected

    def update(self, selected_indices: list[int], correct: bool) -> None:
        for idx in selected_indices:
            self._node_trials[idx] += 1
            if correct:
                self._node_hits[idx] += 1

    def selection_quality(self, idx: int) -> float:
        """Track record of node idx when selected. 0.5 = unknown."""
        t = self._node_trials.get(idx, 0)
        return self._node_hits.get(idx, 0) / t if t > 0 else 0.5

    @staticmethod
    def _coupling_val(
        coupling: dict[tuple[int, int], float], i: int, j: int
    ) -> float:
        return coupling.get((min(i, j), max(i, j)), 0.0)
