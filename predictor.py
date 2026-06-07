import math
from collections import defaultdict
from typing import Any, Callable, Sequence

from components import Allocator, Optimizer

# Initial credibility granted to each node type.
# Correction and exploration nodes start louder so they immediately influence
# predictions in the situations they were created to address.
_INIT_CRED   = {'observed': 1.0, 'exploration': 2.0, 'correction': 3.0}
_MAX_CRED    = 8.0   # credibility ceiling — prevents runaway but allows real spread
_MAX_COUPLING = 1.0  # coupling magnitude ceiling


class UniversalPredictor:
    """
    Sequence predictor with dynamic topology, three-level weighting, and
    inter-unit coupling.

    Node types
    ----------
    observed    : one created per timestep from actual history
    exploration : extra node when current context has no good coverage (max_sim < ρ)
    correction  : extra node when the system was confident but wrong

    The network grows faster than 1-per-observation whenever those triggers fire,
    giving the topology the capacity the problem demands rather than a fixed budget.
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
    ):
        self.k            = context_length
        self.min_k        = max(1, min_context_length)
        self.sim          = similarity_fn
        self.coupling_ema = coupling_ema
        self.lr           = learning_rate
        self.coupling_lr  = coupling_lr
        self.lam          = feedback_strength   # λ — learned inter-unit scale
        self.vigilance    = vigilance           # ρ — novelty threshold

        self.history: list[Any] = []

        # Unified node store — each entry:
        #   [context: list, successor: Any, credibility: float, type: str]
        self._nodes: list[list] = []

        # Sparse directional coupling: (i, j) → float
        # coupling[(i,j)]: additive effect on j when i is present
        self.coupling: dict[tuple[int, int], float] = {}
        # Observation counts per pair — drives adaptive alpha
        self._coupling_counts: dict[tuple[int, int], int] = {}

        # Optimizer + Allocator
        self.optimizer = Optimizer(min_k=3, max_k=20, initial_k=10)
        self.allocator = Allocator(diversity_weight=0.4, coupling_weight=0.3)

        # State preserved from predict() for use in feedback()
        self._last_contributions: dict[int, tuple[float, Any]] = {}
        self._last_base: dict[int, float] = {}
        self._last_max_sim: float = 0.0
        self._last_prediction: Any = None
        self._last_confidence: float = 0.0
        self._last_context: list = []
        self._last_n_candidates: int = 0
        self._last_budget_used: int = 0
        self._last_fullscale_max_sim: float = 0.0

        self._quality_history: list[float] = []

    # ── public interface ──────────────────────────────────────────────────────

    def observe(self, value: Any) -> None:
        """Record a new value in the history sequence."""
        self.history.append(value)

    def predict(self) -> tuple[Any, float]:
        if len(self.history) < self.k:
            return None, 0.0

        context_full = list(self.history[-self.k:])
        self._last_context = context_full

        # Always compute full-scale max_sim for exploration/correction triggers
        fullscale_max_sim = 0.0
        for node in self._nodes:
            if len(node[0]) == self.k:
                s = self.sim(node[0], context_full)
                if s > fullscale_max_sim:
                    fullscale_max_sim = s
        self._last_fullscale_max_sim = fullscale_max_sim

        # Select the longest scale with sufficient coverage; fall back to min_k
        all_candidates: dict[int, tuple[float, Any]] = {}
        max_sim = 0.0

        for length in range(self.k, self.min_k - 1, -1):
            context = context_full[-length:]
            candidates: dict[int, tuple[float, Any]] = {}
            length_max_sim = 0.0

            for i, node in enumerate(self._nodes):
                ctx, succ, cred, _ = node
                if len(ctx) != length:
                    continue
                s = self.sim(ctx, context)
                if s > length_max_sim:
                    length_max_sim = s
                w = s * cred
                if w > 1e-12:
                    candidates[i] = (w, succ)

            if length_max_sim >= self.vigilance or length == self.min_k:
                all_candidates = candidates
                max_sim = length_max_sim
                break

        self._last_max_sim      = max_sim
        self._last_n_candidates = len(all_candidates)

        if not all_candidates:
            self._last_prediction    = None
            self._last_confidence    = 0.0
            self._last_contributions = {}
            self._last_base          = {}
            self._last_budget_used   = 0
            return None, 0.0

        # Optimizer decides how many nodes speak
        quality_now = self._quality_history[-1] if self._quality_history else 1.0
        budget      = self.optimizer.get_budget(max_sim, quality_now)

        # Allocator decides which nodes speak
        base = self.allocator.select(all_candidates, self.coupling, budget)
        self._last_base        = {i: w for i, (w, _) in base.items()}
        self._last_budget_used = len(base)

        # Inter-unit communication via learned coupling
        # coupling[(j, i)] = effect on i when j is present (directional)
        comm: dict[int, float] = {}
        for i, (w, _) in base.items():
            lateral = sum(
                self.coupling.get((j, i), 0.0) * base[j][0]
                for j in base if j != i
            )
            comm[i] = max(1e-12, w + self.lam * lateral)

        self._last_contributions = {i: (comm[i], self._nodes[i][1]) for i in comm}

        total_w   = sum(comm.values())
        total_wsq = sum(v * v for v in comm.values())
        weights: dict[Any, float] = defaultdict(float)
        for i, w in comm.items():
            weights[self._nodes[i][1]] += w

        dist       = {v: w / total_w for v, w in weights.items()}
        prediction = max(dist, key=dist.get)
        eff_n      = (total_w ** 2) / total_wsq if total_wsq > 0 else 1.0
        confidence = _confidence(dist, eff_n)

        self._last_prediction = prediction
        self._last_confidence = confidence
        return prediction, confidence

    def feedback(self, actual: Any) -> None:
        # ── 1. Create observed nodes at all context lengths ───────────────────
        for length in range(self.min_k, self.k + 1):
            if len(self.history) >= length:
                self._add_node(list(self.history[-length:]), actual, 'observed')

        # ── 2. Update credibility (peer-independent) ──────────────────────────
        if self._last_contributions:
            # Recompute raw similarities once — clean, no drift from in-loop updates
            raw_sims: dict[int, float] = {
                i: self.sim(self._nodes[i][0], self._last_context[-len(self._nodes[i][0]):])
                for i in self._last_contributions
                if i < len(self._nodes)
            }

            for i, (w, succ) in self._last_contributions.items():
                sim_i = raw_sims.get(i, 0.0)
                if succ == actual:
                    self._nodes[i][2] *= 1.0 + self.lr * sim_i
                else:
                    self._nodes[i][2] *= 1.0 - self.lr * sim_i
                self._nodes[i][2] = max(0.01, self._nodes[i][2])

            # Cap — preserves relative differences, allows real spread
            max_c = max(n[2] for n in self._nodes)
            if max_c > _MAX_CRED:
                scale = _MAX_CRED / max_c
                for node in self._nodes:
                    node[2] *= scale

            # ── 3. Update coupling (directional) ─────────────────────────────
            # coupling[(i, j)]: when i is present, additive effect on j.
            units = list(self._last_contributions.items())
            if self.coupling_ema:
                # Adaptive-alpha EMA: alpha = max(floor, 1/(1+n)) per pair.
                # Fast early (high uncertainty), stable later (earned trust),
                # floor keeps coupling live for concept drift.
                _FLOOR = 0.05
                for a, (i, (w_i, s_i)) in enumerate(units):
                    for j, (w_j, s_j) in units[a + 1:]:
                        ci, cj = s_i == actual, s_j == actual

                        updates: list[tuple[tuple, float]] = []
                        if ci and cj:
                            updates = [((i, j), 1.0), ((j, i), 1.0)]
                        elif not ci and not cj and s_i == s_j:
                            updates = [((i, j), -1.0), ((j, i), -1.0)]
                        elif ci and not cj:
                            updates = [((i, j), -1.0)]
                        elif cj and not ci:
                            updates = [((j, i), -1.0)]

                        for key, sig in updates:
                            n     = self._coupling_counts.get(key, 0)
                            alpha = max(_FLOOR, self.coupling_lr / (1 + n * self.coupling_lr))
                            self.coupling[key] = (1-alpha)*self.coupling.get(key, 0.0) + alpha*sig
                            self._coupling_counts[key] = n + 1

                        for key in ((i, j), (j, i)):
                            if key in self.coupling and abs(self.coupling[key]) < 1e-6:
                                del self.coupling[key]
            else:
                # Accumulative directional coupling (previous method)
                total_w = sum(w for w, _ in self._last_contributions.values()) or 1e-12
                for a, (i, (w_i, s_i)) in enumerate(units):
                    for j, (w_j, s_j) in units[a + 1:]:
                        ci, cj   = s_i == actual, s_j == actual
                        strength = (w_i / total_w) * (w_j / total_w)
                        if ci and cj:
                            for key in ((i, j), (j, i)):
                                self.coupling[key] = self.coupling.get(key, 0.0) + self.coupling_lr * strength
                        elif not ci and not cj and s_i == s_j:
                            for key in ((i, j), (j, i)):
                                self.coupling[key] = self.coupling.get(key, 0.0) - self.coupling_lr * strength
                        elif ci and not cj:
                            key = (i, j)
                            self.coupling[key] = self.coupling.get(key, 0.0) - self.coupling_lr * strength
                        elif cj and not ci:
                            key = (j, i)
                            self.coupling[key] = self.coupling.get(key, 0.0) - self.coupling_lr * strength
                        for key in ((i, j), (j, i)):
                            if key in self.coupling:
                                v = self.coupling[key]
                                if abs(v) > _MAX_COUPLING:
                                    self.coupling[key] = math.copysign(_MAX_COUPLING, v)
                                elif abs(v) < 1e-6:
                                    del self.coupling[key]

            # ── 4. Update λ ───────────────────────────────────────────────────
            if self.coupling and self._last_base:
                comm_correct = comm_abs = 0.0
                for i in self._last_base:
                    lateral = sum(
                        self.coupling.get((j, i), 0.0) * self._last_base[j]
                        for j in self._last_base if j != i
                    )
                    if i < len(self._nodes) and self._nodes[i][1] == actual:
                        comm_correct += lateral
                    comm_abs += abs(lateral)
                if comm_abs > 1e-10:
                    self.lam = float(min(1.0, max(0.0,
                        self.lam + 0.005 * (comm_correct / comm_abs))))

        # ── 5. Dynamic node creation ─────────────────────────────────────────
        if self._last_context and len(self._last_context) == self.k:

            # Trigger 2 — exploration: full-scale context has no good coverage
            if self._last_fullscale_max_sim < self.vigilance:
                self._add_node(self._last_context, actual, 'exploration')

            # Trigger 3 — correction: system was confident but wrong
            elif (self._last_confidence > 0.6
                  and self._last_prediction is not None
                  and self._last_prediction != actual):
                self._add_node(self._last_context, actual, 'correction')

        # ── 7. Update optimizer and allocator ────────────────────────────────
        was_correct = (self._last_prediction == actual)
        self.optimizer.update(was_correct, self._last_n_candidates, self._last_budget_used)
        self.allocator.update(list(self._last_contributions.keys()), was_correct)

        self._quality_history.append(self.similarity_quality())

    # ── internal ──────────────────────────────────────────────────────────────

    def _add_node(self, context: list, successor: Any, node_type: str) -> None:
        """Add a node only if no same-type, same-length node with near-identical context exists."""
        for node in self._nodes:
            if (node[3] == node_type
                    and len(node[0]) == len(context)
                    and self.sim(node[0], context) > 0.99):
                return
        self._nodes.append([list(context), successor,
                            _INIT_CRED[node_type], node_type])

    # ── diagnostics ───────────────────────────────────────────────────────────

    def similarity_quality(self) -> float:
        creds = [n[2] for n in self._nodes]
        if not creds:
            return 0.0
        sc    = sorted(creds, reverse=True)
        top_n = max(1, len(sc) // 4)
        return sum(sc[:top_n]) / top_n

    def node_stats(self) -> dict:
        counts: dict[str, int] = defaultdict(int)
        for node in self._nodes:
            counts[node[3]] += 1
        vals = list(self.coupling.values())
        return {
            'total_nodes':    len(self._nodes),
            'observed':       counts['observed'],
            'exploration':    counts['exploration'],
            'correction':     counts['correction'],
            'coupling_links': len(vals),
            'mean_coupling':  sum(abs(v) for v in vals) / len(vals) if vals else 0.0,
            'max_coupling':   max(abs(v) for v in vals) if vals else 0.0,
            'lambda':         self.lam,
            'optimizer_budget':      self.optimizer.budget,
            'optimizer_rolling_acc': self.optimizer.rolling_accuracy,
            'allocator_trials':      sum(self.allocator._node_trials.values()),
        }

    def convergence_state(self) -> dict:
        qh          = self._quality_history
        quality_now = qh[-1] if qh else 0.0
        if len(qh) < 6:
            return {'plateau': None, 'tau': None, 'quality_now': quality_now,
                    'steps_to_95pct': None, 'converged': False}
        L, tau      = _fit_convergence(qh)
        n           = len(qh)
        n_95        = int(tau * math.log(20.0)) if math.isfinite(tau) and tau > 0 else None
        steps       = max(0, n_95 - n) if n_95 is not None else None
        return {
            'plateau':        L,
            'tau':            tau,
            'quality_now':    quality_now,
            'steps_to_95pct': steps,
            'converged':      steps is not None and steps == 0,
        }

    def lookahead_quality(self, n_steps: int) -> float:
        state = self.convergence_state()
        L, tau = state['plateau'], state['tau']
        if L is None or tau is None or not math.isfinite(tau) or tau <= 0:
            return state['quality_now']
        return L * (1.0 - math.exp(-(len(self._quality_history) + n_steps) / tau))


# ── module-level helpers ──────────────────────────────────────────────────────

def _fit_convergence(qh: list[float]) -> tuple[float, float]:
    deltas = [qh[i] - qh[i - 1] for i in range(1, len(qh))]
    prev   = qh[:-1]
    n      = len(deltas)
    xbar   = sum(prev) / n
    ybar   = sum(deltas) / n
    num    = sum((x - xbar) * (y - ybar) for x, y in zip(prev, deltas))
    den    = sum((x - xbar) ** 2 for x in prev)
    if abs(den) < 1e-12 or abs(num) < 1e-12:
        return qh[-1], float('inf')
    slope = num / den
    intcp = ybar - slope * xbar
    if slope >= 0 or abs(slope) < 1e-6:
        return qh[-1], float('inf')
    alpha = -slope
    L     = -intcp / slope
    tau   = (-1.0 / math.log(1.0 - alpha)) if 0 < alpha < 1 else 1.0 / alpha
    return max(L, qh[-1]), max(tau, 1.0)


def _confidence(dist: dict[Any, float], effective_n: float) -> float:
    n_vals      = len(dist)
    size_factor = 1.0 - 1.0 / math.sqrt(effective_n + 1.0)
    if n_vals <= 1:
        return size_factor
    entropy    = -sum(p * math.log(p + 1e-12) for p in dist.values())
    peakedness = 1.0 - entropy / math.log(n_vals)
    return peakedness * size_factor
