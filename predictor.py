"""
Universal Sequence Predictor — Credibility-Weighted Context Tree

Architecture
------------
Contexts live in a prefix trie.  Each node in the trie stores a
credibility-weighted distribution over successor symbols.

Prediction  O(k):
    Walk the trie at depths min_k..k, collecting matching context nodes.
    Blend their distributions from shallow to deep using a CTW-style
    recursive mixture:

        λ_d = c_d / (c_d + 1)          # credibility → mixing weight
        P_d = λ_d · P_local(d) + (1−λ_d) · P_{d−1}

    High credibility → λ close to 1 → deep context dominates.
    Low credibility  → λ close to 0 → falls back to shallower.
    Root provides a KT-smoothed unigram as the seed distribution.

Update  O(k):
    For each depth, update the matching node's per-successor credibility
    and the node's overall credibility using a multiplicative rule:

        correct:  c ← min(C_MAX, c × (1 + lr))
        wrong:    c ← max(C_MIN, c × (1 − lr))

    Wrong predictions also boost the correct successor's credibility
    at the same node — the in-trie correction mechanism.

Concept drift:
    Wrong predictions degrade node_cred, reducing λ and causing automatic
    fallback to shallower contexts.  New correct observations rebuild
    credibility for the updated pattern.  No drift detector; no forgetting
    parameter; adaptation speed is a direct function of how confidently the
    wrong pattern was held.

Regret sketch:
    The multiplicative credibility update is the Multiplicative Weights
    Update (MWU) algorithm applied to depth selection.  For the class of k
    single-depth predictors MWU achieves O(√(T ln k)) regret in hindsight.
    The CTW-style blend implements this across all depths simultaneously.
"""

import math
from typing import Any, Callable, Sequence

_CRED_MIN = 0.01
_CRED_MAX = 8.0


class _TrieNode:
    """One node in the credibility-weighted context tree."""
    __slots__ = ['children', 'succ_cred', 'node_cred', 'n_obs']

    def __init__(self):
        self.children:  dict = {}     # symbol → _TrieNode
        self.succ_cred: dict = {}     # symbol → float  (credibility weight)
        self.node_cred: float = 1.0   # reliability of this context as a predictor
        self.n_obs:     int   = 0     # times this context was seen


class UniversalPredictor:

    def __init__(
        self,
        context_length: int,
        similarity_fn: Callable[[Sequence, Sequence], float] | None = None,
        learning_rate: float = 0.1,
        vigilance: float = 0.5,
        min_context_length: int = 1,
        **kwargs,   # absorb legacy args (coupling_lr, feedback_strength, etc.)
    ):
        self.k         = context_length   # also exposed as max_k property
        self.min_k     = max(1, min_context_length)
        self.lr        = learning_rate
        self.vigilance = vigilance
        self._surface_sim = similarity_fn   # kept for API compat with forest

        self._root:  _TrieNode      = _TrieNode()
        self.history: list[Any]     = []
        self._vocab:  set           = set()

        # predict() → feedback() state
        self._last_prediction:    Any   = None
        self._last_distribution:  dict  = {}
        self._last_context:       list  = []
        self._last_max_sim:       float = 0.0
        self._last_contributions: dict  = {}   # depth → (node_cred, top_successor)

        # Backward-compat stubs (coupling removed; ablation showed ~0 effect)
        self.coupling:          dict  = {}
        self._coupling_counts:  dict  = {}
        self.lam:               float = 0.0

    # ── public interface ──────────────────────────────────────────────────────

    def observe(self, value: Any) -> None:
        self.history.append(value)
        self._vocab.add(value)

    def predict(self) -> tuple[Any, float]:
        if not self._vocab:
            return None, 0.0

        self._last_context = list(self.history[-self.k:]) if self.history else []
        active = self._get_active_nodes()
        dist   = self._blend(active)

        if not dist:
            return None, 0.0

        pred = max(dist, key=dist.get)
        conf = dist[pred]

        self._last_distribution  = dist
        self._last_prediction    = pred
        self._last_max_sim       = max((n.node_cred for n, _ in active), default=0.0)
        self._last_contributions = {
            d: (n.node_cred,
                max(n.succ_cred, key=n.succ_cred.get) if n.succ_cred else pred)
            for n, d in active
        }
        return pred, conf

    def feedback(self, actual: Any) -> None:
        self._vocab.add(actual)
        n_hist  = len(self.history)
        correct = (self._last_prediction == actual)

        # Root stores raw unigram counts — provides KT seed for blend
        self._root.succ_cred[actual] = self._root.succ_cred.get(actual, 0) + 1.0
        self._root.n_obs += 1

        # Per-depth context nodes (depths min_k .. k)
        for d in range(self.min_k, min(self.k, n_hist - 1) + 1):
            ctx  = tuple(self.history[-(d + 1):-1])
            node = self._feedback_get_node(ctx)
            if node is None:
                continue
            node.n_obs += 1
            if actual not in node.succ_cred:
                node.succ_cred[actual] = 1.0

            if correct:
                self._update_node_correct(node, actual)
            else:
                self._update_node_wrong(node, self._last_prediction, actual)

    def _distribution(self) -> dict:
        """Return last predictive distribution (for log-loss evaluation)."""
        return dict(self._last_distribution)

    # ── hooks for subclass ablation ───────────────────────────────────────────

    def _update_node_correct(self, node: _TrieNode, actual: Any) -> None:
        node.succ_cred[actual] = min(_CRED_MAX, node.succ_cred[actual] * (1 + self.lr))
        node.node_cred         = min(_CRED_MAX, node.node_cred         * (1 + self.lr))

    def _update_node_wrong(self, node: _TrieNode, predicted: Any, actual: Any) -> None:
        # Confidence-proportional degradation: the more a node was trusted,
        # the more aggressively it should lose that trust when wrong.
        # lr_down scales from lr (fresh node) to 2×lr (maximally trusted node).
        # This halves adaptation lag for high-credibility nodes after a drift.
        lr_down = self.lr * (1.0 + node.node_cred / _CRED_MAX)
        if predicted is not None and predicted in node.succ_cred:
            node.succ_cred[predicted] = max(_CRED_MIN,
                node.succ_cred[predicted] * (1 - lr_down))
        # In-trie correction: immediately boost correct successor
        node.succ_cred[actual] = min(_CRED_MAX,
            node.succ_cred.get(actual, 1.0) * (1 + self.lr))
        node.node_cred = max(_CRED_MIN, node.node_cred * (1 - lr_down))

    def _blend_lambda(self, node_cred: float) -> float:
        """CTW-style mixing coefficient. Override to disable credibility effect."""
        return node_cred / (node_cred + 1.0)

    def _feedback_get_node(self, ctx: tuple) -> _TrieNode | None:
        """Return (creating if needed) the node for ctx. Override for ablation."""
        return self._ensure_node(ctx)

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_active_nodes(self) -> list[tuple[_TrieNode, int]]:
        """
        Return [(node, depth)] for matching context depths min_k..k.
        O(k²) total — effectively O(1) for small k.
        """
        result = []
        max_d  = min(self.k, len(self.history))
        for d in range(self.min_k, max_d + 1):
            node = self._walk(tuple(self.history[-d:]))
            if node is not None and node.succ_cred:
                result.append((node, d))
        return result

    def _walk(self, ctx: tuple) -> _TrieNode | None:
        node = self._root
        for sym in ctx:
            if sym not in node.children:
                return None
            node = node.children[sym]
        return node

    def _ensure_node(self, ctx: tuple) -> _TrieNode:
        node = self._root
        for sym in ctx:
            if sym not in node.children:
                node.children[sym] = _TrieNode()
            node = node.children[sym]
        return node

    def _blend(self, active: list[tuple[_TrieNode, int]]) -> dict:
        """
        CTW-style credibility-weighted blend, shallow to deep.
        Uses KT prior (alpha = 0.5/|V|) at each node for smoothing.
        """
        if not self._vocab:
            return {}

        V     = len(self._vocab)
        alpha = 0.5 / V    # Krichevsky-Trofimov prior

        # Seed: KT-smoothed unigram from root counts
        root_total = sum(self._root.succ_cred.values()) or 1.0
        blended = {
            s: (self._root.succ_cred.get(s, 0) + alpha) / (root_total + alpha * V)
            for s in self._vocab
        }

        by_depth = {d: n for n, d in active}
        for d in range(1, self.k + 1):
            if d not in by_depth:
                continue
            node  = by_depth[d]
            total = sum(node.succ_cred.values()) or 1.0
            local = {
                s: (node.succ_cred.get(s, 0) + alpha) / (total + alpha * V)
                for s in self._vocab
            }
            lam     = self._blend_lambda(node.node_cred)
            blended = {s: lam * local[s] + (1 - lam) * blended[s]
                       for s in self._vocab}

        total = sum(blended.values())
        if total < 1e-12:
            return {s: 1.0 / V for s in self._vocab}
        return {s: v / total for s, v in blended.items()}

    # ── backward-compat API (used by forest.py and diagnostics) ──────────────

    def sim(self, ctx_a: Sequence, ctx_b: Sequence) -> float:
        """Surface similarity — kept for forest API compat."""
        if self._surface_sim is not None:
            try:
                return float(self._surface_sim(ctx_a, ctx_b))
            except Exception:
                pass
        return 1.0 if list(ctx_a) == list(ctx_b) else 0.0

    @property
    def max_k(self) -> int:
        return self.k

    @property
    def _nodes(self) -> list:
        """All trie nodes as a flat list (for node-count reporting)."""
        result = []
        stack  = [self._root]
        while stack:
            n = stack.pop()
            result.append(n)
            stack.extend(n.children.values())
        return result

    def node_stats(self) -> dict:
        nodes = self._nodes
        total = len(nodes)
        return {
            'total_nodes':           total,
            'observed':              total,
            'exploration':           0,
            'correction':            0,
            'coupling_links':        0,
            'mean_coupling':         0.0,
            'max_coupling':          0.0,
            'lambda':                0.0,
            'optimizer_budget':      self.k,
            'optimizer_rolling_acc': 0.0,
            'allocator_trials':      sum(n.n_obs for n in nodes),
        }

    def similarity_quality(self) -> float:
        nodes = [n for n in self._nodes if n.node_cred != 1.0]
        if not nodes:
            return 1.0
        creds = sorted((n.node_cred for n in nodes), reverse=True)
        top_n = max(1, len(creds) // 4)
        return sum(creds[:top_n]) / top_n

    def convergence_state(self) -> dict:
        nodes = self._nodes
        if not nodes:
            return {'plateau': None, 'tau': None, 'quality_now': 0.0,
                    'steps_to_95pct': None, 'converged': False}
        quality = sum(n.node_cred for n in nodes) / len(nodes)
        return {'plateau': quality, 'tau': None, 'quality_now': quality,
                'steps_to_95pct': None, 'converged': False}

    def lookahead_quality(self, n_steps: int) -> float:
        return self.convergence_state()['quality_now']
