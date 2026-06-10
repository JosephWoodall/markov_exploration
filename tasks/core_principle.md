# This Repo's North Star

## The Principle (one sentence)

**Credibility is earned through prediction accuracy and lost through confident error — and this single principle, applied multiplicatively at every scale of context simultaneously, produces online sequence prediction with automatic concept-drift adaptation, theoretically grounded regret bounds, and no forgetting hyperparameter.**

---

## Why This Feels Magically Correct

Every existing approach to sequence prediction has a forgetting problem. Count-based methods (N-gram, PPM-D) only accumulate — they cannot lose confidence in a stale pattern, so concept drift requires an external reset or sliding window. CTW is theoretically optimal under NML but was derived for stationary sources; its Bayesian weights have no mechanism to down-weight contexts that have become unreliable.

This architecture solves the problem at the representation level, not the algorithm level. The credibility score *is* the context's relevance weight, *is* the blending coefficient, *is* the signal for how aggressively to update. One variable does three jobs, and they are self-consistent: a context that predicts well gets used more (higher λ) and degrades more slowly (lower lr_down); a context that predicts badly gets used less and degrades faster. The system automatically allocates attention to what is currently working.

The confidence-proportional degradation rule (`lr_down = lr × (1 + c/C_MAX)`) is not a heuristic — it follows directly from the information-theoretic argument that a node holding high confidence has committed more strongly to a prediction and therefore delivers more evidence of error when wrong. The magnitude of belief update should scale with the strength of the prior belief.

---

## State-of-the-Art Grounding

**CTW (Context Tree Weighting):** Willems, Shtarkov, Tjalkens (1995). Provably achieves the NML (Normalized Maximum Likelihood) minimax redundancy for stationary binary sources. Our blend is CTW-style but replaces the Bayesian Dirichlet weights with credibility scores that can decrease — the key departure that enables nonstationarity.

**MWU (Multiplicative Weights Update):** Arora, Hazan, Kale (2012, "The Multiplicative Weights Update Method: a Meta-Algorithm and Applications"). The multiplicative credibility update is a direct instance of MWU applied to depth selection. MWU achieves O(√(T ln k)) regret over k experts (depths). Our CTW-style blend runs MWU across all depths simultaneously rather than selecting one.

**Krichevsky-Trofimov estimator:** KT (1981). The alpha = 0.5/|V| smoothing used at each trie node achieves asymptotic optimality for i.i.d. sources and is the correct Bayesian prior under a symmetric Dirichlet. We use KT smoothing on the raw succ_cred values for the local distributions.

**Concept drift without detectors:** Losing, Hammer, Wersing (2016, "Incremental on-line learning: A review and comparison of state-of-the-art algorithms") survey the standard approaches (DDM, ADWIN, Page-Hinkley). All require a separate drift detector module. Our approach is the only one where drift adaptation is a consequence of the prediction update rule itself, not a bolt-on.

---

## Alternatives and Why This Wins

**PPM-D (Prediction by Partial Match with escape):**
Counts are monotonically increasing. When the generative process changes, the old counts dominate new observations for O(count) steps. On the concept-drift benchmarks, PPM-D drops to ~2–5% accuracy on reversal tasks (worse than random) and stays there indefinitely. No architectural fix is possible without adding a separate forgetting mechanism — which then requires tuning.

**CTW with fixed Dirichlet weights:**
Theoretically optimal for stationary sources, but the Dirichlet conjugate update only adds counts — it cannot subtract. Under nonstationarity, the theoretical guarantee does not hold, and empirically CTW performs identically to N-gram on concept-drift tasks (~4–6% on reversal). Adding a forgetting window converts it to Switching CTW (Willems 1998), which helps but requires specifying the switching rate in advance.

**LSTM / Transformer:**
Both can handle nonstationarity through gradient updates, but require: fixed vocabulary at training time, a training phase separate from deployment, O(n×d) parameter updates per step, and a learning rate schedule. The trie-based predictor works on any hashable symbol, has no separate training phase, updates in O(k) per step, and has one interpretable hyperparameter (lr). For online deployment on novel symbol streams, the overhead of neural architectures is unjustifiable.

---

## The North Star Applied to Every Decision

Every design choice must be evaluated against this question: **does it make the credibility signal more accurate, faster-updating, or better-calibrated?**

- The CTW blend uses credibility as λ directly. ✓
- Confidence-proportional degradation scales lr_down by the prior credibility. ✓
- The Forest's inter-tree weights apply the same credibility principle at the ensemble level. ✓
- Module 2's retrieval strategy uses successor-distribution similarity — which is the trie's accumulated credibility evidence applied as a semantic search. ✓
- Any change that adds a static parameter (forgetting rate, threshold, window size) is prima facie suspect: it should be replaced by a mechanism that learns the parameter from the credibility signal itself.
