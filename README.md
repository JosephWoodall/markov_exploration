# Universal Sequence Predictor

Online, instance-based sequence prediction. Given any stream of observations, it learns to predict what comes next — for any symbol type, in any domain — without assuming a fixed distribution, a known alphabet, or a stationary process.

---

## The Core Idea

The algorithm keeps a trie of contexts. Every node in the trie stores two things: a credibility-weighted distribution over successor symbols, and a track record of how reliable that context has been as a predictor. When predicting, it blends the distributions from shallow (general) to deep (specific), where each depth's influence is proportional to its track record. When updating after a wrong prediction, a node that was confidently wrong loses trust faster than one that was fresh and uncertain.

That is the entire algorithm. No drift detector, no forgetting parameter, no domain-specific tuning.

---

## Architecture

### Module 1 — Universal Sequence Predictor (`predictor.py`)

**Data structure:** a prefix trie. Each `_TrieNode` stores:
- `succ_cred` — credibility weight per successor symbol
- `node_cred` — reliability of this context as a predictor overall
- `n_obs` — number of times this context has been seen

The root holds raw unigram counts and provides a Krichevsky-Trofimov smoothed prior.

**Prediction O(k):**

Walk the trie at depths `min_k..k`. For each matching node, compute a KT-smoothed local distribution. Blend from shallow to deep using the CTW-style recursive formula:

```
λ_d = node_cred_d / (node_cred_d + 1)
P_d = λ_d · P_local(d)  +  (1 − λ_d) · P_{d−1}
```

High credibility → λ → 1 → deep context dominates.
Low credibility  → λ → 0 → falls back to shallow.
Root provides the seed.

**Update O(k):**

For each depth, find the context node and apply a multiplicative rule:

```
correct:  node_cred ← min(C_MAX, node_cred × (1 + lr))
          succ_cred[actual] ← min(C_MAX, succ_cred[actual] × (1 + lr))

wrong:    lr_down = lr × (1 + node_cred / C_MAX)   # confidence-proportional
          node_cred ← max(C_MIN, node_cred × (1 − lr_down))
          succ_cred[wrong] ← max(C_MIN, succ_cred[wrong] × (1 − lr_down))
          succ_cred[actual] ← min(C_MAX, succ_cred[actual] × (1 + lr))
```

The `lr_down` scaling is the key drift-adaptation mechanism: a node that was highly trusted when it turned wrong loses credibility up to 2× faster than a fresh node. This halves the adaptation lag after a concept drift without requiring any drift detector.

**Concept drift:**

Wrong predictions degrade `node_cred`, reducing λ at that depth, causing the blend to automatically fall back to shallower (more general) contexts. As the new pattern accumulates correct observations, `node_cred` rebuilds at the updated depth. No explicit change detection; no forgetting window; adaptation speed is a function of how confidently the old pattern was held.

**Regret bound:**

The multiplicative credibility update is an instance of the Multiplicative Weights Update (MWU) algorithm applied to depth selection. For a class of k single-depth predictors, MWU achieves O(√(T ln k)) regret. The CTW-style blend runs this across all depths simultaneously.

---

### Module 1 — Forest Ensemble (`forest.py`)

`PredictorForest` is a collection of `UniversalPredictor` instances that start identical and diverge through experience. Diversity comes from four sources:

| Mechanism | How it works |
|---|---|
| Heterogeneous k | Each tree uses a different context length: k, k+1, k+2, … capturing different temporal scales |
| Feedback dropout | Each tree independently skips learning on each step with probability `dropout` — the sequence analogue of bagging |
| Staggered offsets | Tree i doesn't start learning until step `i × stagger`; early topology has outsized influence on later structure |
| Inter-tree credibility | Each tree maintains a persistent weight updated by whether it was right; correct trees speak louder on the next prediction |

**Voting:** product-of-experts (weighted geometric mean of per-tree distributions). An outcome one tree assigns near-zero probability is suppressed even if other trees favour it — this collapses toward uniform when trees disagree, which is the correct response to uncertainty. Mixture voting is available as a fallback for sparse data.

---

### Module 2 — Goal-Directed Generation (`module2.py`)

Module 1 is **intuition** — fast, associative, pattern-matching.
Module 2 is **deliberation** — goal-directed, using Module 1 as a world model.

Module 1 is already generative: call `predict()` autoregressively and it produces continuations. Module 2 adds **steering**: constraining or guiding that generation toward a target.

**Training format:** represent Q&A or any prompt→response task as a flat sequence:
```
[prompt tokens ...] [SEPARATOR] [response tokens ...] [END]
```
Module 1 learns that SEPARATOR is followed by responses, not more prompts. No architectural changes needed.

**Three generation strategies** (all implemented in `module2.py`):

| Strategy | Mechanism | Best for |
|---|---|---|
| **Autoregressive** | Feed `[prompt + SEPARATOR]` as context seed; generate token by token until END | Direct completion, short responses |
| **Beam search** | Maintain N candidate sequences; at each step expand by all vocabulary tokens; prune to top N by cumulative log-probability | Longer responses, controllable diversity |
| **Retrieval** | Find the most similar prompt in training data by successor-distribution similarity; return its stored response | Factual lookup, repeated-query scenarios |

---

## Benchmark Results

Evaluated on 6 standard datasets and 4 concept-drift streams. All methods use the same context length k and train/test split (80/20). Baselines: Persistence (predict last seen), Majority, N-gram(5), PPM-D(5), CTW(5).

**Standard benchmarks (test accuracy %):**

| Dataset | Persistence | PPM-D | CTW | **Predictor** | **Forest** |
|---|---|---|---|---|---|
| Airline passengers | **37.9** | 27.6 | 31.0 | **37.9** | 29.6 |
| Alice in Wonderland | 3.7 | 37.3 | **38.7** | 35.0 | 34.7 |
| DNA sequences | 28.7 | 30.7 | 27.3 | **33.0** | 28.3 |
| Weather | **48.0** | 46.0 | **48.0** | 41.0 | **49.0** |
| PRNG (noise floor) | 10.0 | **18.0** | 16.0 | 14.0 | 13.0 |
| Electricity (45K) | **84.8** | **84.8** | **84.8** | 79.0 | 83.7 |

**Concept-drift streams (test accuracy %, k=1, all methods capped at order 1):**

| Drift type | N-gram | PPM-D | CTW | **Predictor** | **Forest** |
|---|---|---|---|---|---|
| Sudden reversal | 2.5 | 2.5 | 4.5 | **97.0** | **97.0** |
| Gradual ramp | 5.0 | 5.0 | 6.2 | **98.3** | **98.3** |
| Recurring A→B→A | 3.8 | 3.3 | 4.2 | **97.5** | **97.5** |
| Fast (150-step cycles) | 40.0 | 39.6 | 40.4 | **94.6** | 93.3 |

The Electricity gap (79% vs 84.8%) reflects a fundamental property of credibility-based vs. count-based methods on high-autocorrelation stationary data: count-based methods can accumulate arbitrarily sharp predictions, while credibility is capped. The Forest closes this gap to 1.1pp (83.7%). On concept-drift tasks, count-based methods effectively collapse to random guessing while the Predictor maintains 94–98% accuracy.

---

## Files

| File | Purpose |
|---|---|
| `predictor.py` | `UniversalPredictor` — Module 1 core |
| `forest.py` | `PredictorForest` — Module 1 ensemble |
| `module2.py` | `GoalDirectedGenerator` — Module 2 (autoregressive, beam search, retrieval) |
| `datasets.py` | Dataset loaders (airline, text, DNA, weather, PRNG, electricity) |
| `similarity.py` | Surface similarity functions (hamming, gaussian, jaccard) — cold-start fallback |
| `ieee_benchmark.py` | Full benchmark suite generating LaTeX tables |
| `ieee_tables/` | Generated LaTeX tables and figures |
| `forest.py` | Ensemble of predictors |
| `run_experiments.py` | Quick single-predictor experiment runner |
| `run_forest.py` | Quick forest experiment runner |
| `tests.py` | Unit tests |
| `test_forest.py` | Forest-specific tests |
| `tasks/` | Paper roadmap and core-principle manifesto |

---

## How This Differs from Standard Approaches

| Property | N-gram / PPM-D | CTW | **This architecture** |
|---|---|---|---|
| Drift adaptation | None — counts only grow | None — counts only grow | Automatic via credibility degradation |
| Depth selection | Fixed or backoff heuristic | Bayesian mixture (stationary) | MWU — theoretically optimal for adversarial depth selection |
| Concept drift recovery | Requires reset or windowing | Requires reset or windowing | Self-correcting; speed proportional to prior confidence |
| Node count | O(V^k) worst case | O(V^k) worst case | O(sequence length) — only observed contexts |
| Online adaptation | Counts update, predictions sharpen | Weights update | Credibility update; fresh vs. stale nodes naturally separated |

The single deepest difference from count-based methods: **credibility is earned and can be lost.** A context that was reliable on Monday and wrong on Tuesday sees its influence reduced on Wednesday. Counts only accumulate.
