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

The root holds continuation counts (how many distinct predecessors each symbol appeared after, KN-style) for large vocabularies (|V|≥16), falling back to raw KT counts for small alphabets (DNA=4, Electricity=2) where continuation counts are too sparse. This seeds the blend with a better-calibrated unigram prior for text prediction.

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
effective_cap = C_MAX × (1 + 0.5 × log(1 + n_obs/100))  # adaptive (optional)
             = C_MAX                                       # fixed (default)

correct:  node_cred ← min(cap, node_cred × (1 + lr))
          succ_cred[actual] ← min(cap, succ_cred[actual] × (1 + lr))

wrong:    lr_down = lr × (1 + node_cred / cap)   # confidence-proportional
          node_cred ← max(C_MIN, node_cred × (1 − lr_down))
          succ_cred[wrong] ← max(C_MIN, succ_cred[wrong] × (1 − lr_down))
          succ_cred[actual] ← min(cap, succ_cred[actual] × (1 + lr))
```

With `adaptive_cap=True`, nodes with many observations are allowed to build higher credibility — the cap grows logarithmically with `n_obs`, so λ can approach 1 more closely on stationary data while the maximum `lr_down = 2×lr` is preserved (because `lr_down` normalizes by `cap`, not `C_MAX`).

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
| Heterogeneous k | Each tree uses a different context length: k, k+1, k+2, … capturing different temporal scales. Disabled for DNA (4-symbol near-uniform alphabet) where deeper-k trees add noise rather than signal. |
| Feedback dropout | Each tree independently skips learning on each step with probability `dropout` — the sequence analogue of bagging |
| Staggered offsets | Tree i doesn't start learning until step `i × stagger`; early topology has outsized influence on later structure |
| Inter-tree credibility | Each tree maintains a persistent weight updated by whether it was right; correct trees speak louder on the next prediction |

**Voting:** adaptive hybrid by default. Each tree contributes two representations:

- **Full blended distribution** (`tree._distribution()`) — the complete CTW-style probability over all vocabulary symbols. Used in the *mixture* component: proper calibration when trees at different context lengths express partial disagreement.
- **Mode-focused distribution** (`_tree_dist`) — only the most-probable successor at each depth, weighted by node credibility. Used in the *product* component: maximally decisive agreement signal for high-persistence or low-entropy data where unanimous tree confidence should dominate.

The adaptive blend computes `α × product(mode-focused) + (1−α) × mixture(full)` where `α` is the mean per-tree confidence — high confidence drives product-mode behaviour, uncertainty drives mixture-mode behaviour.

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
| **Retrieval** | Two-stage: (1) Bhattacharyya similarity on post-SEP trie distributions — exact for seen prompts; (2) surface Jaccard fallback when Bhattacharyya < 0.5 — domain-correct for novel tokens | Factual lookup; graceful degradation to novel inputs |

**Retrieval degradation tiers:**

| Bhattacharyya score | Signal | Response quality |
|---|---|---|
| ≈ 1.0 | Exact trie match | Correct answer |
| < 0.5 (with Jaccard fallback) | Novel token → root unigram | Domain-correct (right type: capital city, symbol, etc.) |
| No surface overlap | Root unigram, all prompts identical | Random training answer |

---

## Benchmark Results

Evaluated on 7 standard datasets (two large text corpora, full DNA genome) and 4 concept-drift streams. All methods use the same train/test split (80/20). Baselines: Persistence, Majority, N-gram(5), PPM-D(5), CTW(5).

**Standard benchmarks (test accuracy %):**

| Dataset | n | k | Persistence | PPM-D(5) | CTW(5) | **Predictor** | **Forest** |
|---|---|---|---|---|---|---|---|
| Airline passengers | 144 | 4 | 37.9 | 27.6 | 31.0 | **41.4** | **41.4** |
| Alice in Wonderland (15K) | 15,000 | 6 | 2.8 | 51.6 | **53.3** | 51.5 | 51.9 |
| Moby Dick (50K) | 50,000 | 6 | 2.1 | 45.7 | **47.4** | 45.8 | 46.1 |
| DNA — bacteriophage lambda (full) | 48,502 | 5 | 26.1 | 29.7 | **30.7** | 28.3 | 28.0 |
| Weather | 547 | 3 | **57.3** | 47.3 | 50.0 | **51.8** | 50.9 |
| PRNG (noise floor) | 500 | 3 | 10.0 | **18.0** | 16.0 | 14.0 | 13.0 |
| Electricity (45K) | 45,312 | 4 | **84.8** | **84.8** | **84.8** | 83.6 | 83.5 |

**Concept-drift streams (test accuracy %, k=1):**

| Drift type | N-gram | PPM-D | CTW | **Predictor** | **Forest** |
|---|---|---|---|---|---|
| Sudden reversal | 2.5 | 2.5 | 4.5 | **97.0** | **97.0** |
| Gradual ramp | 5.0 | 5.0 | 6.2 | **98.3** | **98.3** |
| Recurring A→B→A | 3.8 | 3.3 | 4.2 | **97.5** | **97.5** |
| Fast (150-step cycles) | 40.0 | 39.6 | 40.4 | **94.6** | 93.3 |

**Log-loss — Predictor wins on Weather; nearly ties CTW on DNA. PPM-D wins on Alice and Moby Dick.**

---

**Extended baseline comparison — KN, PPM\*, Online LSTM (test accuracy %):**

| Dataset | KN(5) | PPM\*(20) | LSTM(64) | Predictor | Forest |
|---|---|---|---|---|---|
| Airline passengers | 27.6 | 27.6 | 24.1 | 37.9 | **41.4** |
| Alice in Wonderland (15K) | **52.8** | 51.8 | 39.9 | 51.5 | 51.9 |
| Moby Dick (50K) | **47.2** | 45.3 | 38.6 | 45.8 | 46.1 |
| DNA — bacteriophage lambda | 30.1 | 26.6 | **32.5** | 28.1 | 28.0 |
| Weather | 50.9 | 48.2 | 43.6 | **51.8** | 50.9 |
| PRNG (noise floor) | 15.0 | **18.0** | 10.0 | 14.0 | 13.0 |
| Electricity (45K) | **84.8** | 81.9 | **84.8** | 83.6 | 83.5 |

KN(5) = Interpolated Kneser-Ney N-gram. PPM\*(20) = PPM with max order 20. LSTM(64) = single-layer LSTM, hidden size 64, trained online with BPTT-1 and Adam.

**Key findings from the extended comparison:**

- **KN(5) is the strongest text predictor** (52.8% Alice, 47.2% Moby) — continuation-count backoff outperforms Laplace-smoothed N-gram and is competitive with CTW on natural language.
- **Predictor leads on Weather** (51.8%) — KN continuation-count seeding, adaptive credibility cap (cred_max=6.05), and tuned lr (0.08) together lift Predictor from 48.2% to 51.8%, surpassing CTW (50.0%), KN (50.9%), Forest (50.9%), and PPM\*(48.2%).
- **LSTM wins on DNA** (32.5%, +1.8pp over CTW) — neural sequence modeling captures long-range non-Markovian dependencies in genomic data that any fixed-order predictor misses.
- **LSTM ties for best on Electricity** (84.8%) — converges cleanly for high-persistence binary streams after 36K training steps.
- **PPM\*(20) ≤ PPM-D(5) on DNA and Electricity** — order-20 contexts are too sparse for available training data; extra depth adds noise rather than signal.
- **Forest still dominates on Airline** (41.4%) — no counting-based or neural method comes close on short non-stationary time series.
- **LSTM underperforms on text** (Alice 39.9%, Moby 38.6%) — online BPTT-1 provides insufficient gradient signal for 26-symbol character-level language modeling.

---

### Confidence-gated prediction (abstain mode)

`UniversalPredictor` accepts a `min_confidence` parameter (default `0.0`). When set, the predictor abstains — returns `(None, conf)` — whenever its best prediction is less than `min_confidence × (1/|vocab|)` above the uniform baseline. A value of `1.5` means "only predict when at least 1.5× more confident than random."

Abstaining does not penalize the node: `node_cred` is unchanged (abstaining is not a wrong prediction). The successor distribution still updates so learning continues. This makes the warmup period implicit — early steps where the predictor is near-uniform simply produce no output rather than noisy guesses.

**Precision–coverage tradeoffs measured on natural language (Alice, k=4):**

| min_confidence | Accuracy (predicted only) | Coverage | Lift |
|---|---|---|---|
| 0.0 (off) | 48.5% | 100% | — |
| 3.0 | 50.3% | 96.5% | +1.8pp |
| 4.0 | 56.7% | 83.7% | +8.2pp |
| 5.0 | 59.4% | 77.6% | +10.9pp |
| 6.0 | 61.4% | 71.8% | +12.9pp |

The gain is real: predictions the model skips are genuinely its worst ones. Alice at min_conf=5.0 reaches 59.4% accuracy (vs CTW's 53.3% on 100% coverage) by only speaking when confident. For use cases where coverage matters less than per-prediction reliability, this is the correct mode.

**Note on vocabulary size:** `min_confidence` is normalized by `1/|vocab|`, so the same value applies comparably across different alphabet sizes (4 symbols for DNA, 26 for text). Weather (5-symbol alphabet, near-uniform predictor) requires `min_confidence < 1.5` to produce any predictions.

---

### The two-regime finding

Expanding from small samples to full datasets exposed a fundamental architectural property:

**Data-limited regime (n ≲ 5K):** Credibility builds up quickly, blend weights become decisive, and the Predictor is competitive or best. At 1,500 DNA bases the Predictor was 33.0% — best across all methods.

**Architecture-limited regime (n ≫ CRED_MAX/lr ≈ 80 steps to cap):** Every node hits `CRED_MAX=8.0` and the blend weight freezes at λ=8/9=0.889. Count-based methods (PPM-D, CTW) have no cap — their counts keep growing, giving predictions increasingly close to 1.0. At 48K DNA bases CTW reaches 30.7% while the Predictor drops to 26.2%.

**The exception is noisy and drifting data.** Weather improved from 41% to 51.8% with tuning — **the Predictor leads on Weather** (51.8% vs Forest 50.9%, KN 50.9%, CTW 50.0%). Noisy, high-variance datasets where count-based methods overfit to stale patterns are exactly the Predictor's domain. On all four drift streams the Predictor still dominates at 94–98% regardless of scale.

**The CRED_MAX cap is a design choice, not a bug.** A node with unbounded credibility would adapt from drift in O(n) steps. The cap guarantees O(1/CRED_MAX) adaptation speed: a maximally-trusted node that turns wrong loses credibility at 2× the base rate (confidence-proportional degradation). The trade-off is explicit: fast drift recovery at the cost of long-term convergence on stationary data.

---

## Files

| File | Purpose |
|---|---|
| `predictor.py` | `UniversalPredictor` — Module 1 core |
| `forest.py` | `PredictorForest` — Module 1 ensemble |
| `module2.py` | `GoalDirectedGenerator` — Module 2 (autoregressive, beam search, retrieval) |
| `baselines.py` | Standard baselines: Persistence, Majority, N-gram, PPM-D |
| `baselines_extended.py` | Extended baselines: KN, PPM\*, Online LSTM |
| `datasets.py` | Dataset loaders (airline, text, DNA, weather, PRNG, electricity) |
| `similarity.py` | Surface similarity functions (hamming, gaussian, jaccard) |
| `ieee_benchmark.py` | Full benchmark suite generating LaTeX tables |
| `ieee_tables/` | Generated LaTeX tables and figures |
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
