# Universal Sequence Predictor

A prediction architecture built from first principles.
Given any sequence of observations, it learns to predict what comes next — for any data type, in any domain —
without assuming a fixed model, a known distribution, or temporal ordering.

---

## The Core Idea

The algorithm builds a living map of reality from observations.
Every time it sees a situation and its outcome, it plants a flag at that location.
When asked to predict, it looks around at the nearest credible flags and asks what they point toward.

The data itself is the model. Nothing is abstracted away.

---

## Architecture Overview

The system is designed in two layers:

**Module 1 — Universal Sequence Predictor**
A domain-agnostic next-element predictor. Given any sequence, it learns to predict what comes next using a dynamic node topology, variable-order context, and self-supervised similarity. No domain knowledge required anywhere in the pipeline.

**Module 2 — Goal-Directed Generation (proposed)**
A deliberation layer that uses Module 1 as a world model to steer generation toward a target — enabling question answering, completion, and other goal-directed tasks. Module 1 is intuition; Module 2 is deliberation.

---

## Module 1 — How It Works

### Step 1 — Observe
A new (situation, outcome) pair arrives. The situation is described by a context window of the last `k` observations. The algorithm plants a flag at **every scale** from length 1 up to length `k`, labelled with the outcome. Each scale captures a different resolution of the same situation. These are **nodes**.

### Step 2 — Predict
When a prediction is needed, three things happen in sequence:

**Scale selection** — the algorithm finds the longest context length that has reliable coverage (at least one node with similarity ≥ vigilance ρ). If the full-length context is well-covered, it uses that. If not, it falls back to progressively shorter contexts until it finds coverage. Predictions are always made at a single coherent scale, not a mixture of all scales simultaneously.

**The Optimizer** decides how many nodes are allowed to speak. It adapts based on query difficulty — unfamiliar territory gets a larger budget, familiar territory a smaller one. It learns from whether past budgets were enough.

**The Allocator** decides which nodes fill that budget. It scores every candidate on four criteria simultaneously:
- **Similarity** — how close is this node's situation to the current one?
- **Credibility** — how reliable has this node proven to be?
- **Diversity** — does this node add a new perspective, or just echo what's already represented?
- **Coupling** — does an already-selected node vouch for this one?

Only the highest-scoring combination speaks.

**Communication** — selected nodes signal each other via learned directional coupling links.
`coupling[(i, j)]` encodes the additive effect on node j when node i is present.
A node that has reliably called it right while another called it wrong earns the ability to suppress that other node's vote. λ, the coupling channel strength, is itself learned: it grows when communication helps and shrinks when it misleads.

The result is a weighted vote across all active nodes. The winning candidate is the prediction. The **confidence score** reflects how decisive the vote was and how much evidence contributed — it is a calibrated measure of certainty, not just a number.

### Step 3 — Feedback
Once the actual outcome is revealed, every component updates:

- **Credibility** — each active node is scored independently on its own raw similarity to the current situation. Correct nodes grow. Wrong nodes shrink. This is peer-independent: a node's update is the same whether three or three thousand other nodes were also active.
- **Successor distributions** — the empirical distribution of outcomes following each context window is updated at all scales. This is the raw material for similarity.
- **Coupling (directional EMA)** — four cases, each updating a specific directed link:
  - Both correct → symmetric positive: both links move toward +1
  - Both wrong, same prediction → symmetric negative: both links move toward -1
  - i right, j wrong → `coupling[(i,j)]` moves toward -1 (i suppresses j when present); `coupling[(j,i)]` unchanged
  - j right, i wrong → symmetric case for the other direction

  Each link updates via an adaptive EMA: `α = max(0.05, coupling_lr / (1 + n · coupling_lr))` where n is the number of times this directed pair has been observed. Fast early, stable later, floored for concept drift.

- **λ** — was the communication channel pointing toward the correct answer? If yes, λ grows. If no, λ shrinks.
- **Optimizer** — was the budget large enough? If constrained and wrong, it grows the ceiling. If slack and right, it trims.
- **Allocator** — tracks which nodes produced correct predictions when selected.

### Step 4 — Dynamic Node Creation
Three triggers can create new nodes beyond the standard observed nodes at each scale:

| Trigger | Condition | Node type | Purpose |
|---|---|---|---|
| Standard | Every observation | Observed | Normal learning at all scales |
| Exploration | Full-scale max_sim < ρ | Exploration | Anchor genuinely novel territory |
| Correction | High confidence + wrong prediction | Correction | Override a blind spot |

Exploration fires on **full-scale** similarity only — coverage by a short-context node does not count as knowing the full situation.

### Step 5 — Convergence Tracking
After each feedback cycle, the algorithm records its overall **similarity quality** — how differentiated the credibility scores have become. These snapshots form a learning curve that is fit to an exponential model, extrapolating a **plateau L** — the best achievable state given the current data.

---

## The Similarity Function — A Self-Supervised General Solution

The only domain-specific component in traditional sequence predictors is the similarity function. A shallow similarity (character matching, Euclidean distance) limits the system to surface patterns. This architecture solves that without domain knowledge.

### Successor-Distribution Similarity

Two contexts are similar if they tend to be followed by the same outcomes. The similarity between context A and context B is the **Bhattacharyya coefficient** between their empirical successor distributions:

```
sim(A, B) = Σ √( P(x|A) · P(x|B) )   for all outcomes x
```

This measure is:
- **Self-supervised** — learned entirely from the sequence, no labels needed
- **Domain-agnostic** — works identically for text, DNA, numbers, events, or any other type
- **Semantically grounded** — two contexts that look completely different on the surface are recognised as similar if they predict the same things

### Bayesian Cold-Start Blend

Successor distributions need observations to be reliable. A single-observation distribution is noise. The system handles this with a smooth Bayesian blend:

```
sim(A, B) = (1 - w) · surface_sim(A, B)  +  w · distribution_sim(A, B)

where  w = f(n_A) · f(n_B)   and   f(n) = n / (n + prior)
```

At zero observations: `w = 0`, pure surface similarity (the prior).
As evidence accumulates: `w → 1`, pure successor-distribution similarity (the posterior).
The product form means we only trust the blend when **both** contexts have sufficient evidence.

The surface similarity function (Hamming, Gaussian, etc.) is retained as a fallback parameter — it provides a useful prior in any domain.

---

## Module 2 — Goal-Directed Generation (Proposed)

Module 1 is already generative: apply it autoregressively — predict next, append to context, predict next again — and you get generation. The challenge for question answering is **conditioning**: making it produce answers rather than continuations.

Module 1 is **intuition** — fast, associative, pattern-matching.
Module 2 is **deliberation** — goal-directed, using Module 1 as a world model.

### Training Format
Represent Q&A as a flat sequence:
```
[question tokens ...] [SEPARATOR] [answer tokens ...] [END]
```
Module 1 learns that SEPARATOR is followed by answers, not more questions. No architectural changes to Module 1 are needed.

### Three Module 2 Strategies

| Strategy | Mechanism | Best for |
|---|---|---|
| **Retrieval** | Use successor-distribution similarity to find the most similar question in the training set; return its stored answer | Factual Q&A, fast lookup |
| **Autoregressive** | Feed `[question + SEPARATOR]` as context; generate token by token until END | Short answers, direct generation |
| **Beam search** | Module 2 runs beam search over Module 1's per-token scores, guided by a goal function | Long-form answers, constrained generation |

**Retrieval is the natural starting point.** The successor-distribution similarity already performs semantic search — it finds similar questions without any additional machinery. The (question, answer) pairs from training become a queryable store.

### Key Constraint
Module 1's fixed context window means the beginning of a long question may fall out of context by the time the answer is being generated. This is the fundamental difference between a sliding-window predictor and full attention. For short Q&A (factual lookups, pattern completion), this is not a limiting factor.

---

## What is a Node?

A node is the fundamental unit of knowledge in the system — a single frozen memory. Every node stores exactly four things:

| Field | What it holds | Example |
|---|---|---|
| **Context** | The last k observations before something happened | `['t', 'h', 'e']` |
| **Successor** | What actually happened next | `' '` (a space) |
| **Credibility** | How reliable this memory has proven to be | `2.4` |
| **Type** | Why this node was created | `observed` / `exploration` / `correction` |

That is the complete node. It contains no rules, no weights in the neural network sense, no abstracted features. It is a specific remembered moment: *"when I saw this exact situation, that happened next, and here is my track record since."*

Nodes do not know about each other directly. The predictor holds the successor distributions (histograms of outcomes seen after each context window) and the coupling links (directed trust relationships between node pairs). Nodes are the raw anchors; the distributions and coupling are annotations built on top of them over time.

### Node types and their starting credibilities

| Type | Created when | Initial credibility | Purpose |
|---|---|---|---|
| `observed` | Every timestep | 1.0 | Normal learning from experience |
| `exploration` | Full-scale max\_sim < ρ | 2.0 | Anchor genuinely novel territory — starts louder to immediately influence predictions |
| `correction` | High confidence + wrong prediction | 3.0 | Override a blind spot — starts loudest because the system was confidently wrong |

The higher starting credibility for exploration and correction nodes ensures they influence predictions immediately rather than having to earn trust from scratch.

---

## Forest Architecture (Module 1 Ensemble)

A `PredictorForest` is a collection of N `UniversalPredictor` instances that start identical and diverge through experience. The key insight is the same one behind random forests: a diverse ensemble where individuals disagree is more robust than a single large predictor, because their disagreements are informative — regions of consensus signal genuine structure; regions of conflict signal uncertainty or noise.

### How diversity is created — four mechanisms

**1. Heterogeneous context lengths**

Each tree uses a different window size: tree 0 gets `k`, tree 1 gets `k+1`, tree 2 gets `k+2`, and so on. This is the most direct implementation of the "derivative at a point" idea — different k values capture different orders of the temporal structure of the sequence. Short-k trees capture broad, fast-changing patterns. Long-k trees capture deep, slow-changing structure. The vote aggregates across all temporal scales simultaneously.

**2. Feedback dropout**

Each tree independently decides, with probability `dropout`, to skip the learning step for a given observation. The observation is still seen (all trees maintain the same history), but the node creation and credibility/coupling updates are skipped. Over time, each tree builds its topology from a different subset of learning events — different nodes get created first, different coupling links form, different credibility scores develop.

This is the sequence analogue of bootstrap aggregating (bagging) in random forests: each tree learns from a random subset of the data.

**3. Staggered offsets**

Tree i does not begin learning until it has seen `i × stagger` steps. This gives each tree a different starting point: tree 0 builds its early topology from the very beginning of the sequence; tree 4 (with stagger=25) doesn't start until step 100. The early structure of a predictor's topology has an outsized influence on how it develops — nodes created first tend to accumulate credibility faster and anchor more coupling links. Staggering the start ensures genuine structural diversity rather than just stochastic variation.

**4. Inter-tree credibility (adaptive weighting)**

Each tree maintains a persistent credibility weight that is updated after every prediction: correct trees grow, incorrect trees shrink, by a fixed learning rate. This weight is multiplied by the per-prediction confidence when aggregating votes, so trees that have been reliably right on similar recent queries speak louder. The weights are normalised periodically to prevent runaway growth.

This is the forest-level analogue of the node-level credibility mechanism inside each tree — the same principle (track record drives trust) operating at two different scales simultaneously.

### Voting: mixture vs. product-of-experts

**Mixture** (confidence-weighted sum): each tree contributes its full distribution, weighted by `confidence × tree_credibility`. The winning outcome accumulates the most total weight. Inclusive and smooth.

**Product-of-experts** (weighted geometric mean): an outcome's score is the weighted geometric mean of its probability across all active trees. An outcome that one tree assigns near-zero probability is strongly suppressed regardless of what other trees think. This means:

- On structured data where trees genuinely agree, the product sharpens the prediction
- On random data where trees learned different noise patterns, they disagree → the product collapses toward uniform → low confidence → correct "I don't know"

Product is the recommended default for general use. Mixture is better when the training data is sparse and you want every tree's signal to count even when they conflict.

### Why this resembles the brain

The forest architecture maps surprisingly closely onto theories of biological cognition:

| Forest mechanism | Neural analogue |
|---|---|
| Heterogeneous k | Cortical hierarchy — each level processes different temporal scales |
| Feedback dropout | Stochastic synaptic transmission — not every spike propagates |
| Staggered offsets | Cortical column specialisation — columns exposed to different stimuli first develop different receptive fields |
| Inter-tree credibility | Neuromodulatory gating — dopamine and acetylcholine selectively amplify reliable circuits |
| Product-of-experts | Coincidence detection — a postsynaptic neuron fires only when multiple inputs agree |
| Node credibility within each tree | Long-term potentiation and depression — synaptic strength reflects track record |
| Node coupling within each tree | Hebbian learning — connections between nodes that fire together and are right together grow stronger |

The entire architecture, from single nodes up to the forest, is built on one repeated idea: **track record drives trust, and trust drives influence.**

---

## Architecture Summary

| Component | What it is | What it does |
|---|---|---|
| **Node** | A stored (situation, outcome) pair with a credibility score | The basic unit of knowledge |
| **Variable-order context** | Nodes at all scales 1..k; scale selected per prediction | Uses the most specific scale with reliable coverage |
| **Successor distribution** | Empirical outcome histogram per context window | Raw material for self-supervised similarity |
| **Similarity (blend)** | Bayesian mix of distribution and surface similarity | Transitions from surface prior to semantic posterior as evidence accumulates |
| **Credibility** | Per-node learned track record | Amplifies reliable nodes, suppresses misleading ones |
| **Coupling** | Directed EMA link between node pairs | Encodes who suppresses whom when they disagree; adaptive learning rate |
| **λ (lambda)** | Learned scalar on the coupling channel | Controls how much nodes trust each other's signal |
| **Optimizer** | Adaptive budget controller | Learns the minimum number of speakers needed per prediction |
| **Allocator** | Intelligent node selector | Picks the most informative combination within budget |
| **Vigilance ρ** | Novelty threshold (full-scale only) | Triggers new node creation when full context is uncovered |
| **Confidence** | Vote decisiveness × evidence size | Calibrated uncertainty |
| **Plateau L** | Asymptotic quality ceiling | The best this system can do given current data |

---

## How This Differs from a Neural Network

| Property | Neural Network | This Architecture |
|---|---|---|
| Topology | Fixed before training | Emergent — grows where the problem demands it |
| Context | Fixed input dimension | Variable-order — uses the longest scale with reliable coverage |
| Similarity | Learned implicitly via weights | Self-supervised from successor distributions — semantically grounded |
| Activation | Dense — all neurons fire for every input | Sparse — only contextually relevant nodes speak |
| Communication | Directional layer → layer only | Lateral — directed coupling between any active pair |
| Weights | Abstract numbers with no semantic meaning | Grounded — credibility is a track record, coupling is a directional trust relationship |
| Learning rate | Fixed or scheduled | Adaptive per relationship — decays with evidence, floored for drift |
| Generalization | Compression — training data discarded | Interpolation — all observations kept, similarity spans the gaps |
| Feedback | Global loss, backpropagation | Local — only contributing nodes update per prediction |
| Memory | None — the model is the weights | Explicit — every observation is a first-class node |
| Generation | Sampling from learned distribution | Autoregressive prediction; Module 2 adds goal-directed steering |
| Self-knowledge | None | Knows its own plateau, confidence calibration, and convergence state |

The single deepest difference: **in a neural network, the architecture is the model. Here, the data is the model.** Learned parameters are annotations on stored observations, not abstractions of them.

---

## Files

| File | Purpose |
|---|---|
| `predictor.py` | Core `UniversalPredictor` class (Module 1) |
| `components.py` | `Optimizer` and `Allocator` components |
| `similarity.py` | Surface similarity functions used as cold-start fallback (gaussian, hamming, jaccard) |
| `datasets.py` | Dataset loaders (airline, text, DNA, weather, PRNG) |
| `forest.py` | `PredictorForest` — ensemble of predictors with heterogeneous k, dropout, staggered offsets, inter-tree credibility, and product-of-experts voting |
| `run_experiments.py` | Five-dataset experiment suite (single predictor) |
| `run_forest.py` | Five-dataset experiment suite (forest) |
| `tests.py` | Architecture-specific test suite (calibration, coupling gain, hypothesis quality, Gini, node efficiency) |
| `animation.py` | Visual animation of the algorithm in action |
| `test_forest.py` | Forest-specific architectural tests (grow/prune mechanisms, regression, adaptive voting) |

---

## Current Issue — Similarity Generalization

**Status:** partially resolved (2026-06-07)

### What the similarity function actually is

`similarity_fn` is a **cold-start prior** — it tells the system which contexts to treat as similar before enough data exists to judge by outcomes.
The architecture already has a domain-agnostic similarity inside `_compute_sim`: two contexts are similar if they lead to the same outcomes (Bhattacharyya coefficient on successor distributions).
As evidence accumulates the blend weight `w → 1` and the distributional similarity takes over regardless of what `similarity_fn` does.

The only question is what happens during cold start (sparse data).

### The cold-start fix

`similarity_fn=None` previously fell back to `0.0` — no signal, no predictions.
Changed to **exact-match** (`1.0` if contexts are identical, `0.0` otherwise):
at cold start the system acts as a pure n-gram lookup; as distributions build it generalises automatically.

### Experiment results (forest, adaptive voting)

| Dataset | Baseline | Hamming / Gaussian | sim=None | PPMI embed |
|---|---|---|---|---|
| Airline Passengers | 12.5% | +121% | **+203%** | +66% |
| Alice in Wonderland | 4.0% | **+208%** | -8% | +8% |
| DNA | 25.0% | +15% | +15% | **+21%** |
| Weather | 33.3% | +38% | **+44%** | +41% |
| PRNG (noise floor) | 10.0% | ~0% | ~0% | ~0% |

### What the results mean

**`sim=None` wins on structured, repeating data (Airline, Weather).**
Seasonal patterns repeat across years — the exact-match prior fires correctly, distributions build quickly, and the distributional similarity does the rest.
No domain knowledge needed.

**Hamming wins on text by a large margin.**
Character overlap is a genuinely useful signal for English: contexts sharing a common prefix ("th_") really do have similar successors.
PPMI embeddings give only a modest improvement over exact-match because character co-occurrence at window=3 captures which characters appear *near* each other, not which contexts are *substitutable* — the distinction that matters for prediction.

**PPMI embeddings improve DNA.**
Nucleotide co-occurrence within a small window captures real biological structure (base-pairing patterns, codon statistics).
With only 4 symbols, PPMI vectors are compact and informative.

### The remaining gap for text

The meaningful unit in English is the word or subword, not the individual character.
"the" and "a" are similar contexts not because they share characters but because they both precede nouns.
Detecting this requires either:
- **More data**: with millions of characters, the distributional similarity learns word-level patterns on its own
- **Pre-trained word/subword embeddings** (GloVe, FastText, BPE): transfer learned similarity from a large corpus; use as the cold-start prior here
- **Larger sequence units**: tokenise at the word level before feeding to the predictor

This is the same bottleneck that motivated Word2Vec (2013) and BERT (2018) in neural NLP.
Those models did not escape the data-scale problem — they solved it by training on billions of tokens and letting gradient descent learn the embedding jointly with the task.
This architecture separates the two steps (embed first, predict second), which is more interpretable but requires the embedding to be good independently.
