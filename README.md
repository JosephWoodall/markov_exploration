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

## How It Works — Step by Step

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
- **Coupling (directional EMA)** — four cases, each updating a specific directed link:
  - Both correct → symmetric positive: `coupling[(i,j)]` and `coupling[(j,i)]` move toward +1
  - Both wrong, same prediction → symmetric negative: both links move toward -1
  - i right, j wrong → `coupling[(i,j)]` moves toward -1 (i suppresses j when i is present); `coupling[(j,i)]` unchanged
  - j right, i wrong → symmetric case for the other direction

  Each link updates via an adaptive EMA: `α = max(0.05, coupling_lr / (1 + n · coupling_lr))` where n is the number of times this specific directed pair has been observed. The learning rate starts at `coupling_lr` and decays naturally with evidence — fast early (high uncertainty), stable later (earned trust) — with a floor of 0.05 to keep links live under concept drift. No fixed alpha. No hand-tuned schedule.

- **λ** — was the communication channel pointing toward the correct answer? If yes, λ grows. If no, λ shrinks.
- **Optimizer** — was the budget large enough? If the system was constrained and wrong, it grows the ceiling. If it had slack and was right, it trims it.
- **Allocator** — tracks which nodes produced correct predictions when selected, building a per-node selection quality history.

### Step 4 — Dynamic Node Creation
Three triggers can create new nodes beyond the standard observed nodes at each scale:

| Trigger | Condition | Node type | Purpose |
|---|---|---|---|
| Standard | Every observation | Observed | Normal learning at all scales |
| Exploration | Full-scale max_sim < ρ | Exploration | Anchor genuinely novel territory |
| Correction | High confidence + wrong prediction | Correction | Override a blind spot |

Exploration fires on **full-scale** similarity only — coverage by a short-context node does not count as knowing the full situation. Exploration and correction nodes start with higher initial credibility so they immediately influence predictions in the situations they were created for.

### Step 5 — Convergence Tracking
After each feedback cycle, the algorithm records its overall **similarity quality** — how differentiated the credibility scores have become across the node population. These snapshots form a learning curve.

The algorithm fits a model to that curve and extrapolates to its asymptote: the **plateau L**. This is the best state achievable given the current similarity function and data. The **lookahead** projects quality at any future point, telling you whether more observations will help or whether the similarity function has become the ceiling.

---

## Architecture Summary

| Component | What it is | What it does |
|---|---|---|
| **Node** | A stored (situation, outcome) pair with a credibility score | The basic unit of knowledge |
| **Variable-order context** | Nodes stored at all scales 1..k; scale selected per prediction | Uses the most specific scale with reliable coverage; falls back when data is sparse |
| **Similarity** | Domain-specific function comparing two situations | Determines which nodes are relevant now |
| **Credibility** | Per-node learned track record | Amplifies reliable nodes, suppresses misleading ones |
| **Coupling** | Directed EMA link between node pairs | Encodes who suppresses whom when they disagree; learned incrementally, adaptive rate |
| **λ (lambda)** | Learned scalar on the coupling channel | Controls how much nodes trust each other's signal |
| **Optimizer** | Adaptive budget controller | Learns the minimum number of speakers needed per prediction |
| **Allocator** | Intelligent node selector | Picks the most informative combination within budget |
| **Vigilance ρ** | Novelty threshold | Triggers new node creation when full-scale context is uncovered |
| **Confidence** | Vote decisiveness × evidence size | Calibrated uncertainty — the algorithm's measure of its own certainty |
| **Plateau L** | Asymptotic quality ceiling | The best this system can do given its similarity function |

---

## The Limiting Factor — and How to Address It

The architecture itself imposes no limit on what it can learn. The only domain-specific component is the **similarity function**. If the similarity function compares surface form (raw characters, raw numbers), the algorithm is limited to surface patterns.

**The solution is always the same: make the similarity function operate on meaning, not form.**

| Domain | Shallow similarity (limited) | Deep similarity (unlimited) |
|---|---|---|
| Text | Character matching (Hamming) | Sentence embeddings (semantic vectors) |
| DNA | Raw nucleotide matching | Learned biological feature representations |
| Numbers | Euclidean distance on raw values | Normalised, domain-scaled feature vectors |
| Events | Set overlap | Learned event embeddings |
| Any domain | Raw values | Features that capture the underlying structure |

When the similarity function captures the true geometry of the domain, data volume ceases to be a limiting factor. The algorithm finds reliable predictors because it is looking in the right space.

---

## How This Differs from a Neural Network

| Property | Neural Network | This Architecture |
|---|---|---|
| Topology | Fixed before training | Emergent — grows where the problem demands it |
| Context | Fixed input dimension | Variable-order — uses the longest scale with reliable coverage |
| Activation | Dense — all neurons fire for every input | Sparse — only contextually relevant nodes speak |
| Communication | Directional layer → layer only | Lateral — directed coupling between any active pair |
| Weights | Abstract numbers with no semantic meaning | Grounded — credibility is a track record, coupling is a directional trust relationship |
| Learning rate | Fixed or scheduled | Adaptive per relationship — high early, decays with evidence, floored for drift |
| Generalization | Compression — training data discarded after training | Interpolation — all observations kept, similarity spans the gaps |
| Feedback | Global loss, backpropagation touches every weight | Local — only contributing nodes update per prediction |
| Memory | None — the model is the weights | Explicit — every observation is a first-class node |
| Self-knowledge | None | Knows its own plateau, confidence calibration, and convergence state |

The single deepest difference: **in a neural network, the architecture is the model. Here, the data is the model.** Learned parameters are annotations on stored observations, not abstractions of them.

---

## Files

| File | Purpose |
|---|---|
| `predictor.py` | Core `UniversalPredictor` class |
| `components.py` | `Optimizer` and `Allocator` components |
| `similarity.py` | Pluggable similarity functions (gaussian, hamming, jaccard) |
| `datasets.py` | Dataset loaders (airline, text, DNA, weather, PRNG) |
| `run_experiments.py` | Full five-dataset experiment suite |
| `tests.py` | Architecture-specific test suite (calibration, coupling gain, hypothesis quality, Gini, node efficiency) |
| `animation.py` | Visual animation of the algorithm in action |
