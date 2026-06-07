# Universal Sequence Predictor

A prediction architecture built from first principles. 
Given any sequence of observations, it learns to predict what comes next — for any data type, in any domain —
 without assuming a fixed model, a known distribution, or temporal ordering.

---

## The Core Idea

The algorithm builds a living map of reality from observations.
 Every time it sees a situation and its outcome,
  it plants a flag at that location.
   When asked to predict, it looks around at the nearest credible flags and asks what they point toward.

The data itself is the model. Nothing is abstracted away.

---

## How It Works — Step by Step

### Step 1 — Observe
A new (situation, outcome) pair arrives.
 The situation is described by a feature vector —
  anything that captures the current state.
   The algorithm plants a flag at this location in feature space, labelled with the outcome. This is a **node**.

### Step 2 — Predict
When a prediction is needed, three things happen in sequence:

**The Optimizer** decides how many nodes are allowed to speak.
 It adapts based on query difficulty — unfamiliar territory gets a larger budget, familiar territory a smaller one. 
 It learns from whether past budgets were enough.
**The Allocator** decides which nodes fill that budget. It scores every candidate on four criteria simultaneously:
- **Similarity** — how close is this node's situation to the current one?
- **Credibility** — how reliable has this node proven to be?
- **Diversity** — does this node add a new perspective, or just echo what's already represented?
- **Coupling** — is this node strongly linked to another already-selected node?

Only the highest-scoring combination speaks.

**Communication** — selected nodes signal each other via learned coupling links.
 Nodes that have reliably co-predicted in the past amplify each other.
  Nodes that have co-misled each other inhibit each other. 
  The strength of this channel, **λ**, is itself learned: it grows when communication helps and shrinks when it misleads.

The result is a weighted vote across all active nodes.
 The winning candidate is the prediction.
  The **confidence score** reflects how decisive the vote was and how much evidence contributed —
   it is a calibrated measure of certainty, not just a number.

### Step 3 — Feedback
Once the actual outcome is revealed, every component updates:

- **Credibility** — each active node is scored independently on its own similarity to the current situation. Correct nodes grow. Wrong nodes shrink. This is peer-independent: a node's update is the same whether three or three thousand other nodes were also active.
- **Coupling** — pairs of nodes that agreed and were correct grow their link strength. Pairs that agreed and were wrong have their link attenuated.
- **λ** — was the communication channel pointing toward the correct answer? If yes, λ grows. If no, λ shrinks.
- **Optimizer** — was the budget large enough? If the system was constrained and wrong, it grows the ceiling. If it had slack and was right, it trims it.
- **Allocator** — tracks which nodes produced correct predictions when selected, building a per-node selection quality history.

### Step 4 — Dynamic Node Creation
Three triggers can create new nodes beyond the standard one-per-observation:

| Trigger | Condition | Node type | Purpose |
|---|---|---|---|
| Standard | Every observation | Observed | Normal learning |
| Exploration | No existing node covers this context well | Exploration | Anchor novel territory |
| Correction | High confidence + wrong prediction | Correction | Override a blind spot |

Exploration and correction nodes start with higher initial credibility so they immediately influence predictions in the situations they were created for.

### Step 5 — Convergence Tracking
After each feedback cycle, the algorithm records its overall **similarity quality** — how differentiated the credibility scores have become across the node population. These snapshots form a learning curve.

The algorithm fits a mathematical model to that curve and extrapolates to its asymptote: the **plateau L**. This is the best state achievable given the current similarity function and data. The **lookahead** projects quality at any future point, telling you whether more observations will help or whether the similarity function has become the ceiling.

---

## Architecture Summary

| Component | What it is | What it does |
|---|---|---|
| **Node** | A stored (situation, outcome) pair with a credibility score | The basic unit of knowledge |
| **Similarity** | Domain-specific function comparing two situations | Determines which nodes are relevant now |
| **Credibility** | Per-node learned track record | Amplifies reliable nodes, suppresses misleading ones |
| **Coupling** | Learned pairwise link between nodes | Nodes that co-predict correctly reinforce each other |
| **λ (lambda)** | Learned scalar on the coupling channel | Controls how much nodes trust each other's signal |
| **Optimizer** | Adaptive budget controller | Learns the minimum number of speakers needed per prediction |
| **Allocator** | Intelligent node selector | Picks the most informative combination within budget |
| **Vigilance ρ** | Novelty threshold | Triggers new node creation in uncovered territory |
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
| Activation | Dense — all neurons fire for every input | Sparse — only contextually relevant nodes speak |
| Communication | Directional layer → layer only | Lateral — any active node can signal any other |
| Weights | Abstract numbers with no semantic meaning | Grounded — credibility is a track record, coupling is a co-success history |
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
