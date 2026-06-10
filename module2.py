"""
Module 2 — Goal-Directed Generation

Uses a trained UniversalPredictor (Module 1) as a world model to steer
generation toward a target, complete a prompt, or retrieve a stored response.

Training convention
-------------------
Format any prompt→response task as a flat token sequence:

    [prompt tokens ...] SEPARATOR [response tokens ...] END

Module 1 learns that SEPARATOR is followed by responses, not more prompts.
No architectural changes to Module 1 are needed.

Three strategies
----------------
  autoregressive  — greedy token-by-token generation from a context seed.
  beam_search     — keep N candidate sequences, expand and prune at each step.
  retrieve        — find the nearest prompt in training data by successor-
                    distribution similarity; return its stored response.

All strategies operate read-only on a frozen predictor: the predictor's trie
and history are not modified during generation.
"""

from __future__ import annotations

import math
from typing import Any, Hashable, Sequence

# ── sentinel tokens ───────────────────────────────────────────────────────────

SEPARATOR: str = "<SEP>"
END:       str = "<END>"


# ── helpers ───────────────────────────────────────────────────────────────────

def _walk_trie(root, context: tuple):
    """Walk the predictor's trie with a context tuple; return node or None."""
    node = root
    for sym in context:
        if sym not in node.children:
            return None
        node = node.children[sym]
    return node


def _kt_dist(node, vocab: set, alpha: float) -> dict[Any, float]:
    """KT-smoothed distribution from a trie node (or root)."""
    V     = len(vocab)
    total = sum(node.succ_cred.values()) or 1.0
    d     = {s: (node.succ_cred.get(s, 0) + alpha) / (total + alpha * V)
             for s in vocab}
    t = sum(d.values())
    return {s: v / t for s, v in d.items()}


def _blend_from_context(predictor, context: tuple) -> dict[Any, float]:
    """
    Compute the CTW-blended distribution for an arbitrary context tuple
    without modifying predictor state.
    """
    V     = max(len(predictor._vocab), 1)
    alpha = 0.5 / V
    vocab = predictor._vocab

    root_total = sum(predictor._root.succ_cred.values()) or 1.0
    blended = {s: (predictor._root.succ_cred.get(s, 0) + alpha) /
                   (root_total + alpha * V)
               for s in vocab}

    k = predictor.k
    for d in range(1, min(k, len(context)) + 1):
        ctx  = context[-d:]
        node = _walk_trie(predictor._root, ctx)
        if node is None or not node.succ_cred:
            continue
        total = sum(node.succ_cred.values()) or 1.0
        local = {s: (node.succ_cred.get(s, 0) + alpha) / (total + alpha * V)
                 for s in vocab}
        lam     = node.node_cred / (node.node_cred + 1.0)
        blended = {s: lam * local[s] + (1 - lam) * blended[s] for s in vocab}

    t = sum(blended.values())
    if t < 1e-12:
        return {s: 1.0 / V for s in vocab}
    return {s: v / t for s, v in blended.items()}


def _bhattacharyya(p: dict, q: dict) -> float:
    """Bhattacharyya coefficient between two distributions over a shared alphabet."""
    keys = set(p) | set(q)
    return sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)


# ── main class ────────────────────────────────────────────────────────────────

class GoalDirectedGenerator:
    """
    Module 2: goal-directed generation over a trained UniversalPredictor.

    Parameters
    ----------
    predictor : UniversalPredictor
        A trained Module 1 instance.  Must have been trained on sequences that
        include the SEPARATOR and END tokens if Q&A formatting is used.
    separator : Hashable
        Token that separates prompts from responses (default: SEPARATOR).
    end : Hashable
        Token that marks the end of a response (default: END).
    """

    def __init__(self, predictor, separator=SEPARATOR, end=END):
        self.predictor = predictor
        self.separator = separator
        self.end       = end

    # ── strategy 1: autoregressive ────────────────────────────────────────────

    def complete(
        self,
        seed: Sequence,
        max_steps: int = 100,
        stop_on: Any = None,
        temperature: float = 1.0,
    ) -> list:
        """
        Greedily extend `seed` token by token using the predictor's trie.

        The predictor's history and trie are NOT modified.

        Parameters
        ----------
        seed        : token sequence to use as the initial context.
        max_steps   : maximum tokens to generate.
        stop_on     : stop and exclude this token if generated (default: self.end).
        temperature : > 1.0 flattens the distribution (more diverse),
                      < 1.0 sharpens it (more deterministic). 1.0 = greedy argmax.

        Returns
        -------
        list of generated tokens (does not include the seed).
        """
        stop_on = stop_on if stop_on is not None else self.end
        context = list(seed)
        result  = []

        for _ in range(max_steps):
            ctx  = tuple(context[-self.predictor.k:])
            dist = _blend_from_context(self.predictor, ctx)

            if not dist:
                break

            if temperature == 1.0 or temperature <= 0:
                token = max(dist, key=dist.get)
            else:
                import random
                # Apply temperature and sample
                scaled = {s: v ** (1.0 / temperature) for s, v in dist.items()}
                total  = sum(scaled.values())
                scaled = {s: v / total for s, v in scaled.items()}
                r = random.random()
                cumulative = 0.0
                token = max(dist, key=dist.get)  # fallback
                for s, p in scaled.items():
                    cumulative += p
                    if r <= cumulative:
                        token = s
                        break

            if token == stop_on:
                break

            result.append(token)
            context.append(token)

        return result

    def answer(self, prompt: Sequence, max_steps: int = 100) -> list:
        """
        Feed `prompt + [SEPARATOR]` as context seed and generate the response.

        Convenience wrapper around `complete()` for the standard Q&A format.
        """
        seed = list(prompt) + [self.separator]
        return self.complete(seed, max_steps=max_steps)

    # ── strategy 2: beam search ───────────────────────────────────────────────

    def beam_search(
        self,
        seed: Sequence,
        beam_width: int = 5,
        max_steps: int = 100,
        stop_on: Any = None,
        length_penalty: float = 0.6,
    ) -> list[tuple[list, float]]:
        """
        Beam search over Module 1's distributions from a context seed.

        Parameters
        ----------
        seed          : initial context sequence.
        beam_width    : number of beams to maintain.
        max_steps     : maximum tokens per beam.
        stop_on       : token that closes a beam (default: self.end).
        length_penalty: score normalization exponent (Google NMT formula:
                        score / len^length_penalty). 0 = no penalty, 1 = linear.

        Returns
        -------
        List of (token_sequence, score) pairs, ordered best-first.
        Each token_sequence does NOT include the seed or the stop token.
        """
        stop_on = stop_on if stop_on is not None else self.end

        # Each beam: (context_so_far, generated_tokens, cumulative_log_prob)
        beams:    list[tuple[list, list, float]] = [(list(seed), [], 0.0)]
        complete: list[tuple[list, float]]        = []

        for _ in range(max_steps):
            if not beams:
                break

            candidates: list[tuple[list, list, float]] = []

            for context, generated, log_prob in beams:
                ctx  = tuple(context[-self.predictor.k:])
                dist = _blend_from_context(self.predictor, ctx)

                if not dist:
                    complete.append((generated, log_prob))
                    continue

                # Expand by all vocab tokens; keep top beam_width per beam
                top = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                top = top[:beam_width]

                for token, prob in top:
                    lp = log_prob + math.log(max(prob, 1e-12))
                    if token == stop_on:
                        complete.append((generated, lp))
                    else:
                        candidates.append((context + [token],
                                           generated + [token],
                                           lp))

            if not candidates:
                break

            # Normalize by length and keep top beam_width
            def _score(item):
                _, gen, lp = item
                n = max(len(gen), 1)
                return lp / (n ** length_penalty)

            beams = sorted(candidates, key=_score, reverse=True)[:beam_width]

        # Drain remaining open beams into complete
        for context, generated, log_prob in beams:
            complete.append((generated, log_prob))

        # Sort complete beams by length-normalized score
        def _final_score(item):
            gen, lp = item
            n = max(len(gen), 1)
            return lp / (n ** length_penalty)

        complete.sort(key=_final_score, reverse=True)
        return complete

    # ── strategy 3: retrieval ─────────────────────────────────────────────────

    def retrieve(
        self,
        query: Sequence,
        corpus: list[tuple[Sequence, Any]],
        top_k: int = 1,
    ) -> list[tuple[Any, float]]:
        """
        Find the most similar prompt(s) in `corpus` using the post-separator
        successor distribution as the comparison signal.

        How it works
        ------------
        After training on [prompt SEP answer END] sequences, the trie learns
        a distribution over the FIRST ANSWER TOKEN at every context ending in
        [prompt[-k:] + SEP].  This distribution encodes what answer the model
        associates with each training prompt — Paris for France, Berlin for
        Germany, etc.

        For retrieval, we compare the model's prediction at [query + SEP] with
        its prediction at [each_training_prompt + SEP].  The training prompt
        whose post-SEP distribution most resembles the query's is returned.

        For seen queries this is exact (Bhattacharyya ≈ 1.0 against the correct
        match, near 0 against others).  For novel queries the model falls back
        to a shallower context; the returned response is the best guess from
        whatever partial context the trie can match.

        Parameters
        ----------
        query  : the query prompt tokens (without SEPARATOR or answer).
        corpus : list of (prompt_tokens, response) pairs from training.
        top_k  : number of (response, similarity_score) results to return.
        """
        # Distribution at [query + SEP] = what the model predicts first for this prompt
        query_seed = tuple(list(query)[-self.predictor.k:] + [self.separator])
        query_dist = _blend_from_context(self.predictor, query_seed)

        scores = []
        for prompt, response in corpus:
            seed = tuple(list(prompt)[-self.predictor.k:] + [self.separator])
            dist = _blend_from_context(self.predictor, seed)
            sim  = _bhattacharyya(query_dist, dist)
            scores.append((response, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── utilities ─────────────────────────────────────────────────────────────

    def train_on_pairs(
        self,
        pairs: list[tuple[Sequence, Sequence]],
        repeats: int = 1,
    ) -> None:
        """
        Train the predictor on (prompt, response) pairs in the standard format:
            [prompt ... SEPARATOR response ... END]

        Parameters
        ----------
        pairs   : list of (prompt_tokens, response_tokens) pairs.
        repeats : number of passes through the corpus (online learning: 1 is usual).
        """
        p = self.predictor
        for _ in range(repeats):
            for prompt, response in pairs:
                sequence = (list(prompt) + [self.separator] +
                            list(response) + [self.end])
                for token in sequence:
                    pred, _ = p.predict()
                    p.observe(token)
                    p.feedback(token)
