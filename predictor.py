# Root shim — kept for backward compatibility with research scripts.
# Source of truth: markov_explorer/predictor.py
from markov_explorer.predictor import *  # noqa: F401,F403
from markov_explorer.predictor import (  # noqa: F401
    UniversalPredictor, _TrieNode, _CRED_MIN, _CRED_MAX,
)
