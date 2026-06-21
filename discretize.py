# Root shim — kept for backward compatibility with research scripts.
# Source of truth: markov_explorer/discretize.py
from markov_explorer.discretize import *  # noqa: F401,F403
from markov_explorer.discretize import (  # noqa: F401
    FeatureDiscretizer, LabelEncoder, MISSING, MISSING_STR,
    _to_rows, _is_missing, _is_numeric_col, _safe_str,
    _quantile_edges, _bin_search,
)
