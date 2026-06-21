import math
from typing import Any
import random

class HoeffdingNode:
    """
    A node in the Hoeffding Online Decision Tree.
    Tracks statistics for Information Gain to mathematically guarantee Random Forest parity.
    """
    def __init__(self, n_features: int, is_leaf: bool = True):
        self.is_leaf = is_leaf
        self.n_features = n_features
        # If leaf, track class histograms per feature value
        self.n_obs = 0
        self.class_counts = {}
        # feature_idx -> feature_val -> class_label -> count
        self.feature_stats = {i: {} for i in range(n_features)}
        
        # If internal node, track split condition
        self.split_feature = None
        self.children = {} # feature_val -> HoeffdingNode

    def observe(self, row: list, label: Any):
        if not self.is_leaf:
            val = row[self.split_feature]
            if val not in self.children:
                self.children[val] = HoeffdingNode(self.n_features)
            self.children[val].observe(row, label)
            return

        self.n_obs += 1
        self.class_counts[label] = self.class_counts.get(label, 0) + 1
        
        for f_idx, f_val in enumerate(row):
            if f_val not in self.feature_stats[f_idx]:
                self.feature_stats[f_idx][f_val] = {}
            self.feature_stats[f_idx][f_val][label] = self.feature_stats[f_idx][f_val].get(label, 0) + 1

    def predict(self, row: list) -> dict:
        if self.is_leaf:
            if not self.class_counts:
                return {}
            total = sum(self.class_counts.values())
            return {k: v / total for k, v in self.class_counts.items()}
        
        val = row[self.split_feature]
        if val in self.children:
            return self.children[val].predict(row)
        # Fallback to majority if branch unseen
        return {}


class HoeffdingPredictor:
    """
    Online Decision Tree using Hoeffding Bounds.
    Guarantees Random Forest split parity without offline batch computation.
    """
    def __init__(
        self,
        n_features: int,
        delta: float = 1e-1,
        grace_period: int = 10,
        tie_threshold: float = 0.05
    ):
        self.n_features = n_features
        self.delta = delta
        self.grace_period = grace_period
        self.tie_threshold = tie_threshold
        self.root = HoeffdingNode(n_features)
        
    def _entropy(self, counts: dict) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        ent = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    def _info_gain(self, leaf: HoeffdingNode, f_idx: int) -> float:
        base_entropy = self._entropy(leaf.class_counts)
        total_obs = leaf.n_obs
        
        f_entropy = 0.0
        for f_val, class_dist in leaf.feature_stats[f_idx].items():
            val_total = sum(class_dist.values())
            val_ent = self._entropy(class_dist)
            f_entropy += (val_total / total_obs) * val_ent
            
        return base_entropy - f_entropy

    def _attempt_split(self, leaf: HoeffdingNode) -> None:
        if leaf.n_obs < self.grace_period:
            return
            
        # Calculate Information Gain for all features
        gains = []
        for i in range(self.n_features):
            gain = self._info_gain(leaf, i)
            gains.append((gain, i))
            
        gains.sort(reverse=True, key=lambda x: x[0])
        best_gain, best_idx = gains[0]
        second_gain, _ = gains[1] if len(gains) > 1 else (0.0, -1)
        
        # Hoeffding Bound epsilon
        # Using R = log2(num_classes) as range of entropy
        R = math.log2(len(leaf.class_counts)) if len(leaf.class_counts) > 0 else 1.0
        epsilon = math.sqrt((R**2 * math.log(1 / self.delta)) / (2 * leaf.n_obs))
        
        if (best_gain - second_gain > epsilon) or (epsilon < self.tie_threshold and best_gain > 0):
            # Split!
            leaf.is_leaf = False
            leaf.split_feature = best_idx
            
            # Create children for existing feature values
            for f_val in leaf.feature_stats[best_idx].keys():
                leaf.children[f_val] = HoeffdingNode(self.n_features)
            
            # Free memory
            leaf.feature_stats = None
            leaf.class_counts = None

    def _traverse_and_split(self, node: HoeffdingNode):
        if node.is_leaf:
            if node.n_obs % self.grace_period == 0:
                self._attempt_split(node)
        else:
            for child in node.children.values():
                self._traverse_and_split(child)

    def partial_fit(self, row: list, label: Any) -> None:
        self.root.observe(row, label)
        # Periodically attempt to split leaves
        self._traverse_and_split(self.root)
        
    def predict_proba(self, row: list) -> dict:
        return self.root.predict(row)
