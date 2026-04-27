"""Greedy Decision Set learner for rule extraction."""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Rule:
    feature_idx: int
    feature_name: str
    threshold: float
    operator: str  # ">=" or "<="
    precision: float
    recall: float
    support: int
    coverage: int

    def to_human_readable(self) -> str:
        return f"IF {self.feature_name} {self.operator} {self.threshold:.2f} THEN anomaly"

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feature_idx]
        if self.operator == ">=":
            return col >= self.threshold
        else:
            return col <= self.threshold


class GreedyDecisionSet:
    """
    Greedy precision-maximizing decision set.
    Prediction: if ANY rule fires → anomaly.
    Each rule targets remaining uncovered positives.
    """

    def __init__(
        self,
        max_rules: int = 50,
        min_precision: float = 0.95,
        min_support: int = 50,
        feature_names: Optional[List[str]] = None,
    ):
        self.max_rules = max_rules
        self.min_precision = min_precision
        self.min_support = min_support
        self.feature_names = feature_names
        self.rules: List[Rule] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pos_idx = set(np.where(y == 1)[0])
        neg_idx = set(np.where(y == 0)[0])
        print(f"Fitting on {len(y)} samples ({len(pos_idx)} pos, {len(neg_idx)} neg)")

        remaining = pos_idx.copy()
        self.rules = []

        for i in range(self.max_rules):
            if len(remaining) < self.min_support:
                print(f"Stopping: {len(remaining)} positives remain (< min_support)")
                break
            rule = self._find_best_rule(X, remaining, neg_idx)
            if rule is None:
                print(f"Stopping: no rule meets precision >= {self.min_precision}")
                break
            self.rules.append(rule)
            covered = set(np.where(rule.evaluate(X))[0])
            remaining -= covered
            print(
                f"Rule {i+1}: {rule.to_human_readable()} "
                f"| prec={rule.precision:.3f} rec={rule.recall:.3f} "
                f"support={rule.support} remaining={len(remaining)}"
            )

        print(f"\nLearned {len(self.rules)} rules, covering {len(pos_idx)-len(remaining)}/{len(pos_idx)} positives")

    def _find_best_rule(
        self, X: np.ndarray, remaining: set, neg_idx: set
    ) -> Optional[Rule]:
        from tqdm import tqdm
        best_rule: Optional[Rule] = None
        best_score = -1.0

        for feat in tqdm(range(X.shape[1]), desc="  scanning features", leave=False):
            col = X[:, feat]
            unique = np.unique(col)
            thresholds = np.unique(np.percentile(unique, np.linspace(0, 100, 101))) if len(unique) > 100 else unique

            for thr in thresholds:
                for op, mask in [(">=", col >= thr), ("<=", col <= thr)]:
                    covered = set(np.where(mask)[0])
                    tp = len(covered & remaining)
                    fp = len(covered & neg_idx)
                    total = tp + fp
                    if total == 0:
                        continue
                    prec = tp / total
                    if prec < self.min_precision or tp < self.min_support:
                        continue
                    score = prec * 100 + tp
                    if score > best_score:
                        best_score = score
                        feat_name = self.feature_names[feat] if self.feature_names else f"feat_{feat}"
                        best_rule = Rule(
                            feature_idx=feat,
                            feature_name=feat_name,
                            threshold=float(thr),
                            operator=op,
                            precision=prec,
                            recall=tp / len(remaining),
                            support=tp,
                            coverage=total,
                        )
        return best_rule

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.rules:
            return np.zeros(X.shape[0], dtype=int)
        fired = np.zeros(X.shape[0], dtype=bool)
        for rule in self.rules:
            fired |= rule.evaluate(X)
        return fired.astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.rules:
            return np.zeros(X.shape[0])
        scores = np.zeros(X.shape[0])
        for rule in self.rules:
            scores += rule.evaluate(X).astype(float)
        return scores / len(self.rules)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {
                "feature_idx": r.feature_idx,
                "feature_name": r.feature_name,
                "threshold": float(r.threshold),
                "operator": r.operator,
                "precision": float(r.precision),
                "recall": float(r.recall),
                "support": r.support,
                "coverage": r.coverage,
            }
            for r in self.rules
        ]

    @classmethod
    def from_dict(cls, rules_data: List[Dict[str, Any]], feature_names: Optional[List[str]] = None) -> "GreedyDecisionSet":
        ds = cls(feature_names=feature_names)
        ds.rules = [
            Rule(
                feature_idx=r["feature_idx"],
                feature_name=r["feature_name"],
                threshold=r["threshold"],
                operator=r["operator"],
                precision=r["precision"],
                recall=r["recall"],
                support=r["support"],
                coverage=r["coverage"],
            )
            for r in rules_data
        ]
        return ds
