import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight


class SurrogateDT:
    def __init__(self, max_depth=10, min_samples_leaf=5):
        self.direct_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        # For predicting soft probabilities from Teacher's float distribution,
        # a regressor minimizes MSE between predicted probabilities and target probabilities
        self.distilled_model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        self._is_direct_fitted = False
        self._is_distilled_fitted = False

    def fit_direct(self, features: np.ndarray, hard_labels: np.ndarray):
        """Trains directly on ground truth labels."""
        self.direct_model.fit(features, hard_labels)
        self._is_direct_fitted = True

    def fit_distilled(
        self,
        features: np.ndarray,
        soft_labels: np.ndarray,
        hard_labels: np.ndarray | None = None,
    ):
        """Trains on soft continuous probabilities emitted from the Teacher.

        hard_labels: if provided, used to compute balanced sample weights so the
        regressor is not dominated by the majority class.
        """
        sample_weight = (
            compute_sample_weight("balanced", hard_labels)
            if hard_labels is not None
            else None
        )
        self.distilled_model.fit(features, soft_labels, sample_weight=sample_weight)
        self._is_distilled_fitted = True

    def evaluate(self, features: np.ndarray, labels: np.ndarray, teacher_preds=None) -> dict:
        """Evaluates both models against ground truth labels."""
        results = {}
        labels_np = np.array(labels)

        def _per_class(preds_np):
            acc = accuracy_score(labels_np, preds_np)
            p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
                labels_np, preds_np, average="macro", zero_division=0
            )
            p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
                labels_np, preds_np, average=None, zero_division=0
            )
            return {
                "accuracy": acc,
                "precision": p_mac, "recall": r_mac, "f1": f1_mac,
                "normal":  {"precision": float(p_cls[0]), "recall": float(r_cls[0]),
                            "f1": float(f1_cls[0]), "support": int(support[0])},
                "attack":  {"precision": float(p_cls[1]), "recall": float(r_cls[1]),
                            "f1": float(f1_cls[1]), "support": int(support[1])},
            }

        if self._is_direct_fitted:
            preds = np.array(self.direct_model.predict(features))
            results["direct"] = _per_class(preds)

        if self._is_distilled_fitted:
            soft_preds = self.distilled_model.predict(features)
            if soft_preds.ndim > 1 and soft_preds.shape[1] > 1:
                preds = np.argmax(soft_preds, axis=1)
            else:
                preds = (soft_preds > 0.5).astype(int)
            results["distilled"] = _per_class(preds)

            if teacher_preds is not None:
                tp = np.array(teacher_preds)
                teacher_hard = np.argmax(tp, axis=1) if tp.ndim > 1 else tp
                results["distilled"]["fidelity_to_teacher"] = accuracy_score(teacher_hard, preds)
                attack_mask = teacher_hard == 1
                if attack_mask.sum() > 0:
                    results["distilled"]["attack_fidelity"] = accuracy_score(
                        teacher_hard[attack_mask], preds[attack_mask]
                    )

        return results

    def extract_rules(self, feature_names=None):
        """
        Extract simplified readable IF-THEN rules from the tree models.
        """
        rules = []
        if self._is_distilled_fitted:
            rules.append("Distilled model rules (stub placeholder for scikit tree traversal)")
            # Standard post-order traversal to extract string conditions 
        return rules
