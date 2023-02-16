import numpy as np
from itertools import product
from typing import Iterator, List

from decision_tree_classifier import DecisionTreeClassifier


class Estimator:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        min_predictions: List[int] = [5, 10, 20],
        max_depth: List[int] = [3, 5, 10, 16],
    ) -> None:
        """
        Init fot eh Estimator

        Args:
        """
        self.min_predictions = min_predictions
        self.max_depth = max_depth
        self.features = features
        self.targets = targets

    @property
    def _next_tree(self) -> Iterator[DecisionTreeClassifier]:
        """
        Train the next Tree so as to check all their accuracies

        Returns:
            DecisionTreeClassifier - the next tree with the new arguments
        """
        for depth, preds in product(self.max_depth, self.min_predictions):
            tree = DecisionTreeClassifier(
                max_depth=depth,
                min_predictions=preds,
            )
            tree.fit(self.features, self.targets)
            yield tree

    @property
    def best_tree(self) -> DecisionTreeClassifier:
        """
        Find best tree for the given predictions and depths

        Returns:
            DecisionTreeClassifier - the best tree
        """
        trees = self._next_tree
        accuracy = -1
        for tree in trees:
            if tree.accuracy > accuracy:
                cur_tree = tree
                accuracy = tree.accuracy
        return cur_tree

# Sorry ;(
if __name__ == "__main__":
    import argparse
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.tree import (
        DecisionTreeClassifier as SklearnDecisionTreeClassifier,
    )
    from sklearn.model_selection import GridSearchCV

    parser = argparse.ArgumentParser(description="Train a decision tree.")
    parser.add_argument(
        "--dataset", choices=["breast", "iris", "wifi"], default="breast"
    )
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_predictions", type=int, default=30)
    parser.add_argument(
        "--hide_details", dest="hide_details", action="store_true"
    )
    parser.set_defaults(hide_details=False)
    parser.add_argument(
        "--use_sklearn", dest="use_sklearn", action="store_true"
    )
    parser.set_defaults(use_sklearn=False)
    args = parser.parse_args()

    # 1. Load dataset.
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    # 2. Fit decision tree.
    clf_sklearn = SklearnDecisionTreeClassifier()
    clf_sklearn.fit(X, y)
    # clf = DecisionTreeClassifier(
    #     max_depth=args.max_depth, min_predictions=args.min_predictions
    # )
    # clf.fit(X, y)
    pred = Estimator(
        features=X,
        targets=y,
        max_depth=list(range(2, 10, 2)),
        min_predictions=list(range(2, 30, 4)),
    )
    clf = pred.best_tree

    # 3. Predict.
    count = 0
    for _ in range(100000):
        input = [np.random.rand() for _ in range(30)]
        pred = clf.predict([input])[0]
        pred_sklearn = clf_sklearn.predict([input])[0]
        print(
            f"""
Prediction: {dataset.target_names[pred]}; \
Sklearn prediction: {dataset.target_names[pred_sklearn]}
        """
        )
        if pred == pred_sklearn:
            count += 1
    print(f"Accuracy: {count / 100000}")
