"""Implementation of the CART algorithm to train decision tree classifiers."""
import numpy as np

from node import Node


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 16, min_predictions: int = 20) -> None:
        self._max_depth = max_depth
        self._min_predictions = min_predictions

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Grow the tree from given data

        Args:
            features: np.ndarray - the array of features
            targets: np.ndarray - the araay of features to split by
        """
        # classes are assumed to go from 0 to n-1
        self._classes = len(set(y))
        self._features = features.shape[1]
        self._tree = self._grow_tree(features, targets)

    def predict(self, object_features: np.ndarray) -> list:
        """
        Predict the value for given features

        Args:
            object_features: np.ndarray - the features of specific object
        """
        return [self._predict(inputs) for inputs in object_features]

    def _gini(self, targets):
        """Compute Gini for a node.

        Args:
            targets: np.ndarray
        """
        m = targets.size
        return 1.0 - sum(
            (np.sum(targets == c) / m) ** 2 for c in range(self._classes)
        )

    def _best_split(
        self, features: np.ndarray, targets: np.ndarray
    ) -> tuple[int, float]:
        """Find the best split for a node.

        Args:
            features: np.ndarray - the array of features
            targets: np.ndarray - the array of targets (results)

        Returns:
            tuple[int, float] - the pair with the index of feature and threshold
        """
        m = targets.size
        if m <= 1:
            return -1, -1

        parent_counts = [np.sum(targets == c) for c in range(self._classes)]

        best_gini = 1.0 - sum((n / m) ** 2 for n in parent_counts)
        id, thr = -1, -1

        for idx in range(self._features):
            thresholds, classes = zip(*sorted(zip(features[:, idx], targets)))

            lefts = [0] * self._classes
            rights = parent_counts.copy()
            for i in range(1, m):
                c = classes[i - 1]
                lefts[c] += 1
                rights[c] -= 1
                gini_left = 1.0 - sum(
                    (lefts[x] / i) ** 2 for x in range(self._classes)
                )
                gini_right = 1.0 - sum(
                    (rights[x] / (m - i)) ** 2 for x in range(self._classes)
                )

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    id = idx
                    thr = (thresholds[i] + thresholds[i - 1]) / 2

        return id, float(thr)

    def _grow_tree(
        self, features: np.ndarray, targets: np.ndarray, depth: int = 0
    ) -> Node:
        """
        Recursively construct the tree

        Args:
            features: np.ndarray - the array of features
            target: np.ndarray - the array of targets (results)
            depth: int - the current tree depth
        """
        samples_per_self = [np.sum(targets == i) for i in range(self._classes)]
        preds = np.argmax(samples_per_self)
        node = Node(
            gini=self._gini(targets),
            samples=y.size,
            samples_per_self=samples_per_self,
            targets=preds,
        )

        if (
            depth < self._max_depth
            and samples_per_self[preds] >= self._min_predictions
        ):
            idx, thr = self._best_split(features, targets)
            print(idx)
            print(thr)
            indices_left = features[:, idx] < thr
            features_left, targets_left = (
                features[indices_left],
                targets[indices_left],
            )
            features_right, targets_right = (
                features[~indices_left],
                targets[~indices_left],
            )
            node.feature_index = idx
            node.threshold = thr
            node.left = self._grow_tree(features_left, targets_left, depth + 1)
            node.right = self._grow_tree(
                features_right, targets_right, depth + 1
            )
        return node

    def _predict(self, object_features: np.ndarray) -> np.intp:
        """
        Predict the class for a single sample

        Args:
            object_features: np.ndarray - the features of a single object
        """
        if self._tree is None:
            raise ValueError
        node = self._tree
        while node.left is not None:
            node = (
                node.left
                if object_features[node.feature_index] < node.threshold
                else node.right
            )
        return node.targets


# Sorry ;(
if __name__ == "__main__":
    import argparse
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.tree import (
        DecisionTreeClassifier as SklearnDecisionTreeClassifier,
    )

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
    clf_sklearn = SklearnDecisionTreeClassifier(max_depth=args.max_depth)
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth, min_predictions=args.min_predictions
    )
    clf.fit(X, y)
    clf_sklearn.fit(X, y)

    # 3. Predict.
    count = 0
    for _ in range(10):
        input = [np.random.rand() for _ in range(30)]
        pred = clf.predict([input])[0]
        pred_sklearn = clf_sklearn.predict([input])[0]
        print(f"""
Prediction: {dataset.target_names[pred]}; \
Sklearn prediction: {dataset.target_names[pred_sklearn]}
        """)
        if pred == 0:
            count += 1
    print(f"Total maligns: {count}")
