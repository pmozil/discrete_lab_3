"""Implementation of the CART algorithm to train decision tree classifiers."""
import numpy as np

from node import Node


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 16) -> None:
        self.max_depth = max_depth

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ) -> None:
        """
        Grow the tree from given data

        Args:
            features: np.ndarray - the array of features
            targets: np.ndarray - the araay of features to split by
        """
        # classes are assumed to go from 0 to n-1
        self.n_classes_ = len(set(y))
        self.n_features_ = features.shape[1]
        self.tree_ = self._grow_tree(features, targets)

    def predict(self, object_features: np.ndarray) -> list:
        """
        Predict the value for given features

        Args:
            features: np.ndarray - the features of specific object
        """
        return [self._predict(inputs) for inputs in object_features]

    def _gini(self, targets):
        """Compute Gini for a node.

        Args:
            targets: np.ndarray
        """
        m = targets.size
        return 1.0 - sum(
                (np.sum(targets == c) / m) ** 2
                for c in range(self.n_classes_)
            )

    def _best_split(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ) -> tuple[int, float]:
        """Find the best split for a node.

        Args:

        Returns:
        """
        m = targets.size
        if m <= 1:
            return -1, -1

        parent_counts = [
            np.sum(targets == c) for c in range(self.n_classes_)
        ]

        best_gini = 1.0 - sum((n / m) ** 2 for n in parent_counts)
        id, thr = -1, -1

        for idx in range(self.n_features_):
            thresholds, classes = zip(
                *sorted(
                    zip(features[:, idx], targets)
                )
            )

            lefts = [0] * self.n_classes_
            rights = parent_counts.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                lefts[c] += 1
                rights[c] -= 1
                gini_left = 1.0 - sum(
                    (lefts[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (rights[x] / (m - i)) ** 2
                    for x in range(self.n_classes_)
                )

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    id = idx
                    thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return id, float(thr)

    def _grow_tree(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        depth=0
    ) -> Node:
        """Build a decision tree by recursively finding the best split."""
        samples_per_self = [
            np.sum(targets == i) for i in range(self.n_classes_)
        ]
        preds = np.argmax(samples_per_self)
        node = Node(
            gini=self._gini(targets),
            samples=y.size,
            samples_per_self=samples_per_self,
            targets=preds,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = self._best_split(features, targets)
            print(idx)
            print(thr)
            indices_left = features[:, idx] < thr
            features_left, targets_left = (
                features[indices_left],
                targets[indices_left]
            )
            features_right, targets_right = (
                features[~indices_left],
                targets[~indices_left]
            )
            node.feature_index = idx
            node.threshold = thr
            node.left = self._grow_tree(
                features_left, targets_left, depth + 1
            )
            node.right = self._grow_tree(
                features_right, targets_right, depth + 1
            )
        return node

    def _predict(
        self,
        inputs: np.ndarray
    ) -> np.ndarray:
        """Predict class for a single sample."""
        if self.tree_ is None:
            return 0
        node = self.tree_
        while node.left is not None:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.targets


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
    from sklearn.tree import export_graphviz
    from sklearn.utils import Bunch

    parser = argparse.ArgumentParser(description="Train a decision tree.")
    parser.add_argument("--dataset", choices=["breast", "iris", "wifi"], default="wifi")
    parser.add_argument("--max_depth", type=int, default=1)
    parser.add_argument("--hide_details", dest="hide_details", action="store_true")
    parser.set_defaults(hide_details=False)
    parser.add_argument("--use_sklearn", dest="use_sklearn", action="store_true")
    parser.set_defaults(use_sklearn=False)
    args = parser.parse_args()

    # 1. Load dataset.
    if args.dataset == "breast":
        dataset = load_breast_cancer()
    elif args.dataset == "iris":
        dataset = load_iris()
    elif args.dataset == "wifi":
        # https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
        df = pd.read_csv("wifi_localization.txt", delimiter="\t")
        data = df.to_numpy()
        dataset = Bunch(
            data=data[:, :-1],
            target=data[:, -1] - 1,
            feature_names=["Wifi {}".format(i) for i in range(1, 8)],
            target_names=["Room {}".format(i) for i in range(1, 5)],
        )
    X, y = dataset.data, dataset.target

    # 2. Fit decision tree.
    if args.use_sklearn:
        clf = SklearnDecisionTreeClassifier(max_depth=args.max_depth)
    else:
        clf = DecisionTreeClassifier(max_depth=args.max_depth)
    clf.fit(X, y)

    # 3. Predict.
    if args.dataset == "iris":
        input = [0, 0, 5.0, 1.5]
    elif args.dataset == "wifi":
        input = [-70, 0, 0, 0, -40, 0, 0]
    elif args.dataset == "breast":
        input = [np.random.rand() for _ in range(30)]
    pred = clf.predict([input])[0]
    print("Input: {}".format(input))
    print("Prediction: " + dataset.target_names[pred])

    # 4. Visualize.
    if args.use_sklearn:
        export_graphviz(
            clf,
            out_file="tree.dot",
            feature_names=dataset.feature_names,
            class_names=dataset.target_names,
            rounded=True,
            filled=True,
        )
        print("Done. To convert to PNG, run: dot -Tpng tree.dot -o tree.png")
