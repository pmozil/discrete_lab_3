import numpy as np 

from decision_tree_classifier import DecisionTreeClassifier

class Estimator:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        min_predicitons: list[int] = [5, 10, 20],
        max_depth: list[int] = [3, 5, 10, 16]
    ) -> None:
        """
        Init fot eh Estimator

        Args:
        """
        self.pred_counter = 0
        self.depth_counter = 0
        self.min_predicitons = min_predicitons
        self.max_depth = max_depth
        self.features = features
        self.targets = targets

    def _next_tree(self) -> DecisionTreeClassifier | None:
        """
        Train the next Tree so as to check all their accuracies

        Returns:
            DecisionTreeClassifier - the next tree with the new arguments
        """
        if self.depth_counter >= len(self.max_depth):
            return None
        
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth[self.depth_counter],
            min_predictions=self.min_predicitons[self.pred_counter]
        )
        tree.fit(self.features, self.targets)

        self.pred_counter = (self.pred_counter + 1) % len(self.min_predicitons)
        if self.pred_counter == 0:
            self.depth_counter += 1

        return tree
    
    def _calc_accuracy(self, tree: DecisionTreeClassifier | None) -> float:
        """
        Calculate the accuracyt of the tree on the given dataset

        Args:
            tree: DecisionTreeClassifier - the decision tree

        Returns:
            float - the number in [0, 1] (the accuracy)
        """
        if tree is None:
            return 0
        features_guessed = 0
        for ind, input in enumerate(self.features):
            pred = tree.predict([input])[0]
            if pred == self.targets[ind]:
                features_guessed += 1
        return features_guessed / len(self.targets)

    def best_tree(self) -> DecisionTreeClassifier:
        """
        Find best tree for the given predictions and depths
        
        Returns:
            DecisionTreeClassifier - the best tree
        """
        tree = self._next_tree()
        accuracy = self._calc_accuracy(tree)
        while (new_tree := self._next_tree()) != None:
            new_acc = self._calc_accuracy(new_tree)
            if new_acc >= accuracy and new_tree is not None:
                accuracy = new_acc
                tree = new_tree
        return tree


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
    # clf = DecisionTreeClassifier(
    #     max_depth=args.max_depth, min_predictions=args.min_predictions
    # )
    # clf.fit(X, y)
    pred = Estimator(features=X, targets=y)
    clf = pred.best_tree()
    clf_sklearn.fit(X, y)

    # 3. Predict.
    count = 0
    for _ in range(1000):
        input = [np.random.rand() for _ in range(30)]
        pred = clf.predict([input])[0]
        pred_sklearn = clf_sklearn.predict([input])[0]
        print(f"""
Prediction: {dataset.target_names[pred]}; \
Sklearn prediction: {dataset.target_names[pred_sklearn]}
        """)
        if pred == pred_sklearn:
            count += 1
    print(f"Accuracy: {count / 1000}")
