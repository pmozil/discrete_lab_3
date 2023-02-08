"""Binary tree with decision tree semantics and ASCII visualization."""
import numpy as np


class Node:
    """A decision tree node.

    Parameters:
        gini: float - the node's gini value
        samples: int - the amount of samples used for fitting
        samples_per_self: int - the same as samples,
            but filters bad data
        targets: np.ndarray - the class to be predicted
        threshold: float - the criterion for going left or right
        left: Node | None - the left leaf
        right: Node | Node - the right leaf
    """

    def __init__(
        self,
        gini: float,
        samples: int,
        samples_per_self: list,
        targets: np.intp
    ) -> None:
        """
        Init for the Node class

        Args:
            gini: float - the node's gini value
            samples: int - the amount of samples used for fitting
            samples_per_self: int - the same as samples,
                but filters bad data
            targets: np.ndarray - the class to be predicted
        """
        self.gini = gini
        self.samples = samples
        self.samples_per_self = samples_per_self
        self.targets = targets
        self._feature_index = 0
        self._threshold = 0
        self._left = None
        self._right = None

    @property
    def feature_index(self) -> float:
        """
        Getter for self.feature_index
        """
        return self._feature_index

    @feature_index.setter
    def feature_index(self, feature_index: float) -> None:
        """
        Setter for feature_index

        Args:
            feature_index: float
        """
        self._feature_index = feature_index

    @property
    def threshold(self) -> float:
        """
        Getter for self._threshold
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """
        Setter for self._threshold

        Args:
            threshold: float
        """
        self._threshold = threshold

    @property
    def left(self) -> "Node":
        """
        Getter for self._left
        """
        return self._left

    @left.setter
    def left(self, left: "Node") -> None:
        """
        Setter for self._left

        Args:
            left: Node | None
        """
        self._left = left

    @property
    def right(self) -> "Node":
        """
        Getter for self._right
        """
        return self._right

    @right.setter
    def right(self, right: "Node") -> None:
        """
        Setter for self._right

        Args:
            right: Node | None
        """
        self._right = right
