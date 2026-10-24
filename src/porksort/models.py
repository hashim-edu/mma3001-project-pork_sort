"""Baseline and classical classifiers for frame-level PASS/FAIL.

Each model maps a feature vector (see `porksort.features`) to a defect score
in [0, 1]. They are the classification counterparts of the regression models
in the unit:

============== ============================== ===========================
name           model                          unit topic it extends
============== ============================== ===========================
``majority``   always predicts the FAIL rate  baseline (5.5)
``colour_lr``  logistic regression on colour  linear regression (5.1)
               histogram only
``logreg``     logistic regression, all       linear regression (5.1)
               features
``tree``       decision tree                  decision tree regression (5.2)
``svm``        RBF support vector machine     support vector regression (5.3)
``mlp``        multilayer perceptron          neural network regression (5.4)
============== ============================== ===========================

All except the trees are preceded by standardisation (zero mean, unit
variance per feature). Without it, features with large ranges dominate the
SVM kernel distance and make gradient descent in the MLP and logistic
regression badly conditioned (slow or failed convergence).

Classes are weighted by inverse frequency (``class_weight="balanced"``) where
the estimator supports it, so the 20 % PASS minority is not ignored.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ColumnSlice(BaseEstimator, TransformerMixin):
    """Keep only the columns in ``[start, stop)``, which selects one feature group.

    Args:
        start: First column kept.
        stop: Column after the last one kept (None = to the end).
    """

    def __init__(self, start: int = 0, stop: int | None = None):
        self.start = start
        self.stop = stop

    def fit(self, X, y=None):
        """Do nothing: the selection has no learned state.

        Args:
            X: Feature matrix (ignored).
            y: Labels (ignored).

        Returns:
            This transformer, so it can be chained in a scikit-learn pipeline.
        """
        return self

    def transform(self, X):
        """Select the columns.

        Args:
            X: (n, d) feature matrix.

        Returns:
            (n, stop - start) array.
        """
        return np.asarray(X)[:, self.start : self.stop]


def make_model(name: str, colour_slice: slice = slice(0, 128), seed: int = 0) -> tuple[Pipeline, dict]:
    """Build an untrained model and its hyperparameter grid.

    Args:
        name: One of ``majority``, ``colour_lr``, ``logreg``, ``tree``, ``svm``, ``mlp``.
        colour_slice: Columns of the colour histogram (for ``colour_lr``).
        seed: Random seed for estimators that use one.

    Returns:
        ``(pipeline, param_grid)``; the grid is for `sklearn.model_selection.GridSearchCV`.

    Raises:
        ValueError: For an unknown name.
    """
    if name == "majority":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))]), {}
    if name == "colour_lr":
        pipe = Pipeline(
            [
                ("cols", ColumnSlice(colour_slice.start, colour_slice.stop)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=5000)),
            ]
        )
        return pipe, {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    if name == "logreg":
        pipe = Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]
        )
        return pipe, {"clf__C": [0.001, 0.01, 0.1, 1.0]}
    if name == "tree":
        pipe = Pipeline([("clf", DecisionTreeClassifier(class_weight="balanced", random_state=seed))])
        return pipe, {"clf__max_depth": [3, 5, 8, 12, None], "clf__min_samples_leaf": [1, 5, 20, 50]}
    if name == "svm":
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
            ]
        )
        return pipe, {"clf__C": [0.3, 1.0, 3.0, 10.0], "clf__gamma": ["scale", 1e-4, 3e-4]}
    if name == "mlp":
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        random_state=seed,
                    ),
                ),
            ]
        )
        return pipe, {"clf__hidden_layer_sizes": [(64,), (256,), (128, 64)], "clf__alpha": [1e-3, 1e-1, 1.0]}
    raise ValueError(f"unknown model {name!r}")


MODEL_NAMES: tuple[str, ...] = ("majority", "colour_lr", "logreg", "tree", "svm", "mlp")


def defect_score(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Defect score in [0, 1] for each row of ``X`` (FAIL = class 1).

    Uses ``predict_proba`` when available. The SVM has no probability output,
    so its signed distance to the boundary is passed through the logistic
    function. That keeps the ranking (so ROC AUC and threshold sweeps are
    unaffected) but the value is not a calibrated probability.

    Args:
        model: Fitted pipeline from `make_model`.
        X: (n, d) feature matrix.

    Returns:
        (n,) array of defect scores in [0, 1].
    """
    clf = model[-1]
    if hasattr(clf, "predict_proba"):
        classes = list(clf.classes_)
        P = model.predict_proba(X)
        # Column of the FAIL class (label 1). A model fitted on one class has no such column.
        return P[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    d = model.decision_function(X)
    # Logistic squashing keeps the SVM's ranking but maps it into (0, 1).
    return 1.0 / (1.0 + np.exp(-d))
