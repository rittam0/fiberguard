"""Raw-feature baseline training and metrics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

LABELS = [0, 1, 2, 3]


def evaluate(model: Any, x: np.ndarray, y: np.ndarray) -> dict:
    started = time.perf_counter()
    predictions = model.predict(x)
    probabilities = model.predict_proba(x) if hasattr(model, "predict_proba") else None
    elapsed = time.perf_counter() - started
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=LABELS, zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y, predictions, labels=LABELS).tolist(),
        "prediction_seconds": elapsed,
    }
    if probabilities is not None:
        result["log_loss"] = float(log_loss(y, probabilities, labels=LABELS))
    return result


def fit_and_measure(model: Any, x: np.ndarray, y: np.ndarray) -> tuple[Any, float]:
    started = time.perf_counter()
    model.fit(x, y)
    return model, time.perf_counter() - started


def models(seed: int = 42) -> dict[str, Any]:
    return {
        "dummy": DummyClassifier(strategy="prior", random_state=seed),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, solver="lbfgs", tol=1e-4, random_state=seed),
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=4,
            random_state=seed,
            eval_metric="mlogloss",
            verbosity=1,
        ),
    }
