"""Probability reliability and calibrated-confidence abstention utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

LABELS = np.arange(4)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean sum of squared probability error across all classes."""
    targets = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def reliability_bins(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> list[dict]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        count = int(mask.sum())
        if count:
            result.append({
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            })
    return result


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    total = len(y_true)
    return float(sum(item["count"] / total * abs(item["accuracy"] - item["mean_confidence"]) for item in reliability_bins(y_true, probabilities, bins)))


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=LABELS)),
        "multiclass_brier_score": multiclass_brier_score(y_true, probabilities),
        "ece_15_bin": expected_calibration_error(y_true, probabilities, bins=15),
    }


def select_review_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    max_review_rate: float = 0.05,
    target_error_capture: float = 0.75,
) -> tuple[float, dict]:
    """Select a validation-only max-probability threshold under a review cap."""
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    wrong = predictions != y_true
    total_errors = int(wrong.sum())
    candidates = []
    for threshold in np.linspace(0.50, 0.99, 50):
        review = confidence < threshold
        review_rate = float(review.mean())
        if review_rate <= max_review_rate:
            flagged = int((review & wrong).sum())
            candidates.append({
                "threshold": float(threshold),
                "review_rate": review_rate,
                "wrong_flagged": flagged,
                "error_capture_rate": float(flagged / total_errors) if total_errors else 1.0,
                "correct_flagged": int((review & ~wrong).sum()),
            })
    if not candidates:
        raise ValueError("no threshold satisfies the validation review-rate cap")
    meeting_target = [item for item in candidates if item["error_capture_rate"] >= target_error_capture]
    if meeting_target:
        selected = min(meeting_target, key=lambda item: (item["review_rate"], item["threshold"]))
    else:
        selected = max(candidates, key=lambda item: (item["error_capture_rate"], -item["review_rate"]))
    selected["selection_rule"] = f"minimum review rate capturing >= {target_error_capture:.0%} of validation errors, capped at {max_review_rate:.0%}; fallback maximizes capture under cap"
    selected["validation_total_errors"] = total_errors
    return selected["threshold"], selected


def abstention_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    review = confidence < threshold
    confident = ~review
    wrong = predictions != y_true
    if not confident.any():
        raise ValueError("threshold sends every prediction to review")
    return {
        "threshold": float(threshold),
        "review_rate": float(review.mean()),
        "review_count": int(review.sum()),
        "confident_count": int(confident.sum()),
        "confident_accuracy": float(accuracy_score(y_true[confident], predictions[confident])),
        "confident_macro_f1": float(f1_score(y_true[confident], predictions[confident], labels=LABELS, average="macro", zero_division=0)),
        "total_wrong_predictions": int(wrong.sum()),
        "wrong_predictions_flagged": int((wrong & review).sum()),
        "correct_predictions_flagged": int((~wrong & review).sum()),
        "remaining_wrong_confident_predictions": int((wrong & confident).sum()),
    }
