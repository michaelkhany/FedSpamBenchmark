# evaluator.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

def _predict_with_threshold(model, X, thr):
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
    else:
        s = model.predict_proba(X)[:, 1]
    return (s >= thr).astype(int)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred = _predict_with_threshold(model, X_test, threshold)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "threshold": float(threshold),
    }

    # extras for analysis
    try:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            scores = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = round(roc_auc_score(y_test, scores), 4)
        metrics["pr_auc"] = round(average_precision_score(y_test, scores), 4)
    except Exception:
        pass

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    return metrics
