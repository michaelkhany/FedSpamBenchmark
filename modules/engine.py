# modules/engine.py
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from diffprivlib.models import LogisticRegression as DPLogistic
from .privacy import gaussian_eps_from_sigma, laplace_eps_from_b, coord_sensitivity_per_client

def _calibrate_threshold(model, X_val, y_val):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_val)
    else:
        scores = model.predict_proba(X_val)[:, 1]
    p, r, thr = precision_recall_curve(y_val, scores)
    f1 = (2 * p * r) / np.maximum(p + r, 1e-12)
    best = np.nanargmax(f1)
    return 0.5 if best >= len(thr) else float(thr[best])

def _train_local(X_local, y_local, noise_config):
    use_dp = (noise_config.get("noise") == "diffprivlib")
    if use_dp:
        X_fit = X_local.toarray().astype(np.float64)
        counts = np.bincount(y_local)
        w = np.ones_like(y_local, dtype=float)
        if len(counts) == 2 and counts.min() > 0:
            inv = counts.sum() / (2.0 * counts)
            w = inv[y_local]
        model = DPLogistic(
            epsilon=noise_config.get("epsilon", 1.0),
            data_norm=1.0,
            max_iter=1000,
            tol=1e-4,
        )
        model.fit(X_fit, y_local, sample_weight=w)
    else:
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        )
        model.fit(X_local, y_local)
    return model

def federated_train(pack, noise_config, calibrate=True, clip_value=1.0, delta=1e-5):
    start = time.time()
    clients = pack["clients"]
    X_val, y_val = pack["X_val"], pack["y_val"]
    n_clients = len(clients)
    n_features = clients[0][0].shape[1]

    coef_sum = np.zeros(n_features, dtype=float)
    intercept_sum = 0.0

    # per-coordinate sensitivity after averaging
    S_coord = coord_sensitivity_per_client(clip_value, n_clients)

    for X_local, y_local in clients:
        model = _train_local(X_local, y_local, noise_config)
        coef = np.clip(model.coef_.flatten(), -clip_value, clip_value)
        intercept = model.intercept_[0]
        coef_sum += coef
        intercept_sum += intercept

    avg_coef = coef_sum / n_clients
    avg_intercept = intercept_sum / n_clients

    accounting = {"accounted": False}
    mech = noise_config.get("noise", "none")

    if mech == "gaussian":
        scale = float(noise_config.get("scale", 0.5))
        avg_coef = avg_coef + np.random.normal(0.0, scale, size=avg_coef.shape)
        avg_intercept = avg_intercept + float(np.random.normal(0.0, scale))
        eps_coord = gaussian_eps_from_sigma(scale, delta, S_coord)
        accounting = {
            "accounted": True,
            "mechanism": "gaussian",
            "epsilon_coord": float(eps_coord),
            "delta": float(delta),
            "clip_value": float(clip_value),
            "sensitivity_coord": float(S_coord),
        }
    elif mech == "laplace":
        scale = float(noise_config.get("scale", 0.5))
        avg_coef = avg_coef + np.random.laplace(0.0, scale, size=avg_coef.shape)
        avg_intercept = avg_intercept + float(np.random.laplace(0.0, scale))
        eps_coord = laplace_eps_from_b(scale, S_coord)
        accounting = {
            "accounted": True,
            "mechanism": "laplace",
            "epsilon_coord": float(eps_coord),
            "clip_value": float(clip_value),
            "sensitivity_coord": float(S_coord),
        }
    elif mech == "diffprivlib":
        accounting = {
            "accounted": True,
            "mechanism": "diffprivlib",
            "epsilon": float(noise_config.get("epsilon", 1.0)),
            "data_norm": 1.0,
        }

    final_model = LogisticRegression()
    final_model.coef_ = np.array([avg_coef])
    final_model.intercept_ = np.array([avg_intercept])
    final_model.classes_ = np.array([0, 1])
    final_model.n_features_in_ = n_features

    thr = _calibrate_threshold(final_model, X_val, y_val) if calibrate else 0.5
    runtime = round(time.time() - start, 2)

    return final_model, runtime, thr, accounting
