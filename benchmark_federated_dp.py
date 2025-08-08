# benchmark_federated_dp.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from modules.data_handler import load_and_split_data
from modules.evaluator import evaluate_model

from modules.engine import federated_train
import inspect, sys
print("DEBUG federated_train from:", federated_train.__module__)
print("DEBUG file:", inspect.getsourcefile(federated_train))
print("DEBUG signature:", inspect.signature(federated_train))


OUT_JSON = "benchmark_results.json"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Sweep configs
gaussian_scales = [0.25, 0.5, 0.75, 1.0]
laplace_scales = [0.25, 0.5, 0.75, 1.0]
dp_epsilons = [1, 2, 5, 10, 20, 30, 50]

# Base pack
pack = load_and_split_data(n_clients=5, test_size=0.2, val_size=0.1)

# Common train args
clip_value = 1.0
delta = 1e-5

configs = [{"noise": "none"}]
configs += [{"noise": "gaussian", "scale": s} for s in gaussian_scales]
configs += [{"noise": "laplace", "scale": s} for s in laplace_scales]
configs += [{"noise": "diffprivlib", "epsilon": e} for e in dp_epsilons]

results = []
for cfg in configs:
    model, runtime, thr, acct = federated_train(
        pack, noise_config=cfg, calibrate=True, clip_value=clip_value, delta=delta
    )
    metrics = evaluate_model(model, pack["X_test"], pack["y_test"], threshold=thr)
    row = {**cfg, **metrics, "runtime": runtime}
    if acct.get("accounted", False):
        row.update({"privacy": acct})
    results.append(row)
    print(row)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {OUT_JSON}")

# ------------- Plotting -------------
def extract_series(results, kind):
    xs, f1s, accs, precs, recs = [], [], [], [], []
    for r in results:
        if kind == "gaussian" and r.get("noise") == "gaussian":
            xs.append(r["privacy"]["epsilon_coord"])
        elif kind == "laplace" and r.get("noise") == "laplace":
            xs.append(r["privacy"]["epsilon_coord"])
        elif kind == "dp" and r.get("noise") == "diffprivlib":
            xs.append(r["privacy"]["epsilon"])
        else:
            continue
        f1s.append(r["f1_score"])
        accs.append(r["accuracy"])
        precs.append(r["precision"])
        recs.append(r["recall"])
    # Sort by x
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(f1s)[order], np.array(accs)[order], np.array(precs)[order], np.array(recs)[order]

def plot_privacy_utility(xs, f1s, accs, precs, recs, title, filename):
    plt.figure()
    plt.plot(xs, f1s, marker="o", label="F1")
    plt.plot(xs, accs, marker="o", label="Accuracy")
    plt.plot(xs, precs, marker="o", label="Precision")
    plt.plot(xs, recs, marker="o", label="Recall")
    plt.xlabel("Epsilon")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved plot: {path}")

# Gaussian curve (epsilon is computed from sigma via accounting)
gxs, gf1, gacc, gpre, grec = extract_series(results, "gaussian")
if len(gxs) > 0:
    plot_privacy_utility(gxs, gf1, gacc, gpre, grec, "Gaussian mechanism: privacy-utility", "gaussian_privacy_utility.png")

# Laplace curve
lxs, lf1, lacc, lpre, lrec = extract_series(results, "laplace")
if len(lxs) > 0:
    plot_privacy_utility(lxs, lf1, lacc, lpre, lrec, "Laplace mechanism: privacy-utility", "laplace_privacy_utility.png")

# DP Logistic sweep
dxs, df1, dacc, dpre, drec = extract_series(results, "dp")
if len(dxs) > 0:
    plot_privacy_utility(dxs, df1, dacc, dpre, drec, "DP Logistic Regression: epsilon sweep", "dp_privacy_utility.png")
