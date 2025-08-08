# modules/privacy.py
import math

def gaussian_eps_from_sigma(sigma, delta, sensitivity):
    """
    Analytical bound (classic): σ >= sensitivity * sqrt(2 ln(1.25/delta)) / ε
    Rearranged to report ε for given σ.
    Assumes per-coordinate Gaussian mechanism with L2 sensitivity.
    """
    if sigma <= 0:
        return float("inf")
    return (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / sigma

def laplace_eps_from_b(scale_b, sensitivity):
    """
    Laplace mechanism: ε = sensitivity / b
    Assumes per-coordinate Laplace mechanism with L1 sensitivity per coordinate.
    """
    if scale_b <= 0:
        return float("inf")
    return sensitivity / scale_b

def coord_sensitivity_per_client(clip_value, n_clients):
    """
    We use per-coordinate clipping of each local coefficient to [-clip_value, +clip_value].
    Changing one client's vector can change any given coordinate by at most 2*clip_value.
    After averaging over n clients, per-coordinate sensitivity S = 2*clip_value / n.
    """
    return (2.0 * clip_value) / float(n_clients)
