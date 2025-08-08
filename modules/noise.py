import numpy as np

def add_noise(coef, intercept, config):
    noise_type = config.get("noise", "none")
    scale = config.get("scale", 1.0)

    if noise_type == "gaussian":
        coef += np.random.normal(0, scale, size=coef.shape)
        intercept += np.random.normal(0, scale, size=intercept.shape)
    elif noise_type == "laplace":
        coef += np.random.laplace(0, scale, size=coef.shape)
        intercept += np.random.laplace(0, scale, size=intercept.shape)
    return coef, intercept
