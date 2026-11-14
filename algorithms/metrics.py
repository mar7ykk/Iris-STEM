import numpy as np

def calculate_mse(filtered, original):
    difference = filtered.astype(float) - original.astype(float)
    return np.sum(difference ** 2) / original.size

def calculate_psnr(mse):
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)
