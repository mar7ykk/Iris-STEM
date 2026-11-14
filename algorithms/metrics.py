import numpy as np

def calculate_mse(filtered, original):
    """
    Обчислює середньоквадратичну помилку (MSE)
    між обробленим та оригінальним зображеннями.
    """
    difference = filtered.astype(float) - original.astype(float)
    return np.sum(difference ** 2) / original.size

def calculate_psnr(mse):
    """
    Обчислює пікове відношення сигналу до шуму (PSNR)
    за значенням середньоквадратичної помилки.
    """
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)
