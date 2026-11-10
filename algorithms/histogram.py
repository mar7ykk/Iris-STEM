import numpy as np


def histogram_equalization(image):
    # Кількість пікселів кожного рівня яскравості
    histogram = np.bincount(image.ravel(), minlength=256)

    # Накопичена гістограма
    cdf = np.cumsum(histogram)

    # Беремо тільки ненульові значення
    cdf_min = cdf[cdf > 0][0]

    # Загальна кількість пікселів
    total = image.size

    # Якщо всі пікселі однакові
    if total == cdf_min:
        return image.copy()

    # Формула вирівнювання гістограми
    new_values = (cdf - cdf_min) / (total - cdf_min) * 255

    # Обмежуємо значення від 0 до 255
    new_values = np.clip(new_values, 0, 255).astype(np.uint8)

    # Замінюємо старі значення пікселів новими
    result = new_values[image]

    return result
