import numpy as np

def otsu_threshold(image):
    histogram = np.bincount(image.ravel(), minlength=256)

    total = image.size
    total_sum = np.sum(np.arange(256) * histogram)

    background_count = 0
    background_sum = 0

    max_variance = -1
    best_threshold = 0

    for t in range(256):
        background_count += histogram[t]
        background_sum += t * histogram[t]

        if background_count == 0:
            continue

        foreground_count = total - background_count

        if foreground_count == 0:
            break

        background_mean = background_sum / background_count
        foreground_mean = (total_sum - background_sum) / foreground_count

        variance = background_count * foreground_count * (
            background_mean - foreground_mean
        ) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    result = np.where(image <= best_threshold, 255, 0).astype(np.uint8)

    return result, best_threshold
