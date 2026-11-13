import numpy as np

def glcm_texture(image, levels=16):
    height, width = image.shape

    gray = image.astype(np.int32) * levels // 256

    matrix = np.zeros((levels, levels), dtype=np.float64)

    for i in range(height):
        for j in range(width - 1):
            a = gray[i, j]
            b = gray[i, j + 1]

            matrix[a, b] += 1

    total = np.sum(matrix)

    if total > 0:
        matrix = matrix / total

    contrast = 0
    energy = 0
    homogeneity = 0

    for i in range(levels):
        for j in range(levels):
            p = matrix[i, j]

            contrast += (i - j) ** 2 * p
            energy += p ** 2
            homogeneity += p / (1 + (i - j) ** 2)

    return matrix, contrast, energy, homogeneity
