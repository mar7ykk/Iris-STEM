import numpy as np

def glcm_texture(image, levels=16, mask=None):
    """
    Аналізує текстуру зображення за допомогою матриці GLCM.

    image — двовимірний масив яскравостей пікселів.
    levels — кількість рівнів яскравості.
    mask — необов'язкова маска області райдужки.
    Повертає матрицю GLCM, контраст, енергію та однорідність.
    """
    height, width = image.shape

    # Зменшуємо кількість рівнів яскравості
    gray = image.astype(np.int32) * levels // 256

    # Створюємо матрицю спільної зустрічальності
    matrix = np.zeros((levels, levels), dtype=np.float64)

    # Підраховуємо пари сусідніх пікселів по горизонталі
    for i in range(height):
        for j in range(width - 1):
            if mask is not None:
                if not (mask[i, j] and mask[i, j + 1]):
                    continue
            a = gray[i, j]
            b = gray[i, j + 1]

            matrix[a, b] += 1

    # Нормалізуємо матрицю
    total = np.sum(matrix)

    if total > 0:
        matrix = matrix / total

    # Обчислюємо текстурні характеристики
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
