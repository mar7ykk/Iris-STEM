import numpy as np

def sobel_filter(image):
    """
    Виділяє контури зображення за допомогою оператора Собеля.

    image — двовимірний масив яскравостей пікселів.
    Повертає зображення з виділеними контурами.
    """
    # Ядра Собеля для горизонтального та вертикального градієнтів
    gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    gy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # Створюємо масив для результату
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.float64)

    # Додаємо рамку навколо зображення для обробки крайніх пікселів
    padded = np.pad(image.astype(float), 1, mode='edge')

    # Послідовно обробляємо кожен піксель
    for i in range(height):
        for j in range(width):
            region = padded[i:i + 3, j:j + 3]

            # Обчислюємо градієнти за двома напрямками
            x = np.sum(region * gx)
            y = np.sum(region * gy)

            # Обчислюємо величину градієнта
            result[i, j] = np.sqrt(x**2 + y**2)

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
