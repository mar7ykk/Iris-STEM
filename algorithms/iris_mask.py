
import numpy as np


def detect_iris(image):
    """
    Створює приблизну кільцеву маску райдужки.

    image — двовимірне зображення у відтінках сірого.
    Повертає маску, координати центра та радіуси зіниці й райдужки.
    """
    # Визначаємо розміри зображення та його центр
    height, width = image.shape

    cx = width // 2
    cy = height // 2

    # Обчислюємо приблизні радіуси зіниці та райдужки
    size = min(height, width)

    pupil_radius = int(size * 0.13)
    iris_radius = int(size * 0.44)

    # Створюємо координати всіх пікселів
    y, x = np.indices((height, width))

    # Обчислюємо відстань кожного пікселя до центра
    distance = np.sqrt(
        (x - cx) ** 2 + (y - cy) ** 2
    )

    # Формуємо кільцеву маску між зіницею та зовнішньою межею райдужки
    mask = (
        (distance >= pupil_radius) &
        (distance <= iris_radius)
    )

    return mask, cx, cy, pupil_radius, iris_radius
