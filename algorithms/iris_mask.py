
import numpy as np


def detect_iris(image):
    height, width = image.shape

    cx = width // 2
    cy = height // 2

    size = min(height, width)

    pupil_radius = int(size * 0.13)
    iris_radius = int(size * 0.44)

    y, x = np.indices((height, width))

    distance = np.sqrt(
        (x - cx) ** 2 + (y - cy) ** 2
    )

    mask = (
        (distance >= pupil_radius) &
        (distance <= iris_radius)
    )

    return mask, cx, cy, pupil_radius, iris_radius
