import numpy as np

def sobel_filter(image):
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

    height, width = image.shape
    result = np.zeros((height, width), dtype=np.float64)

    padded = np.pad(image.astype(float), 1, mode='edge')

    for i in range(height):
        for j in range(width):
            region = padded[i:i + 3, j:j + 3]

            x = np.sum(region * gx)
            y = np.sum(region * gy)

            result[i, j] = np.sqrt(x**2 + y**2)

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
