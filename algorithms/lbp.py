import numpy as np

def lbp_texture(image, mask=None):
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if mask is not None:
                if not np.all(mask[i-1:i+2, j-1:j+2]):
                    continue
            center = image[i, j]

            neighbors = [
                image[i-1, j-1],
                image[i-1, j],
                image[i-1, j+1],
                image[i, j+1],
                image[i+1, j+1],
                image[i+1, j],
                image[i+1, j-1],
                image[i, j-1]
            ]

            value = 0

            for k in range(8):
                if neighbors[k] >= center:
                    value += 2 ** k

            result[i, j] = value

    return result
