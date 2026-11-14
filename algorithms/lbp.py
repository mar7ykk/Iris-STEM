import numpy as np

def lbp_texture(image, mask=None):
    """
    Обчислює локальні бінарні шаблони (LBP) зображення.

    image — двовимірний масив яскравостей пікселів.
    mask — необов'язкова маска області райдужки.
    Повертає зображення з LBP-кодами пікселів.
    """
    # Створюємо порожнє зображення для LBP-кодів
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)

    # Перебираємо пікселі, крім крайніх
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Перевіряємо, чи всі пікселі околу належать масці
            if mask is not None:
                if not np.all(mask[i-1:i+2, j-1:j+2]):
                    continue
            # Беремо центральний піксель та його 8 сусідів
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
            # Формуємо LBP-код порівнянням сусідів із центром
            value = 0

            for k in range(8):
                if neighbors[k] >= center:
                    value += 2 ** k

            result[i, j] = value

    return result
