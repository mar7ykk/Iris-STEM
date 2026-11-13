import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from algorithms.histogram import histogram_equalization
from algorithms.otsu import otsu_threshold
from algorithms.sobel import sobel_filter
from algorithms.lbp import lbp_texture
from algorithms.glcm import glcm_texture

# Знаходимо фотографії
folder = Path("data/images")

images = sorted([
    file for file in folder.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
])

if not images:
    raise ValueError("У папці data/images немає фотографій")

#перше фото
image_path = images[0]

original = Image.open(image_path).convert("RGB")
original.thumbnail((500, 500))

#відтінки сірого
gray = np.array(original.convert("L"))

# алгоритми
histogram = histogram_equalization(gray)
otsu, threshold = otsu_threshold(gray)
sobel = sobel_filter(gray)
lbp = lbp_texture(gray)
glcm, contrast, energy, homogeneity = glcm_texture(gray)

# Виводимо результати
print("Фотографія:", image_path.name)
print("Поріг Оцу:", threshold)
print("Контраст GLCM:", contrast)
print("Енергія GLCM:", energy)
print("Однорідність GLCM:", homogeneity)

# вікно з результатами
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

ax[0, 0].imshow(original)
ax[0, 0].set_title("Оригінал")

ax[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
ax[0, 1].set_title("Відтінки сірого")

ax[0, 2].imshow(histogram, cmap="gray", vmin=0, vmax=255)
ax[0, 2].set_title("Вирівнювання гістограми")

ax[1, 0].imshow(otsu, cmap="gray", vmin=0, vmax=255)
ax[1, 0].set_title("Метод Оцу")

ax[1, 1].imshow(sobel, cmap="gray", vmin=0, vmax=255)
ax[1, 1].set_title("Оператор Собеля")

ax[1, 2].imshow(lbp, cmap="gray", vmin=0, vmax=255)
ax[1, 2].set_title("LBP")

for a in ax.flat:
    a.axis("off")

plt.tight_layout()

# Зберігаю результат
Path("results").mkdir(exist_ok=True)
plt.savefig("results/first_image.png", dpi=150)

plt.show()