import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import csv

from algorithms.histogram import histogram_equalization
from algorithms.otsu import otsu_threshold
from algorithms.sobel import sobel_filter
from algorithms.lbp import lbp_texture
from algorithms.glcm import glcm_texture
from algorithms.iris_mask import detect_iris

# Знаходимо фотографії
folder = Path("data/images")

images = sorted([
    file for file in folder.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
])

if not images:
    raise ValueError("У папці data/images немає фотографій")

image_count = int(input("Скільки фотографій обробити? "))
# Обробка всі фотографії
results = []
for image_path in images[:image_count]:
    print("Обробляємо:", image_path.name)
    
    original = Image.open(image_path).convert("RGB")
    original.thumbnail((500, 500))

    #відтінки сірого
    gray = np.array(original.convert("L"))

    # алгоритми
    histogram = histogram_equalization(gray)
    mask, cx, cy, pupil_radius, iris_radius = detect_iris(gray)
    otsu, threshold = otsu_threshold(gray)
    sobel = sobel_filter(gray)
    lbp = lbp_texture(gray, mask)
    glcm, contrast, energy, homogeneity = glcm_texture(gray, mask=mask)

    # Виводимо результати
    print("Фотографія:", image_path.name)
    print("Поріг Оцу:", threshold)
    print("Контраст GLCM:", contrast)
    print("Енергія GLCM:", energy)
    print("Однорідність GLCM:", homogeneity)

    results.append({
        "Фотографія": image_path.name,
        "Поріг Оцу": threshold,
        "Контраст GLCM": round(contrast, 4),
        "Енергія GLCM": round(energy, 4),
        "Однорідність GLCM": round(homogeneity, 4)
    })

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
    result_folder = Path("results") / image_path.stem
    result_folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(result_folder / "processing.png", dpi=150)

    print("Центр райдужки:", cx, cy)
    print("Радіус зіниці:", pupil_radius)
    print("Радіус райдужки:", iris_radius)

    # Показуємо знайдені межі
    from matplotlib.patches import Circle

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(original)

    ax2.add_patch(Circle(
        (cx, cy), iris_radius,
        fill=False, color="lime", linewidth=2
    ))

    ax2.add_patch(Circle(
        (cx, cy), pupil_radius,
        fill=False, color="red", linewidth=2
    ))

    ax2.set_title("Автоматичне виділення райдужки")
    ax2.axis("off")

    plt.tight_layout()
    fig2.savefig(result_folder / "iris_detection.png", dpi=150)

    plt.close(fig)
    plt.close(fig2)

# Зберігаємо числові результати
with open("results/metrics.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys(), delimiter=";")
    writer.writeheader()
    writer.writerows(results)

print("Результати збережено у results/metrics.csv")
