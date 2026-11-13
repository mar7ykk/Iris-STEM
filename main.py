import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import csv
import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

from algorithms.histogram import histogram_equalization
from algorithms.otsu import otsu_threshold
from algorithms.sobel import sobel_filter
from algorithms.lbp import lbp_texture
from algorithms.glcm import glcm_texture
from algorithms.iris_mask import detect_iris

# Зчитуємо налаштування програми
with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

# Методи, вибрані для обробки
methods = config["methods"]

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
    original.thumbnail((config["image_size"], config["image_size"]))

    #відтінки сірого
    gray = np.array(original.convert("L"))

    mask, cx, cy, pupil_radius, iris_radius = detect_iris(gray)
    # алгоритми
    histogram = histogram_equalization(gray) if "histogram" in methods else None
    otsu, threshold = otsu_threshold(gray)  if "otsu" in methods else (None, None)
    sobel = sobel_filter(gray) if "sobel" in methods else None
    lbp = lbp_texture(gray, mask) if "lbp" in methods else None
    glcm, contrast, energy, homogeneity = glcm_texture(gray, mask=mask) if "glcm" in methods else (None, None, None, None)

    # Виводимо результати
    print("Фотографія:", image_path.name)
    print("Поріг Оцу:", threshold)
    print("Контраст GLCM:", contrast)
    print("Енергія GLCM:", energy)
    print("Однорідність GLCM:", homogeneity)

    results.append({
        "Фотографія": image_path.name,
        "Оброблене фото": "Відкрити",
        "Гістограма": "Відкрити" if histogram is not None else "",
        "Оцу": "Відкрити" if otsu is not None else "",
        "Собель": "Відкрити" if sobel is not None else "",
        "LBP": "Відкрити" if lbp is not None else "",
        "Поріг Оцу": threshold,
        "Контраст GLCM": round(contrast, 4) if contrast is not None else None,
        "Енергія GLCM": round(energy, 4) if energy is not None else None,
        "Однорідність GLCM": round(homogeneity, 4) if homogeneity is not None else None
    })

    # вікно з результатами
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    ax[0, 0].imshow(original)
    ax[0, 0].set_title("Оригінал")

    ax[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title("Відтінки сірого")

    if histogram is not None:
        ax[0, 2].imshow(histogram, cmap="gray", vmin=0, vmax=255)
        ax[0, 2].set_title("Вирівнювання гістограми")
    else:
        ax[0, 2].set_visible(False)

    if otsu is not None:
        ax[1, 0].imshow(otsu, cmap="gray", vmin=0, vmax=255)
        ax[1, 0].set_title("Метод Оцу")
    else:
        ax[1, 0].set_visible(False)

    if sobel is not None:
        ax[1, 1].imshow(sobel, cmap="gray", vmin=0, vmax=255)
        ax[1, 1].set_title("Оператор Собеля")
    else:
        ax[1, 1].set_visible(False)

    if lbp is not None: 
        ax[1, 2].imshow(lbp, cmap="gray", vmin=0, vmax=255)
        ax[1, 2].set_title("LBP")
    else:
        ax[1, 2].set_visible(False)

    for a in ax.flat:
        a.axis("off")

    plt.tight_layout()

    # Зберігаю результат
    result_folder = Path("results") / image_path.stem
    result_folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(result_folder / "processing.png", dpi=150)
        # Зберігаємо результати кожного алгоритму окремо
    if histogram is not None:
        Image.fromarray(histogram).save(result_folder / "histogram.png")

    if otsu is not None:
        Image.fromarray(otsu).save(result_folder / "otsu.png")

    if sobel is not None:
        Image.fromarray(sobel).save(result_folder / "sobel.png")

    if lbp is not None:
        Image.fromarray(lbp).save(result_folder / "lbp.png")

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
# Створюємо Excel-таблицю
wb = Workbook()
ws = wb.active
ws.title = "Аналіз райдужки"

# Заголовки таблиці
ws.append(list(results[0].keys()))

# Записуємо результати
for result in results:
    ws.append(list(result.values()))

# Посилання на результати алгоритмів
files = {
    2: "processing.png",
    3: "histogram.png",
    4: "otsu.png",
    5: "sobel.png",
    6: "lbp.png"
} 

for row in range(2, ws.max_row + 1):
    image_name = Path(ws.cell(row, 1).value).stem

    for column, filename in files.items():
        cell = ws.cell(row, column)

        if cell.value == "Відкрити":
            cell.hyperlink = f"{image_name}/{filename}"
            cell.style = "Hyperlink"

# Оформлення заголовків
for cell in ws[1]:
    cell.font = Font(color="FFFFFF", bold=True, size=12)
    cell.fill = PatternFill("solid", fgColor="243B55")
    cell.alignment = Alignment(horizontal="center", vertical="center")

# Ширина стовпців
for column in "BCDEF":
    ws.column_dimensions[column].width = 20

ws.column_dimensions["A"].width = 25

for column in "GHIJ":
    ws.column_dimensions[column].width = 23

wb.save("results/metrics.xlsx")

print("Excel-таблицю збережено у results/metrics.xlsx")
