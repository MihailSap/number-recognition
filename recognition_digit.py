import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import os
import tkinter as tk
from tkinter import ttk
import argparse

MODEL_PATH = "digit_cnn.h5"


def get_resample_filter():
    """
    Возвращает доступный фильтр ресемплинга для Pillow с учётом версии.
    """
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", Image.BILINEAR))


def load_model(path=MODEL_PATH):
    """
    Загружает сохранённую модель Keras.
    :param path: путь к файлу модели
    :return: объект модели TensorFlow/Keras
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не найдена: {path}")
    return tf.keras.models.load_model(path)


def preprocess_image(path, target_size=(28, 28)):
    """
    Предобрабатывает изображение для подачи в модель:
    конвертация в градации серого, масштабирование, нормализация.
    :param path: путь к изображению
    :param target_size: целевой размер изображения
    :return: numpy-массив готовый для предсказания
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    else:
        img = img.convert("L")

    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), resample=get_resample_filter())

    img = img.resize(target_size, resample=get_resample_filter())
    arr = np.array(img).astype(np.float32)

    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0
    arr = np.where(arr > 0.12, arr, 0.0)
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, path):
    """
    Выполняет предсказание цифры на изображении.
    :param model: загруженная модель
    :param path: путь к изображению
    :return: распознанная цифра и уверенность
    """
    x = preprocess_image(path)
    preds = model.predict(x)
    digit = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return digit, confidence


def create_scrollable_canvas(root):
    """
    Создаёт прокручиваемый контейнер для отображения элементов.
    :param root: корневое окно tkinter
    :return: tuple(canvas, scroll_frame, scrollbar)
    """
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return canvas, scroll_frame, scrollbar


def add_image_tile(parent, photo, filename, digit, conf):
    """
    Добавляет плитку с изображением и подписью в родительский контейнер.
    :param parent: контейнер (Frame), куда добавлять
    :param photo: объект ImageTk.PhotoImage
    :param filename: имя файла изображения
    :param digit: предсказанная цифра
    :param conf: уверенность предсказания
    """
    frame = ttk.Frame(parent, padding=5)
    # Включаем pack внутри плитки: комбинация grid внешне и pack внутри допустима
    label_img = ttk.Label(frame, image=photo)
    label_img.pack()
    label_text = ttk.Label(frame, text=f"{filename}\nЦифра: {digit}, уверенность: {conf:.3f}")
    label_text.pack()
    return frame


def display_results(model, images_dir, grid_parent, max_per_row=5, thumb_size=(100, 100)):
    """
    Отображает изображения и результаты распознавания по сетке.
    :param model: загруженная модель
    :param images_dir: папка с изображениями
    :param grid_parent: контейнер, использующий grid для размещения
    :param max_per_row: максимум изображений в строке
    :param thumb_size: размер миниатюр (ширина, высота)
    :return: список ссылок на PhotoImage для предотвращения GC
    """
    image_refs = []
    row, col = 0, 0

    for filename in os.listdir(images_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(images_dir, filename)
        digit, conf = predict(model, path)

        img = Image.open(path)
        img.thumbnail(thumb_size)
        photo = ImageTk.PhotoImage(img)
        image_refs.append(photo)

        tile = add_image_tile(grid_parent, photo, filename, digit, conf)
        tile.grid(row=row, column=col, padx=10, pady=10)

        col += 1
        if col >= max_per_row:
            col = 0
            row += 1

    return image_refs


def run_app(images_dir, max_per_row=5):
    """
    Запускает GUI-приложение распознавания цифр.
    :param images_dir: папка с изображениями
    :param max_per_row: максимум изображений в строке
    """
    model = load_model()
    root = tk.Tk()
    root.title("Распознавание цифр")

    _, scroll_frame, _ = create_scrollable_canvas(root)
    root.image_refs = display_results(model, images_dir, scroll_frame, max_per_row=max_per_row)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Распознавание цифр в изображениях")
    parser.add_argument("images_dir", help="Папка с изображениями для распознавания")
    parser.add_argument("--per-row", type=int, default=5, help="Максимум изображений в строке (по умолчанию 5)")
    args = parser.parse_args()

    run_app(args.images_dir, max_per_row=args.per_row)
