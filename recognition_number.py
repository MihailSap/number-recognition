import os
import argparse
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
from tkinter import ttk

MODEL_PATH = "digit_cnn.h5"


def load_model(path=MODEL_PATH):
    """
    Загружает сохранённую модель Keras.
    :param path: путь к файлу модели
    :return: объект модели TensorFlow/Keras
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не найдена: {path}")
    return tf.keras.models.load_model(path)


def preprocess_for_seg(img):
    """
    Готовит изображение к сегментации цифр:
    перевод в серый, размытие, порог, инверсия при необходимости и морфологию.
    :param img: исходное BGR-изображение (numpy)
    :return: бинарное изображение для поиска контуров
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = 255 - th
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def extract_digit_rois(bin_img, min_area=50):
    """
    Находит прямоугольники цифр на бинарном изображении и сортирует слева направо.
    :param bin_img: бинарное изображение
    :param min_area: минимальная площадь для фильтрации мелких шумов
    :return: список боксов (x, y, w, h) в порядке слева направо
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes


def roi_to_model_input(bin_img, box, target_size=(28, 28)):
    """
    Преобразует ROI цифры к формату входа модели:
    паддинг, центрирование в квадрате, ресайз, нормализация.
    :param bin_img: бинарное изображение
    :param box: (x, y, w, h) прямоугольник цифры
    :param target_size: размер входа модели
    :return: батч (1, H, W, 1) numpy.float32
    """
    x, y, w, h = box
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bin_img.shape[1], x + w + pad)
    y1 = min(bin_img.shape[0], y + h + pad)
    roi = bin_img[y0:y1, x0:x1]

    h2, w2 = roi.shape
    size = max(h2, w2)
    square = np.zeros((size, size), dtype=roi.dtype)
    y_off = (size - h2) // 2
    x_off = (size - w2) // 2
    square[y_off:y_off + h2, x_off:x_off + w2] = roi

    square = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)
    arr = square.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)


def predict_number(model, image_path):
    """
    Распознаёт последовательность цифр на изображении.
    :param model: загруженная модель
    :param image_path: путь к изображению
    :return: (строка-число, список уверенностей по каждой цифре)
    """
    img = cv2.imread(image_path)
    if img is None:
        return "", []

    bin_img = preprocess_for_seg(img)
    boxes = extract_digit_rois(bin_img)
    digits, confidences = [], []

    for b in boxes:
        x = roi_to_model_input(bin_img, b)
        preds = model.predict(x)
        d = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))
        digits.append(str(d))
        confidences.append(conf)

    number = "".join(digits) if digits else ""
    return number, confidences


def create_scrollable_canvas(root):
    """
    Создаёт прокручиваемый контейнер для сетки элементов.
    :param root: корневое окно tkinter
    :return: (canvas, scroll_frame, scrollbar)
    """
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return canvas, scroll_frame, scrollbar


def add_image_tile(parent, photo, filename, number, confidences):
    """
    Добавляет плитку с миниатюрой и результатами распознавания.
    :param parent: родительский контейнер (Frame)
    :param photo: ImageTk.PhotoImage миниатюра
    :param filename: имя файла
    :param number: распознанное число (строка)
    :param confidences: список уверенностей по каждой цифре
    :return: созданный Frame-плитка
    """
    frame = ttk.Frame(parent, padding=5)

    lbl_img = ttk.Label(frame, image=photo)
    lbl_img.pack()

    conf_str = ", ".join(f"{c:.3f}" for c in confidences) if confidences else "-"
    text = (
        f"{filename}\n"
        f"Число: {number if number else '(не распознано)'}\n"
        f"Уверенности: {conf_str}"
    )
    lbl_text = ttk.Label(frame, text=text)
    lbl_text.pack()

    return frame


def display_results(images_dir, grid_parent, model, max_per_row=5, thumb_size=(140, 140)):
    """
    Отображает миниатюры и результаты распознавания чисел по сетке.
    :param images_dir: папка с изображениями
    :param grid_parent: контейнер для grid
    :param model: загруженная модель
    :param max_per_row: максимум плиток в строке
    :param thumb_size: размер миниатюры (w, h)
    :return: список ссылок на PhotoImage (чтобы не удалились GC)
    """
    image_refs = []
    row, col = 0, 0

    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(images_dir, filename)
        number, confidences = predict_number(model, path)

        try:
            pil_img = Image.open(path)
            pil_img.thumbnail(thumb_size)
            photo = ImageTk.PhotoImage(pil_img)
        except Exception:
            pil_img = Image.new("RGB", thumb_size, color=(220, 220, 220))
            photo = ImageTk.PhotoImage(pil_img)

        image_refs.append(photo)

        tile = add_image_tile(grid_parent, photo, filename, number, confidences)
        tile.grid(row=row, column=col, padx=10, pady=10)

        col += 1
        if col >= max_per_row:
            col = 0
            row += 1

    return image_refs


def run_app(images_dir, max_per_row=5):
    """
    Запускает GUI-приложение распознавания чисел.
    :param images_dir: папка с изображениями
    :param max_per_row: максимум плиток в строке
    """
    model = load_model()
    root = tk.Tk()
    root.title("Распознавание чисел")

    _, scroll_frame, _ = create_scrollable_canvas(root)
    root.image_refs = display_results(images_dir, scroll_frame, model, max_per_row=max_per_row)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Распознавание чисел в изображениях")
    parser.add_argument("images_dir", help="Папка с изображениями для распознавания")
    parser.add_argument("--per-row", type=int, default=5, help="Максимум изображений в строке (по умолчанию 5)")
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        raise NotADirectoryError(f"Папка не найдена: {args.images_dir}")

    run_app(args.images_dir, max_per_row=args.per_row)
