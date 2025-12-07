from __future__ import annotations
import argparse
import os
from typing import Iterable, Tuple, Optional
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def signal_power(img: np.ndarray) -> float:
    """
    Вычислить среднюю мощность сигнала для изображения.
    Ожидается, что img в диапазоне [0,1].
    """
    return float(np.mean(np.square(img)))


def noise_sigma_for_snr_db(img: np.ndarray, snr_db: float) -> float:
    """
    Для данного изображения и требуемого SNR (в дБ) вычислить sigma шума.
    """
    p_signal = signal_power(img)
    if p_signal <= 0.0:
        return 0.0
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(p_noise)
    return float(sigma)


def add_gaussian_noise_for_snr(img: np.ndarray, snr_db: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Добавить гауссов шум к изображению так, чтобы получился заданный SNR (в дБ).
    img: numpy array dtype float32, range [0,1], shape (H,W) или (H,W,1)
    Возвращает зашумлённое изображение в том же формате, клипированное в [0,1].
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = img.copy().astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr2 = arr[..., 0]
    else:
        arr2 = arr
    sigma = noise_sigma_for_snr_db(arr2, snr_db)
    if sigma == 0.0:
        noisy = arr2
    else:
        noise = rng.normal(loc=0.0, scale=sigma, size=arr2.shape).astype(np.float32)
        noisy = arr2 + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[2] == 1:
        noisy = noisy[..., np.newaxis]
    return noisy.astype(np.float32)


def load_mnist_test() -> Tuple[np.ndarray, np.ndarray]:
    """
    Загрузить MNIST (только тестовую часть), вернуть x (N,28,28,1) и y (N,).
    x нормализовано в [0,1], dtype float32.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


def load_images_from_folder(folder: str, target_size: Tuple[int, int] = (28, 28)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загрузить изображения из папки с подкаталогами по меткам.
    Ожидается структура: folder/<label>/<image.png>
    Возвращает x (N,H,W,1) dtype float32 в [0,1] и y (N,) метки int.
    """
    images = []
    labels = []
    for label_name in sorted(os.listdir(folder)):
        label_path = os.path.join(folder, label_name)
        if not os.path.isdir(label_path):
            continue
        try:
            label = int(label_name)
        except ValueError:
            continue
        for fname in os.listdir(label_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            p = os.path.join(label_path, fname)
            try:
                img = Image.open(p).convert('L').resize(target_size, resample=Image.BILINEAR)
                arr = np.array(img).astype(np.float32) / 255.0
                arr = np.expand_dims(arr, axis=-1)
                images.append(arr)
                labels.append(label)
            except Exception:
                continue
    if len(images) == 0:
        raise RuntimeError(f"No images found in folder: {folder}")
    x = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return x, y


def evaluate_model_on_noisy_set(model: tf.keras.Model,
                                x: np.ndarray,
                                y: np.ndarray,
                                snr_db: float,
                                samples: Optional[int] = None,
                                trials: int = 1,
                                rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    Оценить accuracy и среднюю уверенность модели на наборе x,y при заданном SNR.
    - samples: если указано, случайно выбрать не более samples изображений из x.
    - trials: для каждого изображения выполнить trials независимых реализаций шума и усреднить результаты.
    Возвращает (accuracy, mean_confidence).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = x.shape[0]
    idx = np.arange(n)
    if samples is not None and samples < n:
        idx = rng.choice(idx, size=samples, replace=False)
    total = 0
    correct = 0
    confidences = []
    for i in idx:
        img = x[i]
        label = int(y[i])
        for t in range(trials):
            noisy = add_gaussian_noise_for_snr(img, snr_db, rng=rng)
            inp = np.expand_dims(noisy[..., 0] if noisy.ndim==3 and noisy.shape[2]==1 else noisy, axis=0)
            if inp.ndim == 3:
                inp = np.expand_dims(inp, axis=-1)
            preds = model.predict(inp, verbose=0)
            pred_label = int(np.argmax(preds, axis=1)[0])
            conf = float(np.max(preds))
            total += 1
            if pred_label == label:
                correct += 1
            confidences.append(conf)
    accuracy = correct / total if total > 0 else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return accuracy, mean_conf


def run_experiment(model_path: str,
                   dataset: str,
                   data_folder: Optional[str],
                   snr_db_list: Iterable[float],
                   samples_per_level: int,
                   trials: int,
                   out_csv: Optional[str],
                   plot_path: Optional[str],
                   seed: int) -> None:
    """
    Запустить эксперимент по списку SNR значений.
    Сохраняет CSV и строит график, если указаны пути.
    """
    rng = np.random.default_rng(seed)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)

    if dataset == 'mnist':
        x, y = load_mnist_test()
    elif dataset == 'folder':
        if data_folder is None:
            raise ValueError("data_folder must be provided when dataset='folder'")
        x, y = load_images_from_folder(data_folder)
    else:
        raise ValueError("dataset must be 'mnist' or 'folder'")

    samples_per_level = min(samples_per_level, x.shape[0])

    results = []
    for snr_db in tqdm(list(snr_db_list), desc="SNR levels"):
        acc, mean_conf = evaluate_model_on_noisy_set(
            model=model,
            x=x,
            y=y,
            snr_db=float(snr_db),
            samples=samples_per_level,
            trials=trials,
            rng=rng
        )
        results.append({'snr_db': float(snr_db), 'accuracy': float(acc), 'mean_confidence': float(mean_conf)})
        print(f"SNR={snr_db} dB -> accuracy={acc:.4f}, mean_conf={mean_conf:.4f}")

    df = pd.DataFrame(results)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")

    if plot_path:
        plt.figure(figsize=(8,5))
        plt.plot(df['snr_db'], df['accuracy'], marker='o', label='Accuracy', color='blue')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.title('Accuracy vs SNR')
        plt.ylim(-0.02, 1.02)
        plt.gca().invert_xaxis()
        plt.legend()
        plt.savefig(plot_path.replace('.png', '_accuracy.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Accuracy plot saved to {plot_path.replace('.png', '_accuracy.png')}")

        plt.figure(figsize=(8,5))
        plt.plot(df['snr_db'], df['mean_confidence'], marker='s', label='Mean Confidence', color='green')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Mean Confidence')
        plt.grid(True)
        plt.title('Mean Confidence vs SNR')
        plt.ylim(-0.02, 1.02)
        plt.gca().invert_xaxis()
        plt.legend()
        plt.savefig(plot_path.replace('.png', '_confidence.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Confidence plot saved to {plot_path.replace('.png', '_confidence.png')}")

    print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment: accuracy vs SNR for digit recognition")
    p.add_argument('--model', type=str, default='digit_cnn.h5', help='Path to model file')
    p.add_argument('--dataset', type=str, choices=['mnist', 'folder'], default='mnist', help='Dataset to use')
    p.add_argument('--data_folder', type=str, default=None, help='If dataset=folder, path to labeled images folder')
    p.add_argument('--snr_db', type=float, nargs='+', default=[-10, -5, 0, 5, 10, 15, 20, 25, 30],
                   help='List of SNR values in dB to test')
    p.add_argument('--samples_per_level', type=int, default=1000, help='Number of images to sample per SNR level')
    p.add_argument('--trials', type=int, default=1, help='Number of noise realizations per image')
    p.add_argument('--out', type=str, default='results_snr.csv', help='CSV output path')
    p.add_argument('--plot', type=str, default='accuracy_vs_snr.png', help='Plot output path (png)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(
        model_path=args.model,
        dataset=args.dataset,
        data_folder=args.data_folder,
        snr_db_list=args.snr_db,
        samples_per_level=args.samples_per_level,
        trials=args.trials,
        out_csv=args.out,
        plot_path=args.plot,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
