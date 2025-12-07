import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


tf.random.set_seed(42)
np.random.seed(42)

MODEL_PATH = "digit_cnn.h5"


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Постройка и компиляция модели для классификации цифр
    :param input_shape: форма входного изображения
    :param num_classes: количество классов
    :return: скомпилированный объект модели
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_and_prepare_mnist():
    """
    Загрузка данных датасета и подготовка их к обучению
    :return: массивы изображений и меток для обучения
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test


def main():
    """
    Основной метод с подготовкой данных и созданием генератора
    """
    x_train, y_train, x_test, y_test = load_and_prepare_mnist()
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.08
    )
    datagen.fit(x_train)
    model = build_model()
    model.summary()
    batch_size = 128
    epochs = 12

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    model.save(MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Тестовая точность: {acc:.4f}, потеря: {loss:.4f}")


if __name__ == "__main__":
    main()
