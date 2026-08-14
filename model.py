"""CNN architecture for 32x32 single-character classification (89 classes)."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Rescaling, BatchNormalization, \
    GlobalAveragePooling2D


def build_model(num_classes, input_shape=(32, 32, 1)):
    model = tf.keras.Sequential([
        # Rescaling as a layer, not preprocessing: normalisation is saved with
        # the model and cannot be forgotten at inference
        Rescaling(1 / 255, input_shape=input_shape),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  # "accuracy" string resolved to categorical_accuracy and reported chance-level values
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                  )

    return model


def main():
    model = build_model(num_classes=89)
    model.summary()


if __name__ == "__main__":
    main()
