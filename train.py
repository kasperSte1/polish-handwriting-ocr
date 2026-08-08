from dataset import load_dataset, split_data
from model import build_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import random


def main():
    tf.keras.utils.set_random_seed(42)

    dataset_folder = "dataset/phcd/ocr_files"
    signs, labels, dictionary = load_dataset(dataset_folder)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(signs, labels)

    model = build_model(num_classes=89)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath="models/best_model.h5", monitor='val_loss', save_best_only=True)

    model.save('models/CNN.h5')
    history = model.fit(X_train,
                        y_train,
                        epochs=50,
                        batch_size=128,
                        validation_data=(X_val, y_val),
                        verbose=2,
                        callbacks=[early_stopping, model_checkpoint]
                        )
    h = history.history

    plt.subplot(2, 1, 1)
    plt.plot(h["loss"], label='Training Loss')
    plt.plot(h["val_loss"], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.subplot(2, 1, 2)
    plt.plot(h["accuracy"], label='Training Accuracy')
    plt.plot(h["val_accuracy"], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
