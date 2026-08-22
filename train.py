"""Train the character classifier on PHCD.

Callbacks: EarlyStopping restores the best weights, ModelCheckpoint keeps the
best model on disk, ReduceLROnPlateau halves the learning rate on plateau.
"""

from dataset import load_dataset, split_data, add_channel_dim, augment_images
from model import build_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

EPOCHS = 50
BATCH_SIZE = 256
SEED = 42
USE_AUGMENTATION = True

DATASET_DIR = "dataset/phcd/ocr_files"
FINAL_MODEL_PATH = "models/CNN.h5"
CHECKPOINT_PATH = "models/best_model.h5"
FIGURE_PATH = "figures/training_history.png"


def plot_history(history_dict, path):
    """Plot training history against epochs."""

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1)
    ax_loss.plot(history_dict["loss"], label='Training Loss')
    ax_loss.plot(history_dict["val_loss"], label='Validation Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training and Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(history_dict["accuracy"], label='Training Accuracy')
    ax_acc.plot(history_dict["val_accuracy"], label='Validation Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Training and Validation Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    tf.keras.utils.set_random_seed(SEED)

    signs, labels, dictionary = load_dataset(DATASET_DIR)
    X_train, X_val, _, y_train, y_val, _ = split_data(signs, labels)
    if USE_AUGMENTATION:
        X_train, y_train = augment_images(X_train, y_train)
    X_train, X_val = add_channel_dim(X_train, X_val)

    model = build_model(num_classes=len(dictionary))

    # EarlyStopping must be more patient than ReduceLROnPlateau (5 > 2), otherwise
    # training stops before the reduced learning rate has a chance to help
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(patience=2, verbose=1, mode='auto', factor=0.5, cooldown=0, min_lr=0)

    history = model.fit(X_train,
                        y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val),
                        verbose=1,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr]
                        )

    model.save(FINAL_MODEL_PATH)
    plot_history(history.history, FIGURE_PATH)


if __name__ == "__main__":
    main()
