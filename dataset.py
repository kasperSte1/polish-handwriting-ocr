"""Loading, splitting and augmenting the PHCD character dataset.

PHCD stores 32x32 glyphs with background = 255 and ink = 0, each scaled to the
full image height — a detail that matters both for augmentation and for any
preprocessing of new input.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_dataset(dataset_folder):
    """Load the PHCD dataset. Format: see documentation.pdf, ch. 7.

    Returns:
    signs: uint8 (N, 32, 32), background = 255, ink = 0
    labels: uint8 (N,), character codes 0-88
    dictionary: {code: character}
    """

    folder = Path(dataset_folder)
    signs = np.load(folder / "signs.npy").astype("uint8")
    labels = np.load(folder / "labels_int.npy")

    with open(folder / "dictionary.json", encoding="cp1250") as f:
        dictionary = json.load(f)
    dictionary = {int(k): v for k, v in dictionary.items()}

    return signs, labels, dictionary


def build_char_to_code(dictionary):
    """Reverse the PHCD dictionary: {code: char} -> {char: code}"""

    return {char: code for code, char in dictionary.items()}


def describe_dataset(signs, labels, dictionary):
    print("Signs shape: ", signs.shape, "\n Labels shape: ", labels.shape,
          "\n Dictionary length: ", len(dictionary))

    classes, counts = np.unique(labels, return_counts=True)
    print("Classes: ", classes.min(), "-", classes.max())
    print("Samples per class: min = ", counts.min(), "max=", counts.max(),
          "median = ", np.median(counts))


def show_samples(signs, labels, dictionary, n=12, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.choice(signs.shape[0], n, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))

    for ax, i in zip(axes.flat, index):
        ax.imshow(signs[i], cmap="gray")
        ax.set_title(dictionary[labels[i]])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def add_channel_dim(*arrays):
    """Add a trailing channel dimension: (N, 32, 32) -> (N, 32, 32, 1).
    Keras Conv2D expects channels-last input. Takes any number of image
    arrays, returns them in the same order.
    """

    return tuple(np.expand_dims(a, axis=-1) for a in arrays)


def split_data(signs, labels, test_size=0.15, val_size=0.15, seed=42):
    """Stratified 70/15/15 split (train, val, test)"""

    X_temp, X_test, y_temp, y_test = train_test_split(signs,
                                                      labels,
                                                      test_size=test_size,
                                                      random_state=seed,
                                                      stratify=labels
                                                      )
    # second split runs on the remainder, ratio needs rescaling
    val_size_of_remainder = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(X_temp,
                                                      y_temp,
                                                      test_size=val_size_of_remainder,
                                                      random_state=seed,
                                                      stratify=y_temp
                                                      )
    return X_train, X_val, X_test, y_train, y_val, y_test


def augment_images(images, labels, n_copies=1, max_angle=10, seed=42):
    """Return originals plus n_copies randomly rotated variants of each image.

    scale=0.9 is not cosmetic: 98.9% of PHCD characters touch the bottom edge
    (each glyph is scaled to the full 32px height), so rotating without
    shrinking would clip the descenders that distinguish a/ą and e/ę.
    """

    augmented_images = []
    height, width = images[0].shape[:2]
    center = (width // 2, height // 2)
    rng = np.random.default_rng(seed)
    augmented_labels = []
    for _ in range(n_copies):
        for i in range(images.shape[0]):
            random_angle = rng.uniform(-max_angle, max_angle)
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=random_angle, scale=0.9)
            augmented_images.append(cv2.warpAffine(images[i],
                                                   M=rotate_matrix,
                                                   dsize=(width, height),
                                                   borderMode=cv2.BORDER_CONSTANT,
                                                   borderValue=(255,),
                                                   ))
            augmented_labels.append(labels[i])

    all_images = np.concatenate([np.array(augmented_images), images])
    all_labels = np.concatenate([np.array(augmented_labels), labels])

    return all_images, all_labels


def show_augmented_pairs(images, labels, dictionary, n=6, seed=40):
    """Show originals next to their augmented copies.
    Assumes images = [copies, originals] as returned by augment_images.
    """

    half = len(images) // 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(half, n, replace=False)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for col, i in enumerate(idx):
        axes[0, col].imshow(images[i], cmap="gray")
        axes[0, col].set_title(dictionary[labels[i]])
        axes[0, col].axis("off")
        axes[1, col].imshow(images[half + i], cmap="gray")
        axes[1, col].set_title(dictionary[labels[half + i]])
        axes[1, col].axis("off")

    plt.show()


def main(plot=False):
    dataset_folder = "dataset/phcd/ocr_files"
    signs, labels, dictionary = load_dataset(dataset_folder)

    describe_dataset(signs=signs, labels=labels, dictionary=dictionary)
    if plot:
        show_samples(signs=signs, labels=labels, dictionary=dictionary)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(signs, labels)
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                      ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        print(f"{name} shape: {arr.shape}")


if __name__ == "__main__":
    main()
