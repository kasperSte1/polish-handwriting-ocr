import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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


def main():
    dataset_folder = "dataset/phcd/ocr_files"
    signs, labels, dictionary = load_dataset(dataset_folder)
    show_samples(signs=signs, labels=labels, dictionary=dictionary)
    describe_dataset(signs=signs, labels=labels, dictionary=dictionary)


if __name__ == "__main__":
    main()
