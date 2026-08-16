"""Character-level evaluation of the trained classifier.

Reports accuracy, the most frequent confusions, and accuracy with letter case
ignored — the gap between the two is the part no model can recover, since PHCD
normalises every glyph to a fixed height.
"""

import numpy as np
from tensorflow.keras.models import load_model
from dataset import load_dataset, split_data, build_char_to_code, add_channel_dim, build_canonical_map
from sklearn.metrics import confusion_matrix

DATASET_DIR = "dataset/phcd/ocr_files"
MODEL_PATH = "models/best_model.h5"


def top_confusions(y_true, y_pred, dictionary, n=15):
    """Print the n most frequent misclassifications, correct ones excluded."""

    cm_counts = confusion_matrix(y_true, y_pred, labels=np.arange(len(dictionary)))

    np.fill_diagonal(cm_counts, 0)
    cm_counts_flatten = cm_counts.flatten()
    order = np.argsort(cm_counts_flatten)[::-1][:n]

    for pos in order:
        w, k = np.unravel_index(pos, cm_counts.shape)
        print(f"{dictionary[w]} ----> {dictionary[k]}: {cm_counts[w, k]}")


def case_insensitive_accuracy(y_true, y_pred, dictionary):
    """Accuracy with upper/lowercase pairs merged (C and c count as one class).

    Returns accuracy and both label arrays after merging,
    so confusions can be recomputed on them.
    """

    canonical_arr = build_canonical_map(dictionary)

    y_true_ci = canonical_arr[y_true]
    y_pred_ci = canonical_arr[y_pred]

    accuracy = (y_true_ci == y_pred_ci).mean()
    print(f"Accuracy (case-insensitive) = {accuracy}")

    return accuracy, y_true_ci, y_pred_ci


def main():
    signs, labels, dictionary = load_dataset(DATASET_DIR)
    _, _, X_test, _, _, y_test = split_data(signs, labels)
    X_test, = add_channel_dim(X_test)

    model = load_model(MODEL_PATH)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(f"Accuracy = {(y_pred == y_test).mean()}")

    top_confusions(y_test, y_pred, dictionary)

    _, y_true_ci, y_pred_ci = case_insensitive_accuracy(y_test, y_pred, dictionary)
    top_confusions(y_true_ci, y_pred_ci, dictionary)


if __name__ == "__main__":
    main()
