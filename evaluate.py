import numpy as np
from tensorflow.keras.models import load_model
from dataset import load_dataset, split_data
from sklearn.metrics import confusion_matrix


def top_confusions(y_true, y_pred, dictionary, n=15):
    cm_counts = confusion_matrix(y_true, y_pred)

    np.fill_diagonal(cm_counts, 0)
    cm_counts_flatten = cm_counts.flatten()
    order = np.argsort(cm_counts_flatten)[::-1][:n]

    for pos in order:
        w, k = np.unravel_index(pos, cm_counts.shape)
        print(f"{dictionary[w]} ----> {dictionary[k]}: {cm_counts[w, k]}")


def case_insensitive_accuracy(y_true, y_pred, dictionary):
    dictionary_swap = {value: key for key, value in dictionary.items()}

    canonical = {}
    for code, char in dictionary.items():
        lower_char = char.lower()
        canonical[code] = dictionary_swap.get(lower_char, code)

    canonical_arr = np.array([canonical[c] for c in range(89)])

    y_true_ci = canonical_arr[y_true]
    y_pred_ci = canonical_arr[y_pred]

    print(f"Accuracy with canonical = {(y_true_ci == y_pred_ci).mean()}")

    return y_true_ci, y_pred_ci


def main():
    signs, labels, dictionary = load_dataset("dataset/phcd/ocr_files")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(signs, labels)

    model = load_model("models/best_model.h5")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(f"Accuracy = {(y_pred == y_test).mean()}")

    top_confusions(y_test, y_pred, dictionary)

    y_true_ci, y_pred_ci = case_insensitive_accuracy(y_test, y_pred, dictionary)
    top_confusions(y_true_ci, y_pred_ci, dictionary)


if __name__ == "__main__":
    main()
