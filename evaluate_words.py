"""Word-level evaluation of the character classifier.

Measures how much dictionary post-processing improves reading accuracy on whole
words, as a function of word length.

Word images are composed of real PHCD glyphs: for each letter of a target word,
one random sample of that class is drawn from the test split, so the model has
never seen any of them. Each word is then read two ways:

  - argmax:     highest-probability class at each position, independently
  - dictionary: the most likely dictionary word of the same length, scored under
                the model's per-character probability distributions
"""

from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from wordmatch import load_word_list, group_by_length, encode_groups, match_word

from dataset import load_dataset, split_data, add_channel_dim, build_char_to_code
from tensorflow.keras.models import load_model

SAMPLES_PER_LENGTH = 1000
SEED = 42

DATASET_DIR = "dataset/phcd/ocr_files"
MODEL_PATH = "models/best_model.h5"
WORDS_PATH = "dataset/polish.txt"
FIGURE_PATH = "figures/word_accuracy_by_length.png"


def compose_word(word, by_class, X_test, y_test, char_to_code, rng):
    """Pick one random test-set glyph per character.

    Returns (L, 32, 32)
    """
    glyphs = []
    for char in word:
        code = char_to_code[char]
        idx = rng.choice(by_class[code])
        assert y_test[idx] == code, f"'{char}': expected class {code}, got {y_test[idx]}"
        glyphs.append(X_test[idx])

    return np.array(glyphs)


def plot_accuracy_by_length(results, path):
    """Plot argmax vs dictionary word accuracy against word length."""

    results = np.array(results)
    results = results[results[:, 0].argsort()]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(results[:, 0], results[:, 1], marker="o", label="argmax")
    ax.plot(results[:, 0], results[:, 2], marker="o", label="dictionary")
    ax.set_ylim(0, 1)
    ax.set_xlabel("word length")
    ax.set_ylabel("word accuracy")
    ax.legend()
    ax.grid(True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    signs, labels, dictionary = load_dataset(DATASET_DIR)
    _, _, X_test, _, _, y_test = split_data(signs, labels)

    by_class = defaultdict(list)
    for i, label in enumerate(y_test):
        by_class[label].append(i)

    rng = np.random.default_rng(SEED)

    model = load_model(MODEL_PATH)

    words = load_word_list(WORDS_PATH)
    groups = group_by_length(words)
    char_to_code = build_char_to_code(dictionary)
    encoded = encode_groups(groups, char_to_code)

    print(f"{'len':>3} {'argmax':>8} {'dict':>8}")
    results = []
    for length, words_of_length in sorted(groups.items()):

        n = min(SAMPLES_PER_LENGTH, len(words_of_length))
        sample = rng.choice(words_of_length, size=n, replace=False)

        ok_argmax = 0
        ok_dict = 0
        showed = 0

        for word in sample:
            # compose the word image and run the model
            imgs = compose_word(word, by_class, X_test, y_test, char_to_code, rng)
            imgs, = add_channel_dim(imgs)
            probs = model.predict(imgs, verbose=0)

            # reading A: highest-probability class at each position
            codes = probs.argmax(axis=1)
            reading_argmax = "".join(dictionary[c] for c in codes)

            # reading B: constrained to real words from Polish dictionary
            candidates = match_word(probs, encoded, groups)
            reading_dict = candidates[0] if candidates else ""

            if reading_argmax == word:
                ok_argmax += 1
            if reading_dict == word:
                ok_dict += 1

            if reading_argmax != word and showed < 5:
                print(f"{word:12} argmax: {reading_argmax:12} dict: {reading_dict}")
                showed += 1
        results.append((length, ok_argmax / len(sample), ok_dict / len(sample)))

    for length, acc_argmax, acc_dict in results:
        print(f"{length:>3} {acc_argmax:8.3f} {acc_dict:8.3f}")

    plot_accuracy_by_length(results, FIGURE_PATH)


if __name__ == "__main__":
    main()
