"""Dictionary-constrained decoding of character-level model output.

Restricts the reading to words that actually exist, which fixes errors the
classifier cannot avoid on its own — letter case and letter/digit collisions.
"""

import numpy as np
from pathlib import Path
from dataset import build_char_to_code, load_dataset
from collections import defaultdict


def load_word_list(path):
    """Load the Polish word list (UTF-8, one lowercase word per line)."""

    # polish.txt is UTF-8; PHCD's dictionary.json is cp1250
    with open(path, encoding="utf-8") as f:
        words = {line.strip() for line in f}
    return words


def group_by_length(words):
    """Group words by length. {length: [words]}"""

    groups = defaultdict(list)
    # sorted because set iteration order varies between runs (Python hashes strings per process)
    for w in sorted(words):
        groups[len(w)].append(w)
    return groups


def encode_groups(groups, char_to_code):
    """Encode words as PHCD class codes. {length: ndarray(n_words, length)}

    Rows "i" matches groups[length][i]; the codes index the model's output layer
    """

    encoded = {}
    for length, words in groups.items():
        rows = []
        for word in words:
            codes = []
            for char in word:
                codes.append(char_to_code[char])
            rows.append(codes)
        encoded[length] = np.array(rows, dtype=np.uint8)
    return encoded


def match_word(probs, encoded, groups, top_k=5):
    """Most likely dictionary words for probs (L, 89), best first.
    Scores every candidate of length L by sum of log P(char | position).
    """

    L = probs.shape[0]
    if L not in encoded:
        return []
    # softmax can return exact zeros; log(0) = -inf would poison the whole sum
    log_probs = np.log(probs + 1e-12)
    letter_scores = log_probs[np.arange(L), encoded[L]]
    scores = letter_scores.sum(axis=1)
    top = np.argsort(scores)[::-1][:top_k]

    return [groups[L][i] for i in top]


def fake_probs(word, char_to_code, n_classes=89, confidence=0.9):
    """Build synthetic model output for a known word — for testing only."""

    probs = np.full((len(word), n_classes), 0.001)

    for pos, ch in enumerate(word):
        probs[pos, char_to_code[ch]] = confidence

    return probs / probs.sum(axis=1, keepdims=True)


def main():
    """Test: decode a synthetic reading of 'mecz'."""
    dataset_folder = "dataset/phcd/ocr_files"
    word = "mecz"
    polish_dictionary = Path("dataset/polish.txt")
    _, _, dictionary = load_dataset(dataset_folder)

    words = load_word_list(polish_dictionary)
    groups = group_by_length(words)
    char_to_code = build_char_to_code(dictionary)
    encoded = encode_groups(groups, char_to_code)

    probs = fake_probs(word, char_to_code)

    top_groups = match_word(probs, encoded, groups)
    print(top_groups)


if __name__ == "__main__":
    main()
