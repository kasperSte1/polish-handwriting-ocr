# Handwritten Polish Character Recognition

Reading handwritten Polish words: a CNN character classifier combined with
dictionary-constrained decoding that turns per-character predictions into words.

## The problem

Off-the-shelf OCR is trained on printed Latin text and degrades on two things at
once: handwriting, and Polish diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż). The
89-class label space here covers digits, both letter cases, all Polish
diacritics and punctuation.

## Results

| Metric | Accuracy                            |
|---|-------------------------------------|
| Character accuracy (89 classes, test set) | **83.36%**                          |
| Character accuracy ignoring letter case | **91.84%**                          |
| Word accuracy, per-character argmax | **63.8% → 4.6%** (2 to 15 letters)  |
| Word accuracy, argmax lowercased | **76.4% → 20.2%** (2 to 15 letters) |
| Word accuracy, dictionary-constrained | **96.1% → 99.4%** (2 to 15 letters) |

The last three rows in full — both raw readings degrade as words get longer,
while dictionary-constrained reading rises:

![word accuracy by length](figures/word_accuracy_by_length.png)

Full experiment log, error analysis and methodology: **[RESULTS.md](RESULTS.md)**

## Key finding

Character accuracy plateaus at 83.36%, and the reason is largely in the dataset,
not the model. **PHCD scales every glyph to a fixed 20 × 32 px box inside a
32 × 32 image**, removing size — the primary cue distinguishing `c` from `C`.
Measuring accuracy with letter case ignored gives 91.84% — an 8.5 p.p. gap against 
83.36%, almost all of it on letters whose cases differ mainly in size.

Constraining the output to real Polish words reverses the trend entirely. Raw
per-character reading degrades with length — from 63.8% on two-letter words to
4.6% on fifteen-letter ones — while dictionary-constrained reading rises, from
96.1% to 99.4%. Lowercasing the argmax output makes the comparison fair, since
the decoder only ever produces lowercase: averaged over lengths 2–15, that
baseline reads 44.4% of words correctly and the dictionary decoder 98.0%. Raw
argmax can produce any of the roughly $10^{29}$ possible 15-character strings,
almost all of them meaningless; dictionary decoding can only produce one of the
261 663 that are real words.

## How it works

```
  L character images                  32 × 32 px, one per letter
          │
          ▼
  CNN classifier                      89-way softmax per character
          │
          ▼
  probabilities (L, 89)
          │
          ├──────────────► argmax                → "ma9ne+owIdowemU"
          │                highest class per position, independently
          │
          └──────────────► dictionary decoder    → "magnetowidowemu"
                           scores every Polish word of length L by
                           Σ log P(char | position), takes the best
```

The decoder is lexicon-constrained maximum-likelihood decoding, vectorized so that all
candidates of a given length are scored in one operation.

## How to run

TensorFlow 2.10 is the last release with GPU support on native Windows, which
pins the rest of the stack.

| Component  | Version |
|------------|---------|
| Python     | 3.10    |
| TensorFlow | 2.10.0  |
| CUDA       | 11.2    |
| cuDNN      | 8.1.0   |

```bash
conda create -n word_recog_env python=3.10
conda activate word_recog_env
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -r requirements.txt
python scripts/check_env.py
```

`check_env.py` verifies that TensorFlow sees the GPU and that the numpy/OpenCV
interop works before you start training.

### Data

The [PHCD dataset](https://cs.pollub.pl/phcd/?lang=en) (Lublin University of
Technology) is not included — download it and place `ocr_files/` under
`dataset/phcd/`.

The decoder needs a Polish word list at `dataset/polish.txt` — UTF-8, one
lowercase inflected form per line, using only the 35 characters 
(`a-z` plus `ą ć ę ł ń ó ś ź ż`). The results reported here use a
list of 2 703 830 forms. [sjp.pl](https://sjp.pl/sl/odmiany/) publishes a
suitable openly licensed list, stored one line per headword with
comma-separated forms, so it needs expanding to one form per line.

### Usage

```bash
python dataset.py         # dataset statistics and sample glyphs
python model.py           # architecture summary
python train.py           # train, ~45 s/epoch on an RTX 4050
python evaluate.py        # character-level accuracy and confusion analysis
python evaluate_words.py  # word-level accuracy by word length
```

## Project structure

```
.
├── dataset.py              # Loading, stratified splitting, offline augmentation
├── model.py                # CNN architecture (150k parameters)
├── train.py                # Training with early stopping and LR scheduling
├── evaluate.py             # Character-level metrics and confusion analysis
├── wordmatch.py            # Dictionary encoding and constrained decoding
├── evaluate_words.py       # Word-level evaluation by word length
├── scripts/
│   └── check_env.py        # Verifies GPU visibility and package interop
├── figures/                # Generated plots
├── models/                 # Trained models (not tracked)
├── dataset/                # PHCD data and word list (not tracked)
├── RESULTS.md              # Full experiment log and error analysis
└── requirements.txt
```

## Limitations

- **Out-of-vocabulary words cannot be read.** The decoder always returns a real
word, so proper nouns and abbreviations will be wrong.
- **Case is forced, not recovered.** The word list is lowercase only, so `Kość`
reads as `kość`.
- **Two valid words are indistinguishable.** The model reads `mit` as `nit`;
both exist, and the constraint does not help.
- **Photographed input is not supported.** The pipeline assumes pre-segmented
character images; separating joined letters in natural cursive requires a
segmentation-free approach such as CRNN + CTC.

## Data sources

- **PHCD** — Polish Handwritten Characters Database, Department of Computer
  Science, Lublin University of Technology. 558 155 glyphs across 89 classes.
- **Word list** — Polish inflected forms, 2 703 830 entries; see [Data](#data)
  for the source and format.