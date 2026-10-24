# Lab 5 - k-NN Classification (23CSE301)

## Files

| File | Purpose |
|------|---------|
| `mllab5.py` | Complete implementation (A1-A9) in one file; run it to reproduce every result |
| `test_mllab5.py` | Unit tests (`unittest`) |
| `knn_comparison.py` | Timing and accuracy comparison of the manual k-NN and scikit-learn |
| `main.tex`, `references.bib` | IEEE report |
| `requirements.txt` | Python packages |

## Dataset

Place the project dataset in this folder as `uniswap_v3_swaps_100k.csv`
(a name such as `uniswap_v3_swaps_100k(1).csv` is also found automatically).

Target: `fee_tier`, restricted to the classes 100 and 3000 (balanced sample of
600 rows per class, `random_state=42`). `fee_tier` is not used as a feature.

## Run

```text
pip install -r requirements.txt
python mllab5.py
python knn_comparison.py
python -m unittest test_mllab5
```

`mllab5.py` writes the CSV files and LaTeX snippets to `outputs/` and the plots
to `figures/`. Build the report afterwards:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Structure of `mllab5.py`

1. Data loading, imputation, frequency encoding, log transform, scaling, 70/30 stratified split
2. Sorting algorithms (bubble, insertion, selection, merge), selected by name
3. Distance metrics, neighbour selection, majority vote, weighted vote
4. `ManualKNN` class with `fit()`, `predict()`, `score()`
5. Comparison with scikit-learn, plots, metrics and report files
6. Main program (all `print` statements are here)
