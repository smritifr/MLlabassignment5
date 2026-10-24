import time

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from mllab5 import (
    prepare_data,
    predict,
    accuracy_score_manual,
    SORT_FUNCTIONS,
)

X_train, X_test, y_train, y_test, _ = prepare_data()

K = 3

# -----------------------------
# scikit-learn k-NN
# -----------------------------
start = time.perf_counter()

model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, y_train)
sklearn_predictions = model.predict(X_test)

sklearn_time = time.perf_counter() - start

# -----------------------------
# Manual k-NN, every sorting algorithm
# (quadratic sorts are timed on a subset of the data)
# -----------------------------
SUBSET_TRAIN = 250
SUBSET_TEST = 40

rows = []

for algorithm in SORT_FUNCTIONS:

    start = time.perf_counter()

    predictions = predict(
        X_train[:SUBSET_TRAIN],
        y_train[:SUBSET_TRAIN],
        X_test[:SUBSET_TEST],
        k=K,
        sorting_algorithm=algorithm
    )

    elapsed = time.perf_counter() - start

    rows.append({
        "Sorting Algorithm": algorithm,
        "Time (s)": round(elapsed, 4),
        "Accuracy": round(accuracy_score_manual(y_test[:SUBSET_TEST], predictions), 4),
    })

subset_table = pd.DataFrame(rows)

# -----------------------------
# Manual k-NN on the full data
# -----------------------------
start = time.perf_counter()

manual_predictions = predict(
    X_train, y_train, X_test, k=K, sorting_algorithm="merge"
)

manual_time = time.perf_counter() - start

manual_accuracy = accuracy_score_manual(y_test, manual_predictions)
sklearn_accuracy = accuracy_score_manual(y_test, sklearn_predictions)
agreement = float(np.mean(manual_predictions == sklearn_predictions))

print("=" * 60)
print("k-NN Performance Comparison")
print("=" * 60)

print(f"Manual k-NN (merge sort) time : {manual_time:.4f} seconds")
print(f"scikit-learn k-NN time        : {sklearn_time:.4f} seconds")

print()

print(f"Manual accuracy       : {manual_accuracy:.4f}")
print(f"scikit-learn accuracy : {sklearn_accuracy:.4f}")
print(f"Prediction agreement  : {agreement:.4f}")

print()

print(f"Sorting algorithms on a subset ({SUBSET_TRAIN} train / {SUBSET_TEST} test samples)")
print("-" * 60)
print(subset_table.to_string(index=False))

print()

comparison = pd.DataFrame({

    "Feature": [
        "Execution Time",
        "Accuracy (k=3)",
        "Distance Computation",
        "Neighbour Search",
        "Tie Breaking",
        "Sorting Configurable"
    ],

    "Manual": [
        f"{manual_time:.4f} s",
        f"{manual_accuracy:.4f}",
        "Per-pair function",
        "Sort all distances",
        "Index / smaller label",
        "Yes"
    ],

    "scikit-learn": [
        f"{sklearn_time:.4f} s",
        f"{sklearn_accuracy:.4f}",
        "Optimised C / NumPy",
        "Tree or brute force",
        "Library internal",
        "No"
    ]

})

print(comparison.to_string(index=False))

print("\nObservations")
print("-" * 60)

print(f"1. The manual implementation took {manual_time / sklearn_time:.1f} times as long as scikit-learn.")
print(f"2. Predictions agree on {agreement:.2%} of the test samples.")

fastest = subset_table.loc[subset_table["Time (s)"].idxmin(), "Sorting Algorithm"]
slowest = subset_table.loc[subset_table["Time (s)"].idxmax(), "Sorting Algorithm"]

print(f"3. On the subset, {fastest} sort was the fastest and {slowest} sort the slowest.")
if subset_table["Accuracy"].nunique() == 1:
    print("4. All sorting algorithms gave the same accuracy, so the sort only affects running time.")
else:
    print("4. The sorting algorithms gave different accuracies, which needs to be investigated.")
