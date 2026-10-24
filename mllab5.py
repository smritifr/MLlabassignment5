import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = "uniswap_v3_swaps_100k(1).csv"
TARGET_COLUMN = "fee_tier"
CLASSES = [100, 3000]
SAMPLE_PER_CLASS = 600
RANDOM_STATE = 42
TEST_SIZE = 0.30

NUMERIC_FEATURES = [
    "timestamp", "amount0", "amount1", "amountUSD",
    "token0_decimals", "token1_decimals", "block_number",
    "gas_used", "gas_price"
]
CATEGORICAL_FEATURES = ["token0_symbol", "token1_symbol"]

# Heavy-tailed transaction fields that are compressed with a signed log.
LOG_FEATURES = ["amount0", "amount1", "amountUSD", "gas_used", "gas_price"]

OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"


def heading(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_data_file(data_file: str = DATA_FILE) -> str:
    """Returns the dataset path, also accepting names like 'name(1).csv'."""
    if os.path.exists(data_file):
        return data_file

    stem = os.path.splitext(data_file)[0]
    matches = sorted(glob.glob(stem + "*.csv"))

    if not matches:
        raise FileNotFoundError(
            f"Dataset '{data_file}' was not found in the current folder."
        )

    return matches[0]


def load_dataset(data_file: str = DATA_FILE) -> pd.DataFrame:
    """Reads the swap dataset from a CSV file."""
    return pd.read_csv(find_data_file(data_file))


def build_binary_dataset(
    df: pd.DataFrame,
    classes=CLASSES,
    sample_per_class: int = SAMPLE_PER_CLASS,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Keeps exactly two classes and draws a balanced, shuffled sample."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    if len(set(classes)) != 2:
        raise ValueError("Exactly two distinct classes are required.")

    filtered = df[df[TARGET_COLUMN].isin(classes)]

    parts = []

    for label in classes:
        rows = filtered[filtered[TARGET_COLUMN] == label]

        if len(rows) == 0:
            raise ValueError(f"No rows found for class {label}.")

        n = min(sample_per_class, len(rows))
        parts.append(rows.sample(n=n, random_state=random_state))

    balanced = pd.concat(parts, ignore_index=True)

    return balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# A1 (a, b): Encoding and data imputation
# ---------------------------------------------------------------------------

def fit_imputer(df: pd.DataFrame, numeric_strategy: str = "median") -> dict:
    """Learns the fill value of every feature from the given data.

    Numeric columns use the mean or the median, categorical columns use
    the mode.
    """

    if numeric_strategy not in ("mean", "median"):
        raise ValueError("numeric_strategy must be 'mean' or 'median'.")

    fill_values = {}

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(df[column], errors="coerce")

        if numeric_strategy == "mean":
            fill_values[column] = float(values.mean())
        else:
            fill_values[column] = float(values.median())

    for column in CATEGORICAL_FEATURES:
        modes = df[column].mode(dropna=True)
        fill_values[column] = modes.iloc[0] if len(modes) else "__MISSING__"

    return fill_values


def apply_imputer(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    """Fills missing values using values learned by fit_imputer()."""

    data = df.copy()

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(fill_values[column])

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].fillna(fill_values[column])

    return data


def fit_category_maps(df: pd.DataFrame) -> dict:
    """Learns the frequency of every category (frequency encoding)."""

    maps = {}

    for column in CATEGORICAL_FEATURES:
        maps[column] = df[column].value_counts(normalize=True).to_dict()

    return maps


def encode_categorical(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    """Replaces every category with its training frequency.

    A category never seen during training is encoded as 0.0.
    """

    data = df.copy()

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].map(category_maps[column]).fillna(0.0)

    return data


def signed_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Compresses heavy-tailed columns while preserving the sign."""

    data = df.copy()

    for column in LOG_FEATURES:
        data[column] = np.sign(data[column]) * np.log1p(np.abs(data[column]))

    return data


def transform_features(df: pd.DataFrame, fill_values: dict, category_maps: dict):
    """Applies imputation, encoding and the log transform to a dataframe."""

    data = apply_imputer(df, fill_values)
    data = encode_categorical(data, category_maps)
    data = signed_log_transform(data)

    return data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].astype(float)


# ---------------------------------------------------------------------------
# A3: Train / test split and complete preparation pipeline
# ---------------------------------------------------------------------------

def prepare_data(data_file: str = DATA_FILE, numeric_strategy: str = "median"):
    """Loads the data and returns standardized train and test arrays.

    All statistics (fill values, category frequencies, scaling) are learned
    from the training part only.
    """

    raw = load_dataset(data_file)
    binary = build_binary_dataset(raw)

    X_raw = binary.drop(columns=[TARGET_COLUMN])
    y = binary[TARGET_COLUMN].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    fill_values = fit_imputer(X_train_raw, numeric_strategy)
    X_train_filled = apply_imputer(X_train_raw, fill_values)
    category_maps = fit_category_maps(X_train_filled)

    X_train = transform_features(X_train_raw, fill_values, category_maps)
    X_test = transform_features(X_test_raw, fill_values, category_maps)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), scaler


# ---------------------------------------------------------------------------
# A1 (d): Sorting algorithms (selected through a configuration parameter)
# ---------------------------------------------------------------------------

def bubble_sort(items):
    """Bubble sort with early exit. Returns a new sorted list."""
    items = list(items)
    n = len(items)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
                swapped = True

        if not swapped:
            break

    return items


def insertion_sort(items):
    """Insertion sort. Returns a new sorted list."""
    items = list(items)

    for i in range(1, len(items)):
        current = items[i]
        j = i - 1

        while j >= 0 and items[j] > current:
            items[j + 1] = items[j]
            j -= 1

        items[j + 1] = current

    return items


def selection_sort(items):
    """Selection sort. Returns a new sorted list."""
    items = list(items)
    n = len(items)

    for i in range(n):
        minimum_index = i

        for j in range(i + 1, n):
            if items[j] < items[minimum_index]:
                minimum_index = j

        items[i], items[minimum_index] = items[minimum_index], items[i]

    return items


def merge_sort(items):
    """Merge sort, O(n log n). Used for the full-size experiments."""
    items = list(items)

    if len(items) <= 1:
        return items

    middle = len(items) // 2
    left = merge_sort(items[:middle])
    right = merge_sort(items[middle:])

    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


SORT_FUNCTIONS = {
    "bubble": bubble_sort,
    "insertion": insertion_sort,
    "selection": selection_sort,
    "merge": merge_sort,
}


def sort_distance_pairs(pairs, algorithm: str = "merge"):
    """Sorts (distance, index, label) tuples with the chosen algorithm."""

    if algorithm not in SORT_FUNCTIONS:
        raise ValueError(
            "Unsupported sorting algorithm. Use one of: "
            + ", ".join(SORT_FUNCTIONS)
        )

    return SORT_FUNCTIONS[algorithm](pairs)


# ---------------------------------------------------------------------------
# A1 (c, e, f): Distance, neighbours and class assignment
# ---------------------------------------------------------------------------

def calculate_distance(point_a, point_b, metric: str = "euclidean") -> float:
    """Distance between two points for the selected metric."""

    difference = np.abs(np.asarray(point_a, dtype=float) - np.asarray(point_b, dtype=float))

    if metric == "euclidean":
        return float(np.sqrt(np.sum(difference ** 2)))

    if metric == "manhattan":
        return float(np.sum(difference))

    if metric == "chebyshev":
        return float(np.max(difference))

    raise ValueError("Unsupported metric. Use 'euclidean', 'manhattan' or 'chebyshev'.")


def get_sorted_neighbors(
    X_train,
    y_train,
    query,
    metric: str = "euclidean",
    sorting_algorithm: str = "merge"
):
    """Returns all training samples sorted by distance to the query.

    Every entry is (distance, training index, label). Because the training
    index is the second element, two samples at equal distance are ordered
    by index, which is the tie-breaking rule for neighbour selection.
    """

    pairs = []

    for index, row in enumerate(X_train):
        distance = calculate_distance(query, row, metric)
        pairs.append((distance, index, int(y_train[index])))

    return sort_distance_pairs(pairs, sorting_algorithm)


def select_k_neighbors(sorted_neighbors, k: int):
    """Returns the k nearest neighbours from an already sorted list."""

    if k < 1 or k > len(sorted_neighbors):
        raise ValueError("k must be between 1 and the number of training samples.")

    return sorted_neighbors[:k]


def majority_vote(neighbors) -> int:
    """Class with the most votes; equal votes go to the smaller label."""

    counts = {}

    for _, _, label in neighbors:
        counts[label] = counts.get(label, 0) + 1

    highest = max(counts.values())
    winners = [label for label, count in counts.items() if count == highest]

    return min(winners)


def weighted_vote(neighbors, epsilon: float = 1e-12) -> int:
    """Class with the largest total inverse-distance weight.

    The weight of a neighbour is 1 / max(distance, epsilon), so a neighbour
    at distance zero cannot cause a division by zero. Equal totals go to
    the smaller label.
    """

    scores = {}

    for distance, _, label in neighbors:
        weight = 1.0 / max(distance, epsilon)
        scores[label] = scores.get(label, 0.0) + weight

    highest = max(scores.values())
    winners = [
        label for label, score in scores.items()
        if np.isclose(score, highest, rtol=1e-12, atol=1e-12)
    ]

    return min(winners)


def predict_one(
    X_train,
    y_train,
    query,
    k: int = 3,
    metric: str = "euclidean",
    sorting_algorithm: str = "merge",
    weighted: bool = False
) -> int:
    """Predicts the class label of a single query vector."""

    sorted_neighbors = get_sorted_neighbors(
        X_train, y_train, query, metric, sorting_algorithm
    )

    selected = select_k_neighbors(sorted_neighbors, k)

    if weighted:
        return weighted_vote(selected)

    return majority_vote(selected)


def predict(
    X_train,
    y_train,
    X_test,
    k: int = 3,
    metric: str = "euclidean",
    sorting_algorithm: str = "merge",
    weighted: bool = False
):
    """Predicts the class labels of every row of X_test."""

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of samples.")

    predictions = [
        predict_one(X_train, y_train, query, k, metric, sorting_algorithm, weighted)
        for query in X_test
    ]

    return np.asarray(predictions)


def predict_many_k(
    X_train,
    y_train,
    X_test,
    k_values,
    metric: str = "euclidean",
    sorting_algorithm: str = "merge",
    weighted: bool = False
) -> dict:
    """Predicts for several k values while sorting each query only once."""

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    k_values = sorted(set(int(k) for k in k_values))
    results = {k: [] for k in k_values}

    for query in X_test:
        sorted_neighbors = get_sorted_neighbors(
            X_train, y_train, query, metric, sorting_algorithm
        )

        for k in k_values:
            selected = select_k_neighbors(sorted_neighbors, k)

            if weighted:
                results[k].append(weighted_vote(selected))
            else:
                results[k].append(majority_vote(selected))

    return {k: np.asarray(values) for k, values in results.items()}


def accuracy_score_manual(y_true, y_pred) -> float:
    """Fraction of correctly predicted labels."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# A7: fit(), predict() and score() interface
# ---------------------------------------------------------------------------

class ManualKNN:
    """k-NN classifier built from the functions above."""

    def __init__(
        self,
        k: int = 3,
        metric: str = "euclidean",
        sorting_algorithm: str = "merge",
        weighted: bool = False
    ):
        if k < 1:
            raise ValueError("k must be at least 1.")

        if sorting_algorithm not in SORT_FUNCTIONS:
            raise ValueError("Unsupported sorting algorithm.")

        self.k = k
        self.metric = metric
        self.sorting_algorithm = sorting_algorithm
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Stores the training data (k-NN is a lazy learner)."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training samples.")

        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):
        """Predicts class labels for the rows of X."""

        if self.X_train is None:
            raise ValueError("Call fit() before predict().")

        return predict(
            self.X_train,
            self.y_train,
            X,
            k=self.k,
            metric=self.metric,
            sorting_algorithm=self.sorting_algorithm,
            weighted=self.weighted
        )

    def score(self, X, y) -> float:
        """Accuracy of predict(X) against the true labels y."""

        return accuracy_score_manual(y, self.predict(X))


# ---------------------------------------------------------------------------
# A8 / A9: Comparison with scikit-learn, plots and report files
# ---------------------------------------------------------------------------

def evaluate_sklearn_over_k(
    X_train, X_test, y_train, y_test, k_values, weights: str = "uniform"
) -> pd.DataFrame:
    """Train and test accuracy of KNeighborsClassifier for every k."""

    rows = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights)
        model.fit(X_train, y_train)

        rows.append({
            "k": k,
            "sklearn_train_accuracy": model.score(X_train, y_train),
            "sklearn_test_accuracy": model.score(X_test, y_test),
        })

    return pd.DataFrame(rows)


def compare_over_k(
    X_train, X_test, y_train, y_test, k_values,
    weighted: bool = False,
    sorting_algorithm: str = "merge",
    metric: str = "euclidean"
) -> pd.DataFrame:
    """Test accuracy of the manual k-NN and sklearn k-NN for every k."""

    manual_predictions = predict_many_k(
        X_train, y_train, X_test, k_values,
        metric=metric,
        sorting_algorithm=sorting_algorithm,
        weighted=weighted
    )

    manual_accuracy = [
        accuracy_score_manual(y_test, manual_predictions[k]) for k in k_values
    ]

    weights = "distance" if weighted else "uniform"

    sklearn_results = evaluate_sklearn_over_k(
        X_train, X_test, y_train, y_test, k_values, weights
    )

    results = pd.DataFrame({"k": list(k_values), "manual_test_accuracy": manual_accuracy})

    return results.merge(sklearn_results, on="k")


def plot_accuracy_comparison(
    results: pd.DataFrame,
    title: str,
    manual_label: str,
    sklearn_label: str,
    path: str,
    show: bool = True
) -> str:
    """Plots manual and sklearn test accuracy against k and saves the figure."""

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(
        results["k"], results["manual_test_accuracy"],
        marker="o", linewidth=2, label=manual_label
    )

    plt.plot(
        results["k"], results["sklearn_test_accuracy"],
        marker="s", linestyle="--", linewidth=2, label=sklearn_label
    )

    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Test Accuracy")
    plt.xticks(list(results["k"]))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)

    if show:
        plt.show()

    plt.close()

    return path


def plot_train_test_accuracy(
    results: pd.DataFrame, title: str, path: str, show: bool = True
) -> str:
    """Plots sklearn train and test accuracy against k to study fit."""

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(
        results["k"], results["sklearn_train_accuracy"],
        marker="o", linewidth=2, label="Train accuracy"
    )

    plt.plot(
        results["k"], results["sklearn_test_accuracy"],
        marker="s", linewidth=2, label="Test accuracy"
    )

    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.xticks(list(results["k"]))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)

    if show:
        plt.show()

    plt.close()

    return path


def best_k(results: pd.DataFrame, column: str):
    """Returns (k, accuracy) of the highest accuracy; smallest k wins ties."""

    best_row = results.loc[results[column].idxmax()]

    return int(best_row["k"]), float(best_row[column])


def classification_metrics(y_true, y_pred, labels) -> pd.DataFrame:
    """Precision, recall and F1-score of every class, plus the support."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    rows = []

    for label in labels:
        true_positive = int(np.sum((y_pred == label) & (y_true == label)))
        false_positive = int(np.sum((y_pred == label) & (y_true != label)))
        false_negative = int(np.sum((y_pred != label) & (y_true == label)))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append({
            "class": int(label),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(np.sum(y_true == label)),
        })

    return pd.DataFrame(rows)


def confusion_counts(y_true, y_pred, labels) -> pd.DataFrame:
    """Confusion matrix with true classes as rows and predictions as columns."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    matrix = [
        [int(np.sum((y_true == actual) & (y_pred == predicted))) for predicted in labels]
        for actual in labels
    ]

    return pd.DataFrame(
        matrix,
        index=[f"true {label}" for label in labels],
        columns=[f"pred {label}" for label in labels]
    )


def write_metrics_table(metrics: pd.DataFrame, path: str) -> str:
    """Writes the per-class metrics as rows of a LaTeX tabular body."""

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    lines = []

    for _, row in metrics.iterrows():
        lines.append(
            f"{int(row['class'])} & {row['precision']:.4f} & {row['recall']:.4f} "
            f"& {row['f1']:.4f} & {int(row['support'])} \\\\ \\hline"
        )

    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")

    return path


def write_latex_table(results: pd.DataFrame, columns, path: str) -> str:
    """Writes the chosen columns as rows of a LaTeX tabular body.

    The report reads this file with \\input, so the table always contains
    the measured values. Accuracies are written as percentages.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    lines = []

    for _, row in results.iterrows():
        cells = [str(int(row["k"]))]
        cells += [f"{100 * row[column]:.2f}\\%" for column in columns]
        lines.append(" & ".join(cells) + " \\\\ \\hline")

    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")

    return path


def write_summary_macros(values: dict, path: str) -> str:
    """Writes LaTeX macros (name -> value) used by the report text."""

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w") as file:
        for name, value in values.items():
            file.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")

    return path


# ---------------------------------------------------------------------------
# Main program (all print statements are here)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    X_train, X_test, y_train, y_test, scaler = prepare_data()

    k_values = list(range(1, 16, 2))

    # ----- A3 -----
    heading("A3: 70/30 Train-Test Split")

    print("Training samples :", len(y_train))
    print("Testing samples  :", len(y_test))
    print("Train shape      :", X_train.shape)
    print("Test shape       :", X_test.shape)
    print("Classes          :", [int(c) for c in sorted(set(y_train) | set(y_test))])

    for name, labels in (("Train", y_train), ("Test ", y_test)):
        classes, counts = np.unique(labels, return_counts=True)
        print(f"{name} class counts:", {int(c): int(n) for c, n in zip(classes, counts)})

    # ----- A1 -----
    heading("A1: Modular k-NN with Configurable Sorting")

    sample = [4, 1, 3, 2, 5]

    print("Bubble sort    :", bubble_sort(sample))
    print("Insertion sort :", insertion_sort(sample))
    print("Selection sort :", selection_sort(sample))
    print("Merge sort     :", merge_sort(sample))

    # The quadratic sorts are slow, so they are demonstrated on a subset.
    subset_predictions = {}

    for algorithm in SORT_FUNCTIONS:
        subset_predictions[algorithm] = predict(
            X_train[:250], y_train[:250], X_test[:40],
            k=3, metric="euclidean", sorting_algorithm=algorithm
        )

        subset_accuracy = accuracy_score_manual(y_test[:40], subset_predictions[algorithm])

        print(f"{algorithm.capitalize()} sort, k=3, demo subset accuracy: {subset_accuracy:.4f}")

    all_same = all(
        np.array_equal(subset_predictions["merge"], values)
        for values in subset_predictions.values()
    )

    print("All sorting algorithms give identical predictions:", all_same)

    full_predictions = predict(X_train, y_train, X_test, k=3, sorting_algorithm="merge")

    print(f"Manual k-NN full test accuracy (k=3): {accuracy_score_manual(y_test, full_predictions):.4f}")

    # ----- A2 -----
    heading("A2: Weighted k-NN")

    print("Weight rule: inverse distance, 1 / max(distance, epsilon)")

    weighted_predictions = predict(
        X_train, y_train, X_test, k=5, sorting_algorithm="merge", weighted=True
    )

    print(f"Weighted manual k-NN test accuracy (k=5): {accuracy_score_manual(y_test, weighted_predictions):.4f}")

    # ----- A4 -----
    heading("A4: scikit-learn KNeighborsClassifier (k=3)")

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    print("k =", neigh.n_neighbors)
    print("Number of training samples:", len(X_train))

    # ----- A5 -----
    heading("A5: Accuracy using score()")

    print(f"Training accuracy: {neigh.score(X_train, y_train):.4f}")
    print(f"Testing accuracy : {neigh.score(X_test, y_test):.4f}")

    sk_predictions = neigh.predict(X_test)
    class_labels = [int(c) for c in sorted(set(y_train))]
    metrics_table = classification_metrics(y_test, sk_predictions, class_labels)

    print("\nPer-class test metrics (k=3):")
    print(metrics_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nConfusion matrix (k=3, test set):")
    print(confusion_counts(y_test, sk_predictions, class_labels))

    # ----- A6 -----
    heading("A6: Prediction Behaviour using predict()")

    print("First 20 predictions:", sk_predictions[:20])
    print("First 20 actual     :", y_test[:20])

    for i in range(10):
        print(f"Sample {i + 1:>2}: predicted={sk_predictions[i]}, actual={y_test[i]}")

    # ----- A7 -----
    heading("A7: ManualKNN with fit(), predict() and score()")

    model = ManualKNN(k=3, metric="euclidean", sorting_algorithm="merge", weighted=False)
    model.fit(X_train, y_train)

    print("First 20 predictions:", model.predict(X_test[:20]))
    print(f"Training score: {model.score(X_train, y_train):.4f}")
    print(f"Testing score : {model.score(X_test, y_test):.4f}")
    print("Agreement with sklearn predictions:",
          np.array_equal(model.predict(X_test), sk_predictions))

    # ----- A8 -----
    heading("A8: Manual vs scikit-learn k-NN for Different k")

    a8 = compare_over_k(X_train, X_test, y_train, y_test, k_values, weighted=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    a8.to_csv(os.path.join(OUTPUT_DIR, "A8_accuracy_comparison.csv"), index=False)

    print(a8.to_string(index=False))

    plot_accuracy_comparison(
        a8, "A8: Manual vs sklearn k-NN", "Manual k-NN", "sklearn k-NN",
        os.path.join(FIGURE_DIR, "A8_accuracy_comparison.png")
    )

    plot_train_test_accuracy(
        a8, "Train vs Test Accuracy (sklearn k-NN)",
        os.path.join(FIGURE_DIR, "train_test_accuracy.png")
    )

    # ----- A9 -----
    heading("A9: Weighted k-NN Comparison")

    a9 = compare_over_k(X_train, X_test, y_train, y_test, k_values, weighted=True)

    a9.to_csv(os.path.join(OUTPUT_DIR, "A9_weighted_comparison.csv"), index=False)

    print(a9.to_string(index=False))

    plot_accuracy_comparison(
        a9, "A9: Weighted k-NN Comparison", "Manual weighted k-NN", "sklearn weighted k-NN",
        os.path.join(FIGURE_DIR, "A9_weighted_comparison.png")
    )

    # ----- Comparison of A8 and A9 -----
    heading("A8 vs A9: Effect of Distance Weighting")

    combined = pd.DataFrame({
        "k": k_values,
        "uniform_manual": a8["manual_test_accuracy"],
        "weighted_manual": a9["manual_test_accuracy"],
    })

    combined["difference"] = combined["weighted_manual"] - combined["uniform_manual"]

    print(combined.to_string(index=False))

    # ----- Files used by the LaTeX report -----
    best_normal_k, best_normal_accuracy = best_k(a8, "manual_test_accuracy")
    best_weighted_k, best_weighted_accuracy = best_k(a9, "manual_test_accuracy")

    three = a8[a8["k"] == 3].iloc[0]

    max_difference = max(
        (a8["manual_test_accuracy"] - a8["sklearn_test_accuracy"]).abs().max(),
        (a9["manual_test_accuracy"] - a9["sklearn_test_accuracy"]).abs().max()
    )

    write_latex_table(a8, ["manual_test_accuracy", "sklearn_test_accuracy", "sklearn_train_accuracy"],
                      os.path.join(OUTPUT_DIR, "a8_table.tex"))

    write_latex_table(a9, ["manual_test_accuracy", "sklearn_test_accuracy"],
                      os.path.join(OUTPUT_DIR, "a9_table.tex"))

    write_metrics_table(metrics_table, os.path.join(OUTPUT_DIR, "metrics_table.tex"))

    write_summary_macros({
        "nTrain": len(y_train),
        "nTest": len(y_test),
        "bestKNormal": best_normal_k,
        "bestAccNormal": f"{100 * best_normal_accuracy:.2f}",
        "bestKWeighted": best_weighted_k,
        "bestAccWeighted": f"{100 * best_weighted_accuracy:.2f}",
        "trainAccThree": f"{100 * three['sklearn_train_accuracy']:.2f}",
        "testAccThree": f"{100 * three['sklearn_test_accuracy']:.2f}",
        "gapAccThree": f"{100 * (three['sklearn_train_accuracy'] - three['sklearn_test_accuracy']):.2f}",
        "trainAccOne": f"{100 * a8.iloc[0]['sklearn_train_accuracy']:.2f}",
        "testAccOne": f"{100 * a8.iloc[0]['sklearn_test_accuracy']:.2f}",
        "trainAccLast": f"{100 * a8.iloc[-1]['sklearn_train_accuracy']:.2f}",
        "testAccLast": f"{100 * a8.iloc[-1]['sklearn_test_accuracy']:.2f}",
        "lastK": int(a8.iloc[-1]["k"]),
        "maxDiffAll": f"{100 * max_difference:.2f}",
    }, os.path.join(OUTPUT_DIR, "summary_macros.tex"))

    heading("Summary")

    print(f"Best normal k-NN   : k={best_normal_k}, accuracy={best_normal_accuracy:.4f}")
    print(f"Best weighted k-NN : k={best_weighted_k}, accuracy={best_weighted_accuracy:.4f}")
    print(f"Largest manual vs sklearn difference over all k: {100 * max_difference:.2f} percentage points")
    print("Files written to 'outputs/' and 'figures/'.")
