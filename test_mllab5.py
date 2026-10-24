import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from mllab5 import (
    build_binary_dataset,
    fit_imputer,
    apply_imputer,
    fit_category_maps,
    encode_categorical,
    signed_log_transform,
    bubble_sort,
    insertion_sort,
    selection_sort,
    merge_sort,
    sort_distance_pairs,
    calculate_distance,
    get_sorted_neighbors,
    select_k_neighbors,
    majority_vote,
    weighted_vote,
    predict_one,
    predict,
    predict_many_k,
    accuracy_score_manual,
    ManualKNN,
    compare_over_k,
    plot_accuracy_comparison,
    best_k,
    classification_metrics,
    confusion_counts,
    write_metrics_table,
    write_latex_table,
    write_summary_macros,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def make_feature_frame():
    """Small dataframe that contains every feature column."""
    data = {column: [1.0, 2.0, 3.0, 4.0] for column in NUMERIC_FEATURES}
    data["token0_symbol"] = ["A", "A", "B", "A"]
    data["token1_symbol"] = ["X", "Y", "Y", "Y"]
    return pd.DataFrame(data)


def make_blobs():
    """Two well separated groups of points."""
    X = np.array([
        [1, 1], [1, 2], [2, 1], [2, 2],
        [8, 8], [8, 9], [9, 8], [9, 9],
    ], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


class TestPreprocessing(unittest.TestCase):

    def test_build_binary_dataset_balanced(self):
        df = pd.DataFrame({
            "fee_tier": [100] * 10 + [3000] * 10 + [500] * 5,
            "value": range(25),
        })

        result = build_binary_dataset(df, classes=[100, 3000], sample_per_class=6)

        self.assertEqual(len(result), 12)
        self.assertEqual(set(result["fee_tier"]), {100, 3000})
        self.assertEqual((result["fee_tier"] == 100).sum(), 6)

    def test_build_binary_dataset_uses_available_rows(self):
        df = pd.DataFrame({"fee_tier": [100] * 2 + [3000] * 10, "value": range(12)})

        result = build_binary_dataset(df, classes=[100, 3000], sample_per_class=6)

        self.assertEqual((result["fee_tier"] == 100).sum(), 2)

    def test_build_binary_dataset_needs_two_classes(self):
        df = pd.DataFrame({"fee_tier": [100, 3000], "value": [1, 2]})

        with self.assertRaises(ValueError):
            build_binary_dataset(df, classes=[100, 100])

    def test_build_binary_dataset_missing_target(self):
        with self.assertRaises(ValueError):
            build_binary_dataset(pd.DataFrame({"value": [1, 2]}))

    def test_imputer_median_and_mode(self):
        df = make_feature_frame()
        df.loc[0, "gas_used"] = np.nan
        df.loc[1, "token0_symbol"] = np.nan

        fill_values = fit_imputer(df, "median")
        filled = apply_imputer(df, fill_values)

        self.assertEqual(filled["gas_used"].isna().sum(), 0)
        self.assertEqual(filled["gas_used"][0], 3.0)
        self.assertEqual(filled["token0_symbol"][1], "A")

    def test_imputer_mean(self):
        df = make_feature_frame()
        fill_values = fit_imputer(df, "mean")

        self.assertAlmostEqual(fill_values["timestamp"], 2.5)

    def test_imputer_invalid_strategy(self):
        with self.assertRaises(ValueError):
            fit_imputer(make_feature_frame(), "mode")

    def test_frequency_encoding(self):
        df = make_feature_frame()
        maps = fit_category_maps(df)
        encoded = encode_categorical(df, maps)

        self.assertAlmostEqual(encoded["token0_symbol"][0], 0.75)
        self.assertAlmostEqual(encoded["token0_symbol"][2], 0.25)

    def test_unseen_category_is_zero(self):
        df = make_feature_frame()
        maps = fit_category_maps(df)

        new = df.copy()
        new.loc[0, "token0_symbol"] = "NEW"

        self.assertEqual(encode_categorical(new, maps)["token0_symbol"][0], 0.0)

    def test_signed_log_keeps_sign(self):
        df = make_feature_frame()
        df.loc[0, "amount0"] = -100.0

        result = signed_log_transform(df)

        self.assertLess(result["amount0"][0], 0)
        self.assertAlmostEqual(result["amount0"][0], -np.log1p(100.0))


class TestSorting(unittest.TestCase):

    def setUp(self):
        self.data = [5, 2, 9, 1, 5, 6, -3, 0]
        self.functions = [bubble_sort, insertion_sort, selection_sort, merge_sort]

    def test_all_sorts_match_python(self):
        for function in self.functions:
            self.assertEqual(function(self.data), sorted(self.data))

    def test_sorts_do_not_modify_input(self):
        original = list(self.data)

        for function in self.functions:
            function(self.data)

        self.assertEqual(self.data, original)

    def test_empty_and_single_item(self):
        for function in self.functions:
            self.assertEqual(function([]), [])
            self.assertEqual(function([7]), [7])

    def test_sort_tuples(self):
        pairs = [(2.0, 0, 1), (1.0, 1, 0), (1.0, 2, 1)]

        for function in self.functions:
            self.assertEqual(function(pairs), [(1.0, 1, 0), (1.0, 2, 1), (2.0, 0, 1)])

    def test_sort_distance_pairs_selects_algorithm(self):
        self.assertEqual(sort_distance_pairs([3, 1, 2], "bubble"), [1, 2, 3])

    def test_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            sort_distance_pairs([3, 1, 2], "quick")


class TestDistance(unittest.TestCase):

    def test_euclidean(self):
        self.assertAlmostEqual(calculate_distance([0, 0], [3, 4]), 5.0)

    def test_manhattan(self):
        self.assertAlmostEqual(calculate_distance([1, 2], [4, 6], "manhattan"), 7.0)

    def test_chebyshev(self):
        self.assertAlmostEqual(calculate_distance([1, 2], [4, 6], "chebyshev"), 4.0)

    def test_zero_distance(self):
        self.assertEqual(calculate_distance([1, 2, 3], [1, 2, 3]), 0.0)

    def test_unknown_metric(self):
        with self.assertRaises(ValueError):
            calculate_distance([0], [1], "cosine")


class TestNeighboursAndVoting(unittest.TestCase):

    def test_sorted_neighbors_order(self):
        X, y = make_blobs()

        neighbors = get_sorted_neighbors(X, y, np.array([1.0, 1.0]))
        distances = [item[0] for item in neighbors]

        self.assertEqual(distances, sorted(distances))
        self.assertEqual(neighbors[0][1], 0)

    def test_equal_distance_tie_break_uses_index(self):
        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        y = np.array([0, 1, 1])

        neighbors = get_sorted_neighbors(X, y, np.array([0.0, 0.0]))

        self.assertEqual([item[1] for item in neighbors], [0, 1, 2])

    def test_select_k_neighbors(self):
        X, y = make_blobs()
        neighbors = get_sorted_neighbors(X, y, np.array([1.0, 1.0]))

        self.assertEqual(len(select_k_neighbors(neighbors, 3)), 3)

        with self.assertRaises(ValueError):
            select_k_neighbors(neighbors, 0)

        with self.assertRaises(ValueError):
            select_k_neighbors(neighbors, 100)

    def test_majority_vote(self):
        neighbors = [(1.0, 0, 1), (2.0, 1, 1), (3.0, 2, 0)]

        self.assertEqual(majority_vote(neighbors), 1)

    def test_majority_vote_tie_goes_to_smaller_label(self):
        neighbors = [(1.0, 0, 3000), (2.0, 1, 100)]

        self.assertEqual(majority_vote(neighbors), 100)

    def test_weighted_vote_prefers_closer_neighbor(self):
        neighbors = [(0.1, 0, 0), (5.0, 1, 1), (5.0, 2, 1)]

        self.assertEqual(majority_vote(neighbors), 1)
        self.assertEqual(weighted_vote(neighbors), 0)

    def test_weighted_vote_zero_distance(self):
        neighbors = [(0.0, 0, 1), (1.0, 1, 0)]

        self.assertEqual(weighted_vote(neighbors), 1)

    def test_weighted_vote_tie_goes_to_smaller_label(self):
        neighbors = [(2.0, 0, 1), (2.0, 1, 0)]

        self.assertEqual(weighted_vote(neighbors), 0)


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_blobs()

    def test_predict_one(self):
        self.assertEqual(predict_one(self.X, self.y, np.array([1.5, 1.5]), k=3), 0)
        self.assertEqual(predict_one(self.X, self.y, np.array([8.5, 8.5]), k=3), 1)

    def test_predict_length(self):
        result = predict(self.X, self.y, self.X, k=3)

        self.assertEqual(len(result), len(self.y))
        self.assertTrue(np.array_equal(result, self.y))

    def test_all_sorting_algorithms_agree(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 3))
        y = (X[:, 0] > 0).astype(int)
        queries = rng.normal(size=(10, 3))

        expected = predict(X, y, queries, k=5, sorting_algorithm="merge")

        for algorithm in ("bubble", "insertion", "selection"):
            result = predict(X, y, queries, k=5, sorting_algorithm=algorithm)
            self.assertTrue(np.array_equal(result, expected))

    def test_matches_sklearn(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        queries = rng.normal(size=(20, 4))

        for k in (1, 3, 5, 7):
            model = KNeighborsClassifier(n_neighbors=k).fit(X, y)

            self.assertTrue(
                np.array_equal(predict(X, y, queries, k=k), model.predict(queries))
            )

    def test_weighted_matches_sklearn(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        queries = rng.normal(size=(20, 4))

        model = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X, y)
        result = predict(X, y, queries, k=5, weighted=True)

        self.assertTrue(np.array_equal(result, model.predict(queries)))

    def test_predict_many_k_matches_predict(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(30, 2))
        y = (X[:, 0] > 0).astype(int)
        queries = rng.normal(size=(8, 2))

        many = predict_many_k(X, y, queries, [1, 3, 5])

        for k in (1, 3, 5):
            self.assertTrue(np.array_equal(many[k], predict(X, y, queries, k=k)))

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            predict(self.X, self.y, self.X, k=0)

        with self.assertRaises(ValueError):
            predict(self.X, self.y, self.X, k=100)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            predict(self.X, self.y[:3], self.X, k=1)

    def test_accuracy_score(self):
        self.assertAlmostEqual(accuracy_score_manual([1, 0, 1, 1], [1, 0, 0, 1]), 0.75)
        self.assertEqual(accuracy_score_manual([], []), 0.0)

        with self.assertRaises(ValueError):
            accuracy_score_manual([1, 0], [1])


class TestManualKNN(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_blobs()

    def test_fit_returns_self(self):
        model = ManualKNN(k=3)

        self.assertIs(model.fit(self.X, self.y), model)

    def test_predict_and_score(self):
        model = ManualKNN(k=3).fit(self.X, self.y)

        self.assertTrue(np.array_equal(model.predict(self.X), self.y))
        self.assertEqual(model.score(self.X, self.y), 1.0)

    def test_predict_before_fit(self):
        with self.assertRaises(ValueError):
            ManualKNN().predict(self.X)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            ManualKNN(k=0)

        with self.assertRaises(ValueError):
            ManualKNN(sorting_algorithm="quick")

        with self.assertRaises(ValueError):
            ManualKNN(k=50).fit(self.X, self.y)

        with self.assertRaises(ValueError):
            ManualKNN().fit(self.X, self.y[:3])

    def test_weighted_option(self):
        model = ManualKNN(k=3, weighted=True).fit(self.X, self.y)

        self.assertEqual(model.score(self.X, self.y), 1.0)


class TestComparisonAndFiles(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] > 0).astype(int)

        self.X_train, self.X_test = X[:60], X[60:]
        self.y_train, self.y_test = y[:60], y[60:]

    def test_compare_over_k_columns(self):
        result = compare_over_k(
            self.X_train, self.X_test, self.y_train, self.y_test, [1, 3, 5]
        )

        self.assertEqual(len(result), 3)
        self.assertIn("manual_test_accuracy", result.columns)
        self.assertIn("sklearn_test_accuracy", result.columns)
        self.assertIn("sklearn_train_accuracy", result.columns)

    def test_manual_equals_sklearn_over_k(self):
        result = compare_over_k(
            self.X_train, self.X_test, self.y_train, self.y_test, [1, 3, 5, 7]
        )

        self.assertTrue(
            np.allclose(result["manual_test_accuracy"], result["sklearn_test_accuracy"])
        )

    def test_weighted_comparison(self):
        result = compare_over_k(
            self.X_train, self.X_test, self.y_train, self.y_test, [1, 3, 5], weighted=True
        )

        self.assertTrue(
            np.allclose(result["manual_test_accuracy"], result["sklearn_test_accuracy"])
        )

    def test_best_k(self):
        results = pd.DataFrame({"k": [1, 3, 5], "accuracy": [0.8, 0.9, 0.9]})

        self.assertEqual(best_k(results, "accuracy"), (3, 0.9))

    def test_plot_is_saved(self):
        results = compare_over_k(
            self.X_train, self.X_test, self.y_train, self.y_test, [1, 3]
        )

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "figures", "plot.png")

            saved = plot_accuracy_comparison(results, "title", "a", "b", path, show=False)

            self.assertTrue(os.path.exists(saved))

    def test_latex_files(self):
        results = pd.DataFrame({"k": [1, 3], "acc": [0.5, 0.75]})

        with tempfile.TemporaryDirectory() as folder:
            table = write_latex_table(results, ["acc"], os.path.join(folder, "t.tex"))
            macros = write_summary_macros({"bestK": 3}, os.path.join(folder, "m.tex"))

            with open(table) as file:
                lines = file.read().splitlines()

            with open(macros) as file:
                macro_text = file.read()

        self.assertEqual(lines[0], "1 & 50.00\\% \\\\ \\hline")
        self.assertEqual(lines[1], "3 & 75.00\\% \\\\ \\hline")
        self.assertIn("\\newcommand{\\bestK}{3}", macro_text)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        self.y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 0])

    def test_precision_recall_f1(self):
        metrics = classification_metrics(self.y_true, self.y_pred, [0, 1])
        positive = metrics[metrics["class"] == 1].iloc[0]

        self.assertAlmostEqual(positive["precision"], 2 / 3)
        self.assertAlmostEqual(positive["recall"], 2 / 3)
        self.assertAlmostEqual(positive["f1"], 2 / 3)
        self.assertEqual(positive["support"], 3)

    def test_class_never_predicted(self):
        metrics = classification_metrics([0, 1], [0, 0], [0, 1])
        positive = metrics[metrics["class"] == 1].iloc[0]

        self.assertEqual(positive["precision"], 0.0)
        self.assertEqual(positive["f1"], 0.0)

    def test_confusion_counts(self):
        matrix = confusion_counts(self.y_true, self.y_pred, [0, 1])

        self.assertEqual(matrix.loc["true 1", "pred 1"], 2)
        self.assertEqual(matrix.loc["true 0", "pred 1"], 1)
        self.assertEqual(int(matrix.to_numpy().sum()), 8)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            classification_metrics([1, 0], [1], [0, 1])

    def test_metrics_table_file(self):
        metrics = classification_metrics(self.y_true, self.y_pred, [0, 1])

        with tempfile.TemporaryDirectory() as folder:
            path = write_metrics_table(metrics, os.path.join(folder, "m.tex"))

            with open(path) as file:
                lines = file.read().splitlines()

        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].endswith("\\\\ \\hline"))


if __name__ == "__main__":
    unittest.main()
