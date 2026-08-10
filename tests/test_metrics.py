"""
Tests for modules.metrics.

conftest.py already inserts src/ into sys.path, so the bare module
import ``from modules.metrics import ...`` works without any path hacks.

Test classes
------------
TestRegressionMetrics    — mse, rmse, mae, r2_score
TestClassificationMetrics — accuracy, precision, recall, f1_score,
                            confusion_matrix
TestClassificationReport  — classification_report (string output)
TestRegressionSummary     — regression_summary (string output)
TestEvaluate              — evaluate() for both tasks
"""
import numpy as np
import pytest

from modules.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    evaluate,
    f1_score,
    mae,
    mse,
    precision,
    r2_score,
    recall,
    regression_summary,
    rmse,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_reg():
    """Perfect regression predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return y_true, y_pred


@pytest.fixture
def perfect_clf():
    """Perfect binary classification predictions (0/1 labels)."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    return y_true, y_pred


@pytest.fixture
def mixed_clf():
    """Mixed binary classification: 3 correct out of 4."""
    #               TP   TN   FP   TN
    y_true = np.array([1,   0,   0,   0])
    y_pred = np.array([1,   0,   1,   0])
    # TP=1, TN=2, FP=1, FN=0
    return y_true, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestRegressionMetrics:

    # --- mse ---

    def test_mse_perfect_predictions(self, perfect_reg):
        y_true, y_pred = perfect_reg
        np.testing.assert_almost_equal(mse(y_true, y_pred), 0.0)

    def test_mse_known_value(self):
        # both errors = 1 → MSE = (1² + 1²) / 2 = 1.0
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        np.testing.assert_almost_equal(mse(y_true, y_pred), 1.0)

    def test_mse_asymmetric_errors(self):
        # errors: 1, 2 → MSE = (1 + 4) / 2 = 2.5
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        np.testing.assert_almost_equal(mse(y_true, y_pred), 2.5)

    def test_mse_handles_2d_input(self):
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_almost_equal(mse(y_true, y_pred), 0.0)

    def test_mse_returns_float(self, perfect_reg):
        y_true, y_pred = perfect_reg
        assert isinstance(mse(y_true, y_pred), float)

    # --- rmse ---

    def test_rmse_perfect_predictions(self, perfect_reg):
        y_true, y_pred = perfect_reg
        np.testing.assert_almost_equal(rmse(y_true, y_pred), 0.0)

    def test_rmse_equals_sqrt_of_mse(self):
        y_true = np.array([1.0, 3.0, 5.0])
        y_pred = np.array([2.0, 2.0, 4.0])
        np.testing.assert_almost_equal(rmse(y_true, y_pred), np.sqrt(mse(y_true, y_pred)))

    def test_rmse_known_value(self):
        # MSE = 4.0 → RMSE = 2.0
        y_true = np.zeros(4)
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])
        np.testing.assert_almost_equal(rmse(y_true, y_pred), 2.0)

    # --- mae ---

    def test_mae_perfect_predictions(self, perfect_reg):
        y_true, y_pred = perfect_reg
        np.testing.assert_almost_equal(mae(y_true, y_pred), 0.0)

    def test_mae_known_value(self):
        # |1-2| + |2-2| + |3-2| = 1 + 0 + 1 = 2, mean = 2/3
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        np.testing.assert_almost_equal(mae(y_true, y_pred), 2.0 / 3.0)

    def test_mae_is_less_than_or_equal_to_rmse(self):
        # MAE ≤ RMSE (Jensen's inequality) for any y
        rng = np.random.default_rng(7)
        y_true = rng.normal(0, 1, 50)
        y_pred = rng.normal(0, 1, 50)
        assert mae(y_true, y_pred) <= rmse(y_true, y_pred) + 1e-10

    # --- r2_score ---

    def test_r2_perfect_fit(self, perfect_reg):
        y_true, y_pred = perfect_reg
        np.testing.assert_almost_equal(r2_score(y_true, y_pred), 1.0)

    def test_r2_mean_predictor_is_zero(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])   # mean of y_true
        np.testing.assert_almost_equal(r2_score(y_true, y_pred), 0.0)

    def test_r2_negative_for_bad_predictions(self):
        # predicting all zeros for [1,2,3] is worse than mean baseline
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.zeros(3)
        assert r2_score(y_true, y_pred) < 0.0

    def test_r2_constant_y_true_returns_zero(self):
        # SS_tot = 0 → undefined, convention is 0.0
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        np.testing.assert_almost_equal(r2_score(y_true, y_pred), 0.0)

    def test_r2_returns_float(self, perfect_reg):
        y_true, y_pred = perfect_reg
        assert isinstance(r2_score(y_true, y_pred), float)


# ──────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestClassificationMetrics:

    # --- accuracy ---

    def test_accuracy_perfect_classifier(self, perfect_clf):
        y_true, y_pred = perfect_clf
        np.testing.assert_almost_equal(accuracy(y_true, y_pred), 1.0)

    def test_accuracy_all_wrong(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])   # completely inverted
        np.testing.assert_almost_equal(accuracy(y_true, y_pred), 0.0)

    def test_accuracy_mixed_predictions(self, mixed_clf):
        y_true, y_pred = mixed_clf
        # TP=1, TN=2, FP=1, FN=0 → 3 correct out of 4
        np.testing.assert_almost_equal(accuracy(y_true, y_pred), 3.0 / 4.0)

    def test_accuracy_probability_inputs(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.3])   # probabilities → all correct
        np.testing.assert_almost_equal(accuracy(y_true, y_pred), 1.0)

    def test_accuracy_custom_threshold(self):
        # With threshold=0.35: 0.4 is now positive → one more FP
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.4, 0.3, 0.1])
        # threshold=0.5  → predicted [1,0,0,0] → 3 correct (acc=0.75)
        # threshold=0.35 → predicted [1,1,0,0] → 4 correct (acc=1.00)
        np.testing.assert_almost_equal(accuracy(y_true, y_pred, threshold=0.35), 1.0)
        np.testing.assert_almost_equal(accuracy(y_true, y_pred, threshold=0.50), 0.75)

    # --- precision ---

    def test_precision_perfect(self, perfect_clf):
        y_true, y_pred = perfect_clf
        np.testing.assert_almost_equal(precision(y_true, y_pred), 1.0)

    def test_precision_no_positive_predictions_returns_zero(self):
        # All-negative predictor → no TP, no FP → precision undefined → 0.0
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        np.testing.assert_almost_equal(precision(y_true, y_pred), 0.0)

    def test_precision_all_positive_predictor(self):
        # TP=2, FP=2 → precision = 2/4 = 0.5
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        np.testing.assert_almost_equal(precision(y_true, y_pred), 0.5)

    def test_precision_known_value(self, mixed_clf):
        y_true, y_pred = mixed_clf
        # TP=1, FP=1 → precision = 0.5
        np.testing.assert_almost_equal(precision(y_true, y_pred), 0.5)

    # --- recall ---

    def test_recall_perfect(self, perfect_clf):
        y_true, y_pred = perfect_clf
        np.testing.assert_almost_equal(recall(y_true, y_pred), 1.0)

    def test_recall_no_actual_positives_returns_zero(self):
        # y_true has no positives → TP + FN = 0 → recall undefined → 0.0
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        np.testing.assert_almost_equal(recall(y_true, y_pred), 0.0)

    def test_recall_all_negative_predictor(self):
        # TP=0, FN=2 → recall = 0.0
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        np.testing.assert_almost_equal(recall(y_true, y_pred), 0.0)

    def test_recall_all_positive_predictor(self):
        # TP=2, FN=0 → recall = 1.0
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        np.testing.assert_almost_equal(recall(y_true, y_pred), 1.0)

    def test_recall_known_value(self, mixed_clf):
        y_true, y_pred = mixed_clf
        # TP=1, FN=0 → recall = 1.0
        np.testing.assert_almost_equal(recall(y_true, y_pred), 1.0)

    # --- f1_score ---

    def test_f1_perfect(self, perfect_clf):
        y_true, y_pred = perfect_clf
        np.testing.assert_almost_equal(f1_score(y_true, y_pred), 1.0)

    def test_f1_zero_when_no_correct_predictions(self):
        # All-wrong: both precision and recall are 0 → f1 = 0.0
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        np.testing.assert_almost_equal(f1_score(y_true, y_pred), 0.0)

    def test_f1_known_value(self, mixed_clf):
        y_true, y_pred = mixed_clf
        # precision=0.5, recall=1.0 → f1 = 2*(0.5*1.0)/(0.5+1.0) = 2/3
        np.testing.assert_almost_equal(f1_score(y_true, y_pred), 2.0 / 3.0)

    def test_f1_with_probability_inputs(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])   # → perfect 0/1 after threshold
        np.testing.assert_almost_equal(f1_score(y_true, y_pred), 1.0)

    # --- confusion_matrix ---

    def test_confusion_matrix_perfect(self, perfect_clf):
        y_true, y_pred = perfect_clf
        cm = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm, np.array([[2, 0], [0, 2]]))

    def test_confusion_matrix_all_positive_predictor(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        # TN=0, FP=2, FN=0, TP=2
        cm = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm, np.array([[0, 2], [0, 2]]))

    def test_confusion_matrix_all_negative_predictor(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        # TN=2, FP=0, FN=2, TP=0
        cm = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm, np.array([[2, 0], [2, 0]]))

    def test_confusion_matrix_mixed(self):
        #          TP   TN   FN   FP
        y_true = np.array([1,   0,   1,   0])
        y_pred = np.array([1,   0,   0,   1])
        # TN=1, FP=1, FN=1, TP=1
        cm = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm, np.array([[1, 1], [1, 1]]))

    def test_confusion_matrix_dtype_is_int(self, perfect_clf):
        y_true, y_pred = perfect_clf
        cm = confusion_matrix(y_true, y_pred)
        assert cm.dtype == np.int32

    def test_confusion_matrix_shape(self, perfect_clf):
        y_true, y_pred = perfect_clf
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)


# ──────────────────────────────────────────────────────────────────────────────
# classification_report
# ──────────────────────────────────────────────────────────────────────────────

class TestClassificationReport:

    def test_contains_all_keywords(self, perfect_clf):
        y_true, y_pred = perfect_clf
        report = classification_report(y_true, y_pred)
        for keyword in ["Threshold", "Accuracy", "Precision", "Recall",
                        "F1", "Confusion", "TN", "FP", "FN", "TP"]:
            assert keyword in report, f"'{keyword}' not found in report"

    def test_perfect_accuracy_shown(self, perfect_clf):
        y_true, y_pred = perfect_clf
        report = classification_report(y_true, y_pred)
        # Perfect accuracy → 1.0000 must appear somewhere
        assert "1.0000" in report

    def test_threshold_value_shown(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0.9, 0.1])
        report = classification_report(y_true, y_pred, threshold=0.7)
        assert "0.70" in report

    def test_returns_string(self, perfect_clf):
        y_true, y_pred = perfect_clf
        assert isinstance(classification_report(y_true, y_pred), str)

    def test_multiline_output(self, perfect_clf):
        y_true, y_pred = perfect_clf
        report = classification_report(y_true, y_pred)
        assert report.count("\n") >= 5


# ──────────────────────────────────────────────────────────────────────────────
# regression_summary
# ──────────────────────────────────────────────────────────────────────────────

class TestRegressionSummary:

    def test_returns_string(self, perfect_reg):
        y_true, y_pred = perfect_reg
        assert isinstance(regression_summary(y_true, y_pred), str)

    def test_contains_all_metric_names(self, perfect_reg):
        y_true, y_pred = perfect_reg
        result = regression_summary(y_true, y_pred)
        for metric in ["MSE", "RMSE", "MAE", "R2"]:
            assert metric in result, f"'{metric}' not found in regression_summary"

    def test_perfect_values_shown(self, perfect_reg):
        y_true, y_pred = perfect_reg
        result = regression_summary(y_true, y_pred)
        # Perfect fit → R2 = 1.000000 and all errors = 0.000000
        assert "1.000000" in result
        assert "0.000000" in result

    def test_multiline_output(self, perfect_reg):
        y_true, y_pred = perfect_reg
        result = regression_summary(y_true, y_pred)
        assert result.count("\n") >= 4


# ──────────────────────────────────────────────────────────────────────────────
# evaluate()
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluate:

    def test_regression_returns_correct_keys(self, perfect_reg):
        y_true, y_pred = perfect_reg
        result = evaluate(y_true, y_pred, task="regression")
        assert set(result.keys()) == {"mse", "rmse", "mae", "r2"}

    def test_regression_perfect_values(self, perfect_reg):
        y_true, y_pred = perfect_reg
        result = evaluate(y_true, y_pred, task="regression")
        np.testing.assert_almost_equal(result["mse"],  0.0)
        np.testing.assert_almost_equal(result["rmse"], 0.0)
        np.testing.assert_almost_equal(result["mae"],  0.0)
        np.testing.assert_almost_equal(result["r2"],   1.0)

    def test_regression_values_match_individual_functions(self):
        rng = np.random.default_rng(99)
        y_true = rng.normal(0, 1, 20)
        y_pred = rng.normal(0, 1, 20)
        result = evaluate(y_true, y_pred, task="regression")
        np.testing.assert_almost_equal(result["mse"],  mse(y_true, y_pred))
        np.testing.assert_almost_equal(result["rmse"], rmse(y_true, y_pred))
        np.testing.assert_almost_equal(result["mae"],  mae(y_true, y_pred))
        np.testing.assert_almost_equal(result["r2"],   r2_score(y_true, y_pred))

    def test_classification_returns_correct_keys(self, perfect_clf):
        y_true, y_pred = perfect_clf
        result = evaluate(y_true, y_pred, task="classification")
        assert set(result.keys()) == {
            "accuracy", "precision", "recall", "f1", "confusion_matrix"
        }

    def test_classification_perfect_scalar_values(self, perfect_clf):
        y_true, y_pred = perfect_clf
        result = evaluate(y_true, y_pred, task="classification")
        np.testing.assert_almost_equal(result["accuracy"],  1.0)
        np.testing.assert_almost_equal(result["precision"], 1.0)
        np.testing.assert_almost_equal(result["recall"],    1.0)
        np.testing.assert_almost_equal(result["f1"],        1.0)

    def test_classification_confusion_matrix_in_result(self, perfect_clf):
        y_true, y_pred = perfect_clf
        result = evaluate(y_true, y_pred, task="classification")
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            np.array([[2, 0], [0, 2]])
        )

    def test_default_task_is_regression(self, perfect_reg):
        y_true, y_pred = perfect_reg
        result = evaluate(y_true, y_pred)   # no task= argument
        assert "mse" in result
        assert "r2" in result

    def test_invalid_task_raises_value_error(self, perfect_reg):
        y_true, y_pred = perfect_reg
        with pytest.raises(ValueError, match="task"):
            evaluate(y_true, y_pred, task="clustering")

    def test_classification_threshold_forwarded(self):
        # threshold=0.35 lets 0.4 be classified as positive
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.4, 0.3, 0.1])
        result_low  = evaluate(y_true, y_pred, task="classification", threshold=0.35)
        result_high = evaluate(y_true, y_pred, task="classification", threshold=0.50)
        # Lower threshold captures both positives → recall = 1.0
        np.testing.assert_almost_equal(result_low["recall"],  1.0)
        # Higher threshold misses the 0.4 sample → recall = 0.5
        np.testing.assert_almost_equal(result_high["recall"], 0.5)
