import pytest

from app.metrics import macro_f1, per_label_scores, percentile


def test_macro_f1_matches_hand_calculation():
    # A: P=1.0, R=0.5 -> F1=0.6667 ; B: P=2/3, R=1.0 -> F1=0.8 ; macro = 0.7333
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    assert macro_f1(y_true, y_pred, ["A", "B"]) == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_four_classes_with_invalid_prediction():
    labels = ["P", "A", "S", "G"]
    y_true = ["P", "P", "A", "A", "S", "S", "G", "G"]
    y_pred = ["P", "P", "A", "P", "S", None, "G", "G"]
    scores = per_label_scores(y_true, y_pred, labels)
    # P: tp=2 fp=1 fn=0 -> P=2/3 R=1 F1=0.8
    assert scores["P"]["f1"] == pytest.approx(0.8)
    # A: tp=1 fp=0 fn=1 -> F1=2/3 ; S: tp=1 fp=0 fn=1 (None is a miss) -> 2/3 ; G: 1.0
    assert scores["A"]["f1"] == pytest.approx(2 / 3)
    assert scores["S"]["f1"] == pytest.approx(2 / 3)
    assert scores["G"]["f1"] == pytest.approx(1.0)
    assert macro_f1(y_true, y_pred, labels) == pytest.approx((0.8 + 2 / 3 + 2 / 3 + 1.0) / 4)


def test_perfect_and_label_without_predictions():
    assert macro_f1(["A", "B"], ["A", "B"], ["A", "B"]) == 1.0
    # B never predicted -> F1 0 for B, not a division error.
    assert macro_f1(["A", "B"], ["A", "A"], ["A", "B"]) == pytest.approx((2 / 3 + 0) / 2)


def test_percentile_nearest_rank():
    values = list(range(1, 21))  # 1..20
    assert percentile(values, 95) == 19
    assert percentile(values, 50) == 10
    assert percentile([], 95) == 0.0
