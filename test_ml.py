import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """A 1,000-row sample of the census data, so the tests run fast."""
    df = pd.read_csv("data/census.csv")
    return df.sample(n=1000, random_state=42)


@pytest.fixture(scope="module")
def processed(data):
    """Split and process the sample the same way train_model.py does."""
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test


def test_train_test_split_sizes(data):
    """The split should put 80% of rows in train and 20% in test."""
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )
    assert len(train) == 800
    assert len(test) == 200
    assert list(train.columns) == list(data.columns)


def test_process_data_shapes(processed):
    """Processed train and test data should have matching column counts."""
    X_train, y_train, X_test, y_test = processed
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
    assert set(np.unique(y_train)) <= {0, 1}


def test_train_model_uses_random_forest(processed):
    """train_model should return a fitted RandomForestClassifier."""
    X_train, y_train, _, _ = processed
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")


def test_inference_returns_one_binary_prediction_per_row(processed):
    """inference should return a numpy array of 0/1 labels, one per row."""
    X_train, y_train, X_test, _ = processed
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X_test.shape[0],)
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_known_values():
    """Metrics on a hand-made example should match values worked out by hand."""
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)


def test_save_and_load_model_round_trip(processed, tmp_path):
    """A saved and reloaded model should give identical predictions."""
    X_train, y_train, X_test, _ = processed
    model = train_model(X_train, y_train)
    path = tmp_path / "model.pkl"
    save_model(model, path)
    loaded = load_model(path)
    np.testing.assert_array_equal(inference(model, X_test), inference(loaded, X_test))


def test_apply_label():
    """apply_label should turn 1/0 predictions into salary strings."""
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"
