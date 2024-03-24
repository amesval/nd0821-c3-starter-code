import pytest
from ml.model import (
    train_model,
    save_model_attributes,
    load_model_attributes,
    compute_model_metrics,
    inference
    )


@pytest.fixture(scope='session')
def train_model_function():
    return train_model


@pytest.fixture(scope='session')
def save_model_attributes_function():
    return save_model_attributes


@pytest.fixture(scope='session')
def load_model_attributes_function():
    return load_model_attributes


@pytest.fixture(scope='session')
def compute_model_metrics_function():
    return compute_model_metrics


@pytest.fixture(scope='session')
def inference_function():
    return inference
