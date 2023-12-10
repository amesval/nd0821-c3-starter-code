import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from pathlib import Path
import pickle


def test_train_model(train_model_function):
    """Test trained model."""
    X = np.array([[1,1],[2,2], [3, 3]])
    y = np.array([0, 1, 0])
    model = train_model_function(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 3


def test_save_model_attributes(save_model_attributes_function):
    """Test function to save model."""
    model_atts = {
        'model': LogisticRegression(),
        'encoder': OneHotEncoder(),
        'lb': LabelBinarizer(),
        'cat_features': []
    }
    filename = Path.cwd().joinpath("test.pkl")
    save_model_attributes_function(model_atts,"test.pkl")
    assert filename.exists()
    filename.unlink()


def test_load_model_attributes(load_model_attributes_function):
    """Test funcion to load model."""
    model_atts = {
        'model': LogisticRegression(),
        'encoder': OneHotEncoder(),
        'lb': LabelBinarizer(),
        'cat_features': []
    }
    filename = Path.cwd().joinpath("test.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(model_atts, f)
    new_model_atts = load_model_attributes_function(filename)
    assert type(new_model_atts) == type(model_atts)
    assert set(new_model_atts.keys()) == set(model_atts.keys())
    for key in model_atts:
        assert type(new_model_atts[key]) == type(model_atts[key])
    filename.unlink()


def test_compute_model_metrics(compute_model_metrics_function):
    """Test values of metrics computation"""
    y_true = np.array([1,0,1])
    y_pred = np.array([0,1,1])
    precision, recall, fbeta = compute_model_metrics_function(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(inference_function):
    """Test inference output."""
    dataset = pd.DataFrame({'sex': ['male', 'female', 'male'], 
                            'age': [15, 20, 35],
                            'salary': ['<=50K', '>50K', '>50K']})
    X = np.array([[0, 1, 15],[1,0,20], [0, 1, 35]])
    y = np.array([1, 0, 0]).ravel()
    model = LogisticRegression()
    model.fit(X, y)
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoder.fit(dataset[['sex']].values)
    lb = LabelBinarizer()
    lb.fit(y)
    model_atts = {
        'model': model,
        'encoder': encoder,
        'lb': lb,
        'cat_features': ['sex']
    }
    preds = inference_function(model_atts, dataset.drop('salary', axis=1))
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 3

