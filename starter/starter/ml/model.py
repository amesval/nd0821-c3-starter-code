from sklearn.metrics import fbeta_score, precision_score, recall_score
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from starter.ml.data import process_data
import pickle


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # model = LogisticRegression(max_iter=1000, random_state=123)
    model = RandomForestClassifier(n_estimators=600,
                                   max_depth=15,
                                   random_state=123456789,
                                   max_features=None)
    model.fit(X_train, y_train)
    return model


def save_model_attributes(model_attributes, filename):
    """
    Save model attributes to pickle file.

    Inputs
    ------
    model_attributes : dict
        Dictionary which contains different attributes
        to make predictions with the model:
            a. model: Trained machine learning model.
            b. encoder: OneHotEncoder for categorical variables.
            c. lb: LabelBinarizer for binary variables.
            d. cat_features: Categorical features use for preprocessing.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model_attributes, f)


def load_model_attributes(filename):
    """
    Load model attributes.

    Inputs
    ------
    filename: filename for model attributes.
    Returns
    -------
    model_attributes : dict
        Dictionary which contains different attributes
        to make predictions with the model:
            a. model: Trained machine learning model.
            b. encoder: OneHotEncoder for categorical variables.
            c. lb: LabelBinarizer for binary variables.
            d. cat_features: Categorical features use for preprocessing.
    """
    with open(filename, 'rb') as f:
        model_attributes = pickle.load(f)
    return model_attributes


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model_attributes, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model_attributes : dict
        Dictionary which contains different attributes
        to make predictions with the model:
            a. model: Trained machine learning model.
            b. encoder: OneHotEncoder for categorical variables.
            c. lb: LabelBinarizer for binary variables.
            d. cat_features: Categorical features use for preprocessing.
    X : pd.DataFrame
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    model = model_attributes['model']
    encoder = model_attributes['encoder']
    lb = model_attributes['lb']
    cat_features = model_attributes['cat_features']

    X_preprocessed, _, encoder, lb = process_data(
        X,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb)
    preds = model.predict(X_preprocessed)
    return preds


def performance_for_feature_slice(dataset, feature, model_attributes):
    """ Calculate performance over a particular feature.
    The metrics are calculated for each feature value.
    The results are written into "./model/slice_output.txt".

    Inputs
    ------
    dataset: pd.DataFrame
        Dataset which includes the training features.
    feature: str
        Dataset column name.
    model_attributes : dict
        Dictionary which contains different attributes
        to make predictions with the model:
            a. model: Trained machine learning model.
            b. encoder: OneHotEncoder for categorical variables.
            c. lb: LabelBinarizer for binary variables.
            d. cat_features: Categorical features use for preprocessing.
    """
    model = model_attributes['model']
    encoder = model_attributes['encoder']
    lb = model_attributes['lb']
    cat_features = model_attributes['cat_features']
    if feature not in cat_features:
        raise ValueError("Invalid value. The feature should be\
                          a categorical feature.")
    with open("./model/slice_output.txt", 'w') as f:
        for value in dataset[feature].unique():
            feat_filter = dataset[feature] == value
            X, y, encoder, lb = process_data(
                dataset[feat_filter],
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder, lb=lb
            )
            y_preds = model.predict(X)
            precision, recall, f1 = compute_model_metrics(y, y_preds)
            f.write(f"feature: {feature}    value: {value}    ")
            f.write(f"precision: {np.round(precision,4)}    ")
            f.write(f"recall: {np.round(recall, 4)}    ")
            f.write(f"f1: {np.round(f1, 4)}\n")
