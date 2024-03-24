# Script to train machine learning model.
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import (
    train_model,
    save_model_attributes,
    load_model_attributes,
    compute_model_metrics,
    performance_for_feature_slice
    )

parser = argparse.ArgumentParser(description='Training and evaluation\
                                  of a ML model')
parser.add_argument('--csv_dataset',
                    type=str,
                    help='CSV dataset',
                    default='./data/clean_census.csv')
parser.add_argument('--model_name',
                    type=str,
                    help='Name of the trained model',
                    default='./model/model.pkl')
parser.add_argument('--test_size',
                    type=float,
                    help='Proportion of test set',
                    default=0.2
                    )
parser.add_argument('--feature_to_slice',
                    type=str,
                    help='Evaluate metrics for a fix feature',
                    default='workclass')
parser.add_argument('--random_seed',
                    type=int,
                    help='Random seed for the experiment',
                    default=123456789
                    )
args = parser.parse_args()

np.random.seed(args.random_seed)
data = pd.read_csv(args.csv_dataset)
training_set, test_set = train_test_split(data, test_size=args.test_size)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    training_set,
    categorical_features=cat_features,
    label="salary",
    training=True,
    encoder=None,
    lb=None
    )

X_test, y_test, encoder, lb = process_data(
    test_set,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
    )

# Train and save a model.
model = train_model(X_train, y_train)
model_attributes = {
    'model': model,
    'encoder': encoder,
    'lb': lb,
    'cat_features': cat_features
    }

save_model_attributes(model_attributes, args.model_name)

# Compute classification metrics in the training set.
y_preds = model.predict(X_train)
training_precision, training_recall, training_f1 = compute_model_metrics(
    y_train,
    y_preds)
print("Training set metrics:")
print(f"precision: {np.round(training_precision,4)}")
print(f"recall: {np.round(training_recall, 4)}")
print(f"f1: {np.round(training_f1, 4)}\n")


# Compute classification metrics in the test set.
model_attributes = load_model_attributes(args.model_name)
model = model_attributes['model']
y_preds = model.predict(X_test)
test_precision, test_recall, test_f1 = compute_model_metrics(y_test, y_preds)
print("Test set metrics:")
print(f"precision: {np.round(test_precision,4)}")
print(f"recall: {np.round(test_recall, 4)}")
print(f"f1: {np.round(test_f1, 4)}\n")

report_name = args.model_name.replace(".pkl", "").split("/")[-1]
with open(f'./model/metrics_report_{report_name}.txt', 'w') as f:
    f.write("Training set metrics:\n")
    f.write(f"precision: {np.round(training_precision,4)}    ")
    f.write(f"recall: {np.round(training_recall, 4)}    ")
    f.write(f"f1: {np.round(training_f1, 4)}\n")
    f.write("Test set metrics:\n")
    f.write(f"precision: {np.round(test_precision,4)}    ")
    f.write(f"recall: {np.round(test_recall, 4)}    ")
    f.write(f"f1: {np.round(test_f1, 4)}\n")

performance_for_feature_slice(test_set,
                              args.feature_to_slice,
                              model_attributes)
