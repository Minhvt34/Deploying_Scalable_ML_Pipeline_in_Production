
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from model import (
    train_model,
    compute_model_metrics,
    inference,
)
from data import process_data

# upload the census_cleaned.csv file
DATA_PATH = "data/census_cleaned.csv"
data = pd.read_csv(DATA_PATH)

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

# prepare the data
training_datasets, testing_datasets, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)


def test_train_model():
    """
    Test the model training
    """
    # test the train_model function
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        training_datasets, testing_datasets, test_size=0.2, random_state=3
    )
    model = train_model(training_data, training_labels, random_state=3)
    assert model is not None
    # assert a prediction
    # extract the first row of the numpy array data
    first_row = testing_data[10]
    # shape the data to be in the correct format for the model
    first_row = first_row.reshape(1, -1)
    # print the type that the prediction output os
    assert isinstance(model.predict(first_row)[0], np.int64)
    assert testing_labels[0] == 1


def test_compute_metrics():
    """
    This functions tests the computation of the metrics
    """
    # test the train_model function
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        training_datasets, testing_datasets, test_size=0.2, random_state=10
    )
    model = train_model(training_data, training_labels, random_state=10)
    predictions = model.predict(testing_data)
    precision, recall, fbeta = compute_model_metrics(testing_labels, predictions)
    assert precision >= 0.70
    assert recall >= 0.52
    assert fbeta >= 0.6


def test_inference():
    """
    Test the inference function
    """
    # test the train_and_test_on_slices function
    training_data, testing_data, training_labels, _ = train_test_split(
        training_datasets, testing_datasets, test_size=0.2, random_state=10
    )
    model = train_model(training_data, training_labels, random_state=10)
    assert all(model.predict(testing_data)) == all(inference(model, testing_data))
