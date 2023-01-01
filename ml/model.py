from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, random_state=42):
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
    model = RandomForestClassifier(random_state)
    model.fit(X_train,y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds=model.predict(X)
    return preds
    
def train_and_test_on_slices(train,test,test_size_def=.2):
    metrics=[]
    for i in range(10):
        print(f"slice {i}:")
        train_data, test_data, train_label, test_label = train_test_split(
            train,
            test,
            test_size=test_size_def,
            random_state=i,
        )
        model=train_model(train_data,train_label)
        predictions=model.predict(test_data)
        precision,recall,fbeta = compute_model_metrics(test_data,test_label)
        print_metrics(precision, recall, fbeta, model.score(test_data,test_label))
        metrics.append(
            [
                int(i),
                test_size_def,
                precision,
                recall,
                fbeta,
                model.score(test_data, test_label),
            ]
        )
    metrics_df=pd.DataFrame(
        metrics,
        columns=["random_state","test_size","precision","recall","f1","accuracy"],
    )
    metrics_mean=metrics_df.mean()
    return metrics_df, metrics_mean
    
def print_metrics(precision, recall, fbeta, accuracy):
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1: {round(fbeta, 2)}")
    print(f"Accuracy: {round(accuracy, 2)}")
    
def evaluate_with_feature_fixed(model, train_data, fixed_metric, cat_features, encoder, label_binarizer):
    unique_values = train_data[fixed_metric].unique()
    
    with open(f"Sliced_output_{fixed_metric}.txt", "w", encoding="utf-8") as file:
        file.write(f"Performance metrics for {fixed_metric}")
        file.write("\n")
        file.write("-" * 10)
        file.write("\n")
        file.write("-" * 10)
        file.write("\n")
        for fixed_slice in unique_values:
            file.write(fixed_slice)
            file.write("\n")
            metric_fixed_df = train_data.loc[train_data.loc[:, fixed_metric] == fixed_slice, :]
            data_processed, labels_processed, encoder, label_binarizer = process_data(
                metric_fixed_df,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                label_bin=label_binarizer
            )
            predictions=inference(model, data_processed)
            precision,recall,fbeta = compute_model_metrics(labels_processed, predictions)
            file.write(f"Precision: {precision}\n")
            file.write(f"Recall: {recall}\n")
            file.write(f"fbeta: {fbeta}\n")
            file.write(f"Accuracy: {model.score(data_processed, labels_processed)}\n")
            file.write("-" * 10)
            file.write("\n")
        file.close()
