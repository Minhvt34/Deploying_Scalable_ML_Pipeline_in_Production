# Script to train machine learning model.
import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
#import sys
#sys.path.append(".")
import data as impd
from model import (
    train_model,
    compute_model_metrics,
    train_and_test_on_slices,
    print_metrics,
    evaluate_with_feature_fixed,
)

# Add the necessary imports for the starter code.

# Add code to load in the data.
DATA_PATH = "data/census_cleaned.csv"
data=pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
#train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = impd.process_data(
#    train, categorical_features=cat_features, label="salary", training=True
    data,categorical_features=cat_features, label="salary", training=True
)

#train different splits
metrics_df, metrics_mean=train_and_test_on_slices(X_train,y_train,test_size_def=0.2)
print("mean of all training metrics")
print(
    f"Precision: {metrics_mean['precision']:.2f},"
    + f"Revall: {metrics_mean['recall']:.2f},"
    + f"F1: {metrics_mean['f1']:.2f},"
    + f"Accuracy: {metrics_mean['accuracy']:.2f}"
)


best = metrics_df.loc[metrics_df["f1"].idxmax()]
print("Best row is:")
print(
    f"random_state: {best['random_state']},"
    +f"test_size: {best['test_size']},"
    +f"precision: {best['precision']:.2f},"
    +f"recall: {best['recall']:.2f},"
    +f"f1: {best['f1']:.2f},"
    +f"accuracy: {best['accuracy']:.2f}"
)

random_state=int(best["random_state"])
test_size = best["test_size"]

#create model with best split result
train_data, test_data,train_labels, test_labels = train_test_split(
    X_train,y_train,test_size=test_size,random_state=random_state,
)

model=train_model(train_data,train_labels)

fixed_features = ["education", "occupation"]
for feature in fixed_features:
    evaluate_with_feature_fixed(model,data,feature,cat_features,encoder,lb)

predictions=model.predict(test_data)
precision,recall,fbeta = compute_model_metrics(test_labels,predictions)
print_metrics(precision,recall,fbeta,model.score(test_data,test_labels))
f_metrics=pd.DataFrame(
    [[precision, recall, fbeta, model.score(test_data, test_labels)]],
    columns=["precision", "recall", "f1", "accuracy"],
)


#save model
MODEL_PATH = "model/model.pkl"
ENCODER_PATH = "model/encoder.pkl"
LB_PATH = "model/lb.pkl"

with open(MODEL_PATH, "wb") as f:
    pkl.dump(model, f)
with open(ENCODER_PATH, "wb") as f:
    pkl.dump(encoder, f)
with open(LB_PATH, "wb") as f:
    pkl.dump(lb, f)
 
