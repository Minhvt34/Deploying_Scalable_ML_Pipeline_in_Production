# pylint: disable=too-few-public-methods
"""
This is the main code that works with the API
"""
# Put the code for your API here.
import os
import sys
import pandas as pd

# Import libraries related to fastAPI
from fastapi import FastAPI
from pydantic import BaseModel  # pylint: disable=no-name-in-module

# Import the inference function to be used to predict the values
from ml.data import process_data
from ml.model import inference

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

# create FastPI app
app = FastAPI()


# Give Heroku DVC capabilites
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        sys.exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Model that makes predictions
model = pd.read_pickle(r"model/model.pkl")
encoder = pd.read_pickle(r"model/encoder.pkl")


class DataOut(BaseModel):
    """
    This pydantic class if used for the API output
    """

    # The expected prediction is <=50K
    prediction: str = "Income < 50k"


class DataIn(BaseModel):
    """
    This pydantic class if used for the API input
    """

    # This is the input data to the API
    # Expected prediction = <=50K
    age: int = 28
    workclass: str = "Private"
    fnlgt: int = 338409
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Married-civ-spouse"
    occupation: str = "Prof-specialty"
    relationship: str = "Wife"
    race: str = "Black"
    sex: str = "Female"
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = "Cuba"


# Display welcome message using @app.get("/")
@app.get("/")
def welcome():
    """
    This is the welcoming message
    """
    return {"message": "Welcome to the project"}


@app.post("/predict", response_model=DataOut, status_code=200)
def get_pred(cencus: DataIn):
    """
    This function performs the inference
    """
    # convert cencus to a dictionary
    cencus = dict(cencus)

    # Convert the input data to a dataframe
    cencus_dataframe = pd.DataFrame(cencus, columns=cencus.keys(), index=[0])

    # convert all the "_ " to "-" in the dataframe
    cencus_dataframe.columns = cencus_dataframe.columns.str.replace("_", "-")

    # prepare the data
    processed_epoch, _, _, _ = process_data(
        cencus_dataframe,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        label=None,
    )
    # Calling the inference function to make a prediction
    income_prediction = inference(model, processed_epoch)

    # Return the prediction in the expected format
    if income_prediction == 0:
        income_prediction = "Income < 50k"
    elif income_prediction == 1:
        income_prediction = "Income > 50k"

    # Return the prediction in the form of a JSON
    response_object = {"prediction": income_prediction}
    return response_object
