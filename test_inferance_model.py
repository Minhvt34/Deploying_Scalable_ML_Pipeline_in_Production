from starter.inferance_model import run_inference


def test_run_inference_low(inference_data_low, cat_features):
    prediction = run_inference(inference_data_low, cat_features)

    assert prediction == "<=50K"


def test_run_inference_high(inference_data_high, cat_features):
    prediction = run_inference(inference_data_high, cat_features)

    assert prediction == ">50K"
