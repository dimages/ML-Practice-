import catboost
import joblib
import pandas as pd




rf_model = joblib.load("ml_models/random_forest_model.joblib")
svc_model = joblib.load("ml_models/svc_pipeline_model.joblib")
# catboost_model = joblib.load("ml_models/catboost_model.joblib")
catboost_model = catboost.CatBoostClassifier().load_model("ml_models/catboost.model")


def rf_model_predict(data: pd.DataFrame) -> None:
    predictions = rf_model.predict(data)
    print(predictions)
    return predictions


def svc_model_predict(data: pd.DataFrame) -> None:
    predictions = svc_model.predict(data)
    print(predictions)
    return predictions


def catboost_model_predict(data: pd.DataFrame) -> None:
    predictions = catboost_model.predict(data)
    print(predictions)
    return predictions
