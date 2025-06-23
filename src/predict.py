import joblib
import pandas as pd

def load_model(path: str):
    return joblib.load(path)

def make_prediction(model, X: pd.DataFrame):
    return model.predict(X)
