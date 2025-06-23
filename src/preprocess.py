import pandas as pd

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    # Example preprocessing: fill missing values and drop non-numeric columns
    df = df.fillna(0)
    return df.select_dtypes(include=['float64', 'int64'])
