import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.transformers import feature_engineering
from sklearn.metrics import mean_squared_error


def train_pipeline(data_path):

    # 1. LOAD DATA
    df = load_data(data_path)

    # 2. CLEAN DATA
    df = clean_data(df)

    # 3. FEATURE ENGINEERING
    df = feature_engineering(df)

    # 4. ENCODE
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    # 5. SPLIT
    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. MODEL
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # ✅ 7. SAVE MODEL (FIXED)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "final_model.pkl")
    joblib.dump(model, model_path)
    
    rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5


    return model ,rmse