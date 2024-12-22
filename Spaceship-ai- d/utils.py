# utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def preprocess_data(train_set, is_training=True):
    # Fill missing values
    train_set[["Cabin", "HomePlanet", "CryoSleep", "Destination", "VIP"]] = train_set[["Cabin", "HomePlanet", "CryoSleep", "Destination", "VIP"]].fillna(method="pad")
    train_set[["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]] = train_set[["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(value=train_set[["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].mean())
    train_set["Name"] = train_set["Name"].fillna(value="None")

    # Convert boolean columns to integers
    if "CryoSleep" in train_set.columns:
        train_set["CryoSleep"] = train_set["CryoSleep"].astype(int)
    if "VIP" in train_set.columns:
        train_set["VIP"] = train_set["VIP"].astype(int)

    # Encode categorical columns
    encoder = LabelEncoder()
    for column in train_set.select_dtypes(include=["object"]):
        train_set[column] = encoder.fit_transform(train_set[column])

    if is_training and "Transported" in train_set.columns:
        train_set["Transported"] = train_set["Transported"].astype(int)

    return train_set, encoder

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def evaluate_model(y_true, y_pred):
    return accuracy_score(y_true, y_pred)