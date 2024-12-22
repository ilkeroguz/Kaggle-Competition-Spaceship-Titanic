from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import preprocess_data, evaluate_model, save_model
import pandas as pd

# Load dataset
train_set = pd.read_csv("data/train.csv")

# Preprocess data
train_set, encoder = preprocess_data(train_set)

# Separate target variable
X = train_set.drop(columns=["Transported", "Name"])
y = train_set["Transported"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=43, stratify=y, test_size=0.5
)

# Train model
model = LogisticRegression(max_iter=300, penalty="l1", class_weight="balanced", solver="liblinear", C=0.9)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = evaluate_model(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save model
save_model(model, "model.pkl")