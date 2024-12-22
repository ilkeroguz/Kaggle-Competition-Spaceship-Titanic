from utils import preprocess_data, load_model
import pandas as pd

# Load test dataset
test_set = pd.read_csv("data/test.csv")

# Preprocess data
X_test, _ = preprocess_data(test_set, is_training=False)
X_test = X_test.drop(columns=["Name"], errors="ignore")

# Load model
model = load_model("model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Output predictions
output = pd.DataFrame({"Id": test_set["PassengerId"], "Transported": y_pred})
output.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")
