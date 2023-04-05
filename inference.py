import sklearn
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load model from MLflow
run_id = "79632da3f94c4de39c6261b49b7bddda"  # Replace with the ID of the MLflow run containing your model
model = mlflow.sklearn.load_model(f"runs:/{run_id}/linear-regression-model")

# Load data
data = pd.read_csv("Salary_Data.csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Make predictions with the loaded model
predictions = model.predict(X)

# Evaluate model performance
mse = mean_squared_error(y, predictions)
print(f"Mean squared error: {mse:.2f}")