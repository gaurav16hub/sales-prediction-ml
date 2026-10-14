import pandas as pd

# Load the dataset
df = pd.read_csv("Advertising.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("Advertising.csv")

# Remove unnecessary index column if present
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# 2. Define features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 5. Train Random Forest
forest_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
forest_model.fit(X_train, y_train)

# 6. Compare models
models = {
    "Linear Regression": linear_model,
    "Random Forest": forest_model
}

best_model = None
best_r2 = float("-inf")

print("\n===== MODEL EVALUATION =====")

for name, model in models.items():
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring="r2"
    )

    print(f"\nModel: {name}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"5-Fold CV R2: {cv_scores.mean():.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# 7. Save best model
joblib.dump(best_model, "sales_model.pkl")

print("\nBest model saved as sales_model.pkl")
print("Model training completed successfully!")