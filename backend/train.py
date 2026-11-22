from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBRegressor

# Locate files safely relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "backend" / "model.joblib"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Convert date column and extract time features
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

# Define inputs (X) and target (y)
features = ["store", "item", "year", "month", "day", "day_of_week"]
X = df[features]
y = df["sales"]

print("Training XGBoost model (this takes ~10-20 seconds)...")
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"Done! Model saved cleanly to: {MODEL_PATH}")
