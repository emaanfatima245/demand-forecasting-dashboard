from datetime import timedelta
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Retail Demand Forecasting API")

# Locate files safely relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "backend" / "model.joblib"

# Load model and historical dataset once on startup
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
MAX_DATE = df["date"].max()

@app.get("/")
def home():
    return {"status": "API is running cleanly"}

@app.get("/predict")
def predict_demand(store_id: int = 1, item_id: int = 1, days_ahead: int = 30):
    if days_ahead < 1 or days_ahead > 365:
        raise HTTPException(status_code=400, detail="days_ahead must be between 1 and 365")

    # Filter historical sales (last 60 days) for context
    history = df[(df["store"] == store_id) & (df["item"] == item_id)].sort_values("date").tail(60)
    if history.empty:
        raise HTTPException(status_code=404, detail="Store ID or Item ID not found in dataset")

    # Generate future dates starting right after the last dataset date
    future_dates = [MAX_DATE + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_df = pd.DataFrame({
        "store": store_id,
        "item": item_id,
        "year": [d.year for d in future_dates],
        "month": [d.month for d in future_dates],
        "day": [d.day for d in future_dates],
        "day_of_week": [d.weekday() for d in future_dates],
    })

    # Run predictions
    preds = model.predict(future_df)

    # Return clean JSON for the frontend
    return {
        "store_id": store_id,
        "item_id": item_id,
        "history": [
            {"date": d.strftime("%Y-%m-%d"), "sales": float(s)}
            for d, s in zip(history["date"], history["sales"])
        ],
        "forecast": [
            {"date": d.strftime("%Y-%m-%d"), "predicted_sales": round(float(p), 2)}
            for d, p in zip(future_dates, preds)
        ],
    }
