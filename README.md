
# Retail Demand Forecasting Dashboard

An end-to-end machine learning web application that predicts future product sales across multiple stores and items. Built with an **XGBoost** regression model, a **FastAPI** REST backend, and an interactive **Streamlit** + **Plotly** dashboard.

## Dashboard Preview
<img width="1343" height="636" alt="image" src="https://github.com/user-attachments/assets/b85f28fc-b3b8-4b80-8684-ff857ba463e2" />
<img width="1298" height="429" alt="image" src="https://github.com/user-attachments/assets/fcdf8158-ca19-4a66-b5a5-070a9a50821f" />


## Live Demo



https://retail-demand-forecast-emaan.streamlit.app/

## Features
* **Multi-Store & Item Forecasting:** Predicts daily product demand from 7 up to 90 days ahead.
* **RESTful ML API:** FastAPI backend serving real-time JSON predictions from a pre-trained XGBoost model (`model.joblib`).
* **Interactive Analytics:** Visualizes historical sales trajectories alongside predicted demand and weekly day-of-week trends using Plotly.
* **Exportable Reports:** Download generated forecast schedules directly as CSV files.

## Tech Stack
* **Machine Learning & Data:** Python, Pandas, Scikit-learn, XGBoost, Joblib
* **Backend API:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit, Plotly, Requests

## Project Structure
```text
demand-forecasting-dashboard/
├── backend/
│   ├── train.py          # Trains and serializes the XGBoost forecasting model
│   ├── api.py            # FastAPI server exposing the /predict endpoint
│   └── model.joblib      # Saved trained model
├── data/
│   └── train.csv         # Historical store-item sales dataset
├── frontend/
│   └── app.py            # Streamlit interactive dashboard
└── requirements.txt      # Project dependencies
```

## How to Run Locally

### 1. Clone the repository & set up environment
```bash
git clone https://github.com
cd demand-forecasting-dashboard
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend (Terminal 1)
```bash
uvicorn backend.api:app --reload
```
The API will run at `http://127.0.0.1:8000` (interactive API docs available at `http://127.0.0`).

### 3. Start the Streamlit Frontend (Terminal 2)
```bash
venv\Scripts\activate
streamlit run frontend/app.py
```
The dashboard will open automatically at `http://localhost:8501`.
