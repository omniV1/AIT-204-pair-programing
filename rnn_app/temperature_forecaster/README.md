# LSTM Temperature Forecasting — Jena Climate Dataset

A full-stack application that uses a **Long Short-Term Memory (LSTM)** recurrent neural network to forecast temperature from the Jena Climate dataset. Built with PyTorch, FastAPI, and Streamlit for AIT-204.

---

## Overview

This project adapts an RNN architecture for **time-series forecasting** — predicting future temperature values based on 5 days of historical hourly observations. The Jena Climate dataset, collected by the Max Planck Institute for Biogeochemistry, provides real-world weather measurements recorded every 10 minutes from 2009 to 2016.

### Architecture

```
┌──────────────────────┐
│   Streamlit Frontend  │  (streamlit_app.py)
│   - Forecast dashboard│
│   - Prediction charts │
│   - Training history  │
└──────────┬───────────┘
           │ HTTP
┌──────────▼───────────┐
│   FastAPI Backend     │  (main.py)
│   - /predict          │
│   - /model/info       │
│   - /predictions/sample│
│   - /training/history │
└──────────┬───────────┘
           │ PyTorch
┌──────────▼───────────┐
│   LSTM Model          │  (temperature_forecaster.py)
│   - 2-layer LSTM      │
│   - 64 hidden units   │
│   - MSE loss          │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│   Jena Climate CSV    │  (data/)
│   420k observations   │
│   14 weather features │
└──────────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Backend
cd rnn_app/temperature_forecaster/backend
pip install -r requirements.txt

# Frontend (separate terminal)
cd rnn_app/temperature_forecaster/frontend
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd rnn_app/temperature_forecaster/backend
python app/train.py
```

Training will:
- Load and subsample the Jena Climate CSV (hourly from 10-min data)
- Normalize temperature with Min-Max scaling
- Create sliding-window sequences (120 hours = 5 days)
- Train the LSTM with early stopping
- Evaluate on the test set and report RMSE
- Save model artifacts to `models/`
- Generate visualization plots in `visualizations/`

### 3. Run the Application

**Terminal 1 — API server:**
```bash
cd rnn_app/temperature_forecaster/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit frontend:**
```bash
cd rnn_app/temperature_forecaster/frontend
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`.

API docs available at `http://localhost:8000/docs`.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | Max Planck Institute for Biogeochemistry |
| **Location** | Jena, Germany |
| **Period** | January 2009 — December 2016 |
| **Raw interval** | Every 10 minutes |
| **Total observations** | ~420,000 |
| **Features** | 14 (temperature, pressure, humidity, wind, etc.) |
| **Target variable** | `T (degC)` — air temperature in Celsius |
| **Subsampled interval** | Hourly (every 6th row) |

---

## Model Details

| Hyperparameter | Value |
|----------------|-------|
| Architecture | 2-layer stacked LSTM |
| Hidden units | 64 per layer |
| Input window | 120 hours (5 days) |
| Dropout | 0.2 |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 256 |
| Evaluation metric | RMSE (°C) |

---

## Project Structure

```
rnn_app/
├── backend/
│   ├── app/
│   │   ├── __init__.py                 # Package marker
│   │   ├── temperature_forecaster.py   # LSTM model + forecaster class
│   │   ├── train.py                    # Training orchestrator
│   │   └── main.py                     # FastAPI server
│   ├── data/
│   │   └── jena_climate_2009_2016.csv  # Dataset
│   ├── models/                         # Saved model artifacts (after training)
│   ├── visualizations/                 # Generated plots (after training)
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py                # Streamlit web UI
│   └── requirements.txt
├── docs/
│   └── lstm_rnn_guide.html             # Educational LSTM/RNN guide
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Lightweight health check |
| POST | `/predict` | Predict next temperature from a sequence |
| GET | `/model/info` | Model architecture, hyperparameters, RMSE |
| GET | `/model/architecture-image` | Base64 PNG architecture diagram |
| GET | `/training/history` | Training loss history (JSON) |
| GET | `/predictions/sample` | Actual vs predicted test set values |
| GET | `/predictions/plot` | Base64 PNG of prediction chart |
| GET | `/docs` | Swagger UI |

---

## Documentation

See `docs/lstm_rnn_guide.html` for a detailed educational guide covering:
- RNN fundamentals and the vanishing gradient problem
- LSTM architecture (forget gate, input gate, output gate, cell state)
- Time-series forecasting with sliding windows
- Min-Max normalization and RMSE evaluation
- Application to the Jena Climate dataset

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch (LSTM) |
| Data Processing | pandas, scikit-learn (MinMaxScaler) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Deployment | Render (cloud) |

---

## Troubleshooting

### "Model not loaded" on server startup
Run the training script first: `python app/train.py`

### Training is very slow
- Reduce epochs: `python app/train.py --epochs 20`
- Reduce hidden units: `--hidden-units 32`
- PyTorch uses CUDA automatically when a GPU is detected

### Port already in use
```bash
uvicorn app.main:app --reload --port 8001
```
Then update the API URL in the Streamlit sidebar.
