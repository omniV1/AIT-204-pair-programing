# Temperature Forecasting — Backend

A FastAPI backend that trains and serves an LSTM temperature forecasting model using the Jena Climate dataset. Built for AIT-204.

---

## Quick Start

### 1. Set up the environment

```bash
cd rnn_app/backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Train with default settings
python app/train.py

# Custom hyperparameters
python app/train.py --epochs 30 --hidden-units 128 --learning-rate 0.0005

# Quick test run
python app/train.py --epochs 10 --batch-size 512
```

Training artifacts saved to `models/`:
- `model.pt` — PyTorch model weights
- `scaler.pkl` — Fitted MinMaxScaler (for denormalization)
- `config.json` — Hyperparameters used during training
- `training_history.json` — Loss per epoch
- `test_results.json` — Actual vs predicted values on the test set

### 3. Start the API server

```bash
# Development (auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4. Test the API

```bash
# Check health
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Get predictions sample
curl http://localhost:8000/predictions/sample?num_points=100

# Interactive docs
open http://localhost:8000/docs
```

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py                 # Package marker
│   ├── temperature_forecaster.py   # LSTM model + TemperatureForecaster class
│   ├── train.py                    # Training orchestrator script
│   └── main.py                     # FastAPI application and endpoints
├── data/
│   └── jena_climate_2009_2016.csv  # Jena Climate dataset
├── models/                         # Auto-created after training
│   ├── model.pt                    # PyTorch model weights
│   ├── best_model.pt               # Best checkpoint (lowest val_loss)
│   ├── scaler.pkl                  # Fitted MinMaxScaler
│   ├── config.json                 # Training configuration
│   ├── training_history.json       # Loss curves
│   └── test_results.json           # Actual vs predicted on test set
├── visualizations/                 # Auto-created after training
│   ├── training_history.png        # Loss plot
│   └── predictions.png             # Actual vs predicted plot
├── requirements.txt
└── README.md
```

---

## Model Architecture

```
Input (batch, 120, 1) — 120 hours of normalized temperature
        |
  [LSTM Layer 1]       — hidden_size=64, processes all timesteps
        |
  [Dropout 20%]        — regularization
        |
  [LSTM Layer 2]       — hidden_size=64, returns final hidden state h_T
        |
  [Dropout 20%]        — regularization
        |
  [Linear]             — 64 -> 1, predicted normalized temperature
        |
Output (batch, 1) — next temperature (denormalized to °C)
```

**Loss:** Mean Squared Error (MSE) — standard for regression tasks

**Evaluation:** RMSE (Root Mean Squared Error) in °C

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check with model status |
| GET | `/health` | Lightweight health check |
| POST | `/predict` | Predict next temperature from input sequence |
| GET | `/model/info` | Architecture, hyperparameters, test RMSE |
| GET | `/model/architecture-image` | Base64 PNG architecture diagram |
| GET | `/training/history` | Training loss history (JSON) |
| GET | `/predictions/sample` | Actual vs predicted values (JSON) |
| GET | `/predictions/plot` | Base64 PNG prediction chart |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc documentation |

### POST /predict

Request:
```json
{
  "temperatures": [5.2, 5.1, 4.9, 5.0, 5.3, ...]
}
```

Response:
```json
{
  "predicted_temperature": 5.15,
  "input_length": 120,
  "sequence_length": 120
}
```

### GET /predictions/sample

Returns actual vs predicted temperatures on the test set:
```json
{
  "actual": [5.2, 5.1, ...],
  "predicted": [5.15, 5.08, ...],
  "rmse_celsius": 1.85,
  "num_samples": 14000,
  "num_returned": 500
}
```

---

## Hyperparameter Tuning Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `sequence_length` | 120 | 24–240 | Hours of context. 120 = 5 days. Longer = more context, slower training. |
| `hidden_units` | 64 | 32–256 | LSTM capacity. Larger = more expressive, higher overfitting risk. |
| `num_layers` | 2 | 1–4 | Stacked LSTM depth. Deeper captures hierarchical patterns. |
| `dropout_rate` | 0.2 | 0.0–0.5 | Regularization strength. Higher prevents overfitting. |
| `batch_size` | 256 | 64–512 | Samples per gradient update. |
| `epochs` | 50 | 10–200 | Maximum training passes. Early stopping halts if no improvement. |
| `learning_rate` | 0.001 | 1e-4–1e-2 | Adam optimizer step size. |
| `subsample_step` | 6 | 1–12 | Subsampling factor. 6 = hourly from 10-min data. |

### Example: experimenting with larger model

```bash
python app/train.py \
  --hidden-units 128 \
  --num-layers 3 \
  --epochs 100 \
  --learning-rate 0.0005
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models` | Directory containing trained model artifacts |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | LSTM model (nn.LSTM, nn.Linear) |
| `pandas` | CSV data loading and manipulation |
| `scikit-learn` | MinMaxScaler for normalization |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response validation |
| `numpy` | Numerical arrays |
| `matplotlib` | Training and prediction plots |
