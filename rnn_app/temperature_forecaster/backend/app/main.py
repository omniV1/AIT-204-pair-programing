"""
main.py - FastAPI Application Server for Temperature Forecasting
=================================================================
REST API that serves the trained LSTM temperature forecasting model.
Loads the model on startup and exposes endpoints for:
  - Temperature prediction
  - Model information
  - Training history visualization
  - Test set predictions (actual vs predicted)
  - Health check

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API Documentation (auto-generated):
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

import os
import io
import json
import base64
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.temperature_forecaster import TemperatureForecaster


# ─── PYDANTIC MODELS (Request/Response Schemas) ──────────────────────────────

class PredictRequest(BaseModel):
    """Schema for temperature prediction requests."""
    temperatures: List[float] = Field(
        ...,
        min_length=1,
        description="List of past temperature values (°C) for prediction. "
                    "Should match the model's sequence_length."
    )

class PredictResponse(BaseModel):
    """Schema for temperature prediction responses."""
    predicted_temperature: float
    input_length: int
    sequence_length: int

class ModelInfoResponse(BaseModel):
    """Schema for model information responses."""
    status: str
    sequence_length: Optional[int] = None
    hidden_units: Optional[int] = None
    num_layers: Optional[int] = None
    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None
    rmse_celsius: Optional[float] = None
    layers: Optional[list] = None
    config: Optional[dict] = None


# ─── FASTAPI APPLICATION ──────────────────────────────────────────────────────

app = FastAPI(
    title="Temperature Forecasting API",
    description="""
    ## LSTM Temperature Forecasting API

    This API serves a trained LSTM model for temperature prediction
    using the Jena Climate dataset.

    ### How it works:
    1. Train a model using `python app/train.py`
    2. Start this server: `uvicorn app.main:app --reload`
    3. Query endpoints for predictions, model info, and visualizations

    ### Dataset:
    Jena Climate dataset (2009-2016) from the Max Planck Institute
    for Biogeochemistry. Temperature is subsampled to hourly intervals
    and the model uses 5 days of history to predict the next hour.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model instance ──────────────────────────────────────────────────────
forecaster: Optional[TemperatureForecaster] = None
MODEL_DIR = os.environ.get("MODEL_DIR", "models")


# ─── STARTUP EVENT ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model():
    """Load the trained model when the server starts."""
    global forecaster
    forecaster = TemperatureForecaster()

    model_path = os.path.join(MODEL_DIR, 'model.pt')
    if os.path.exists(model_path):
        try:
            forecaster.load(MODEL_DIR)
            info = forecaster.get_model_info()
            print(f"Model loaded successfully from {MODEL_DIR}/")
            print(f"  Total parameters: {info.get('total_parameters', 'N/A'):,}")
            if info.get('rmse_celsius'):
                print(f"  Test RMSE: {info['rmse_celsius']:.2f}°C")
        except Exception as e:
            print(f"Failed to load model: {e}")
            forecaster = None
    else:
        print(f"No trained model found at {model_path}")
        print("  Run 'python app/train.py' to train a model first")
        forecaster = None


# ─── API ENDPOINTS ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint with model status."""
    model_loaded = forecaster is not None and forecaster.model is not None
    return {
        "status": "running",
        "model_loaded": model_loaded,
        "message": "Temperature Forecasting API is running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Lightweight health check for load balancers."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_temperature(request: PredictRequest):
    """
    Predict the next temperature given a sequence of past values.

    Send a list of past temperature readings (in °C), and the model
    will predict the next temperature. The input is automatically
    normalized, padded/truncated to match the model's sequence length,
    and the prediction is returned in °C.
    """
    if forecaster is None or forecaster.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python app/train.py' first."
        )

    import numpy as np

    seq_len = forecaster.sequence_length
    temps = np.array(request.temperatures)

    # Normalize the input using the fitted scaler
    temps_norm = forecaster.scaler.transform(temps.reshape(-1, 1)).flatten()

    # Pad or truncate to sequence_length
    if len(temps_norm) < seq_len:
        temps_norm = np.pad(temps_norm, (seq_len - len(temps_norm), 0), mode='edge')
    else:
        temps_norm = temps_norm[-seq_len:]

    predicted = forecaster.predict(temps_norm)

    return PredictResponse(
        predicted_temperature=round(predicted, 2),
        input_length=len(request.temperatures),
        sequence_length=seq_len
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model architecture, hyperparameters, and test RMSE."""
    if forecaster is None:
        return ModelInfoResponse(status="not_loaded")
    info = forecaster.get_model_info()
    return ModelInfoResponse(**info)


@app.get("/model/architecture-image", tags=["Model"])
async def get_architecture_image():
    """Return a base64-encoded PNG of the model architecture diagram."""
    if forecaster is None or forecaster.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(8, 10))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        layers_info = [
            ("Input",
             f"shape: (batch, {forecaster.sequence_length}, 1) — normalized temps",
             "#7c6af5"),
            (f"nn.LSTM (layer 1)",
             f"input=1, hidden={forecaster.hidden_units}, all timesteps",
             "#6af5c8"),
            ("nn.Dropout",
             f"p={forecaster.dropout_rate}",
             "#f5d76a"),
            (f"nn.LSTM (layer 2)",
             f"hidden={forecaster.hidden_units}, final h_T only",
             "#6af5c8"),
            ("nn.Dropout",
             f"p={forecaster.dropout_rate}",
             "#f5d76a"),
            ("nn.Linear",
             f"({forecaster.hidden_units} -> 1) — predicted temperature",
             "#f56a6a"),
        ]

        y_positions = [10.5, 9.0, 7.7, 6.2, 4.9, 3.4]
        for i, ((name, detail, color), y) in enumerate(zip(layers_info, y_positions)):
            rect = mpatches.FancyBboxPatch(
                (1.5, y - 0.5), 7, 0.9,
                boxstyle="round,pad=0.1",
                facecolor=color + '33', edgecolor=color, linewidth=2
            )
            ax.add_patch(rect)
            ax.text(5, y, name, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)
            ax.text(5, y - 0.3, detail, ha='center', va='center',
                    fontsize=8, color='#aaaacc')
            if i < len(y_positions) - 1:
                ax.annotate('', xy=(5, y_positions[i+1] + 0.4),
                           xytext=(5, y - 0.5),
                           arrowprops=dict(arrowstyle='->', color='#888899', lw=1.5))

        ax.text(5, 11.5, 'LSTM Temperature Forecasting Architecture',
                ha='center', fontsize=14, fontweight='bold', color='white')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='#1e1e2e')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()

        return {"image_base64": img_base64, "format": "png"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/history", tags=["Training"])
async def get_training_history():
    """Return training history (loss per epoch) as JSON."""
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    history_path = os.path.join(MODEL_DIR, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            return json.load(f)
    elif forecaster.training_history:
        return forecaster.training_history
    else:
        raise HTTPException(
            status_code=404,
            detail="No training history found. Train the model first."
        )


@app.get("/predictions/sample", tags=["Predictions"])
async def get_prediction_sample(num_points: int = 500):
    """
    Return a sample of actual vs predicted temperatures from the test set.
    Used by the frontend to plot prediction accuracy.

    Args:
        num_points: Number of data points to return (default 500, max 5000)
    """
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results_path = os.path.join(MODEL_DIR, 'test_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    elif forecaster.test_results:
        results = forecaster.test_results
    else:
        raise HTTPException(
            status_code=404,
            detail="No test results found. Train and evaluate the model first."
        )

    num_points = min(num_points, 5000, len(results.get('actual', [])))
    return {
        'actual': results['actual'][:num_points],
        'predicted': results['predicted'][:num_points],
        'rmse_celsius': results['rmse_celsius'],
        'num_samples': results['num_samples'],
        'num_returned': num_points,
    }


@app.get("/predictions/plot", tags=["Predictions"])
async def get_predictions_plot():
    """Return a base64-encoded PNG of actual vs predicted temperatures."""
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results_path = os.path.join(MODEL_DIR, 'test_results.json')
    if not os.path.exists(results_path) and not forecaster.test_results:
        raise HTTPException(status_code=404, detail="No test results found.")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
        else:
            results = forecaster.test_results

        actual = np.array(results['actual'][:500])
        predicted = np.array(results['predicted'][:500])

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#2e2e3e')

        ax.plot(actual, label='Actual', color='#6af5c8', linewidth=1.5, alpha=0.8)
        ax.plot(predicted, label='Predicted', color='#7c6af5', linewidth=1.5, alpha=0.8)

        rmse = results['rmse_celsius']
        ax.set_title(f'Actual vs Predicted (RMSE: {rmse:.2f}°C)',
                     fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel('Time Step', color='white')
        ax.set_ylabel('Temperature (°C)', color='white')
        ax.legend(facecolor='#2e2e3e', labelcolor='white')
        ax.grid(alpha=0.3, color='#555577')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#555577')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#1e1e2e')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()

        return {"image_base64": img_base64, "format": "png", "rmse": rmse}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
