"""
streamlit_app.py - Streamlit Frontend for LSTM Temperature Forecasting
=======================================================================
Web interface for the Jena Climate temperature forecasting API.

Run with:
    streamlit run streamlit_app.py

Configuration:
    - API URL is configurable in the sidebar (default: http://localhost:8000)
    - Requires the FastAPI backend to be running

Tabs:
    1. Forecast   — Overview and RMSE metrics
    2. Predictions — Actual vs predicted temperature chart
    3. Training   — Training loss curves
    4. Model Info — Architecture, hyperparameters, diagram
    5. About      — Project info, dataset description, architecture
"""

import streamlit as st
import requests
import json
import base64
import plotly.graph_objects as go

# --- PAGE CONFIGURATION -------------------------------------------------------
st.set_page_config(
    page_title="Temperature Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    h1, h2, h3 { color: #7c6af5 !important; }
    .info-card {
        background: rgba(124, 106, 245, 0.1);
        border: 1px solid rgba(124, 106, 245, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #1e1e3a, #252545);
        border: 1px solid #7c6af5;
        border-radius: 12px;
        padding: 20px 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(124, 106, 245, 0.2);
    }
    .success-box {
        background: rgba(106, 245, 194, 0.1);
        border: 1px solid rgba(106, 245, 194, 0.4);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background: rgba(245, 106, 106, 0.1);
        border: 1px solid rgba(245, 106, 106, 0.4);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    [data-testid="metric-container"] {
        background: rgba(124, 106, 245, 0.1);
        border: 1px solid rgba(124, 106, 245, 0.3);
        border-radius: 8px;
        padding: 10px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a, #1a1a2e);
        border-right: 1px solid rgba(124, 106, 245, 0.2);
    }
    .stTabs [data-baseweb="tab"] { color: #aaaacc; }
    .stTabs [aria-selected="true"] {
        color: #7c6af5 !important;
        border-bottom-color: #7c6af5 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c6af5, #5a4dd4);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #9080ff, #7c6af5);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 106, 245, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# --- CONSTANTS ----------------------------------------------------------------
DEFAULT_API_URL = "https://ait-204-pair-programing-1.onrender.com"

# --- PLOTLY CHART LAYOUT ------------------------------------------------------
CHART_LAYOUT = dict(
    plot_bgcolor='rgba(30,30,60,1)',
    paper_bgcolor='rgba(20,20,40,1)',
    font=dict(color='white'),
    legend=dict(bgcolor='rgba(40,40,70,1)'),
)


# --- HELPER FUNCTIONS ---------------------------------------------------------

def check_api_health(api_url: str) -> dict:
    """Check if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "online", "data": response.json()}
        return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.ConnectionError:
        return {"status": "offline", "message": "Cannot connect to API"}
    except requests.Timeout:
        return {"status": "timeout", "message": "API request timed out"}


def get_model_info(api_url: str) -> dict:
    """Fetch model information from the /model/info endpoint."""
    try:
        response = requests.get(f"{api_url}/model/info", timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "message": "Failed to fetch model info"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_architecture_image(api_url: str) -> str | None:
    """Fetch and decode the model architecture image."""
    try:
        response = requests.get(f"{api_url}/model/architecture-image", timeout=30)
        if response.status_code == 200:
            return response.json().get("image_base64")
    except Exception:
        pass
    return None


def get_training_history(api_url: str) -> dict | None:
    """Fetch training history from the API."""
    try:
        response = requests.get(f"{api_url}/training/history", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_prediction_sample(api_url: str, num_points: int = 500) -> dict | None:
    """Fetch actual vs predicted sample from the API."""
    try:
        response = requests.get(
            f"{api_url}/predictions/sample",
            params={"num_points": num_points},
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# --- SIDEBAR ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    api_url = st.text_input(
        "API URL",
        value=DEFAULT_API_URL,
        help="URL of the FastAPI backend"
    )

    st.markdown("---")

    # API Status
    st.markdown("### API Status")
    health = check_api_health(api_url)

    if health["status"] == "online":
        st.success("API Online")
    elif health["status"] == "offline":
        st.error("API Offline")
        st.caption("Start backend: `uvicorn app.main:app --reload`")
    else:
        st.warning(f"{health['status']}")

    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown(f"[API Docs]({api_url}/docs)")
    st.markdown(f"[ReDoc]({api_url}/redoc)")

    st.markdown("---")
    st.markdown("### Dataset Info")
    st.markdown("""
    **Jena Climate Dataset**
    - Source: Max Planck Institute
    - Period: 2009-2016
    - Features: 14 weather variables
    - Interval: 10 min (subsampled hourly)
    - Focus: Temperature (T degC)
    """)


# --- MAIN CONTENT -------------------------------------------------------------

st.markdown("""
<h1 style='text-align: center; background: linear-gradient(135deg, #7c6af5, #6af5c8);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em;'>
LSTM Temperature Forecasting
</h1>
<p style='text-align: center; color: #aaaacc; font-size: 1.1em;'>
Jena Climate Dataset &mdash; Predicting temperature with recurrent neural networks
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -- TABS ----------------------------------------------------------------------
tab_forecast, tab_predictions, tab_training, tab_model, tab_about = st.tabs([
    "Forecast", "Predictions", "Training", "Model Info", "About"
])


# -- TAB 1: FORECAST ----------------------------------------------------------
with tab_forecast:
    st.markdown("### Temperature Forecasting Overview")

    st.markdown("""
    This application uses a **Long Short-Term Memory (LSTM)** neural network to forecast
    temperature based on the Jena Climate dataset. The model learns temporal patterns from
    historical weather data and predicts future temperature values.
    """)

    # Fetch model info for metrics
    info_result = get_model_info(api_url)
    predictions = get_prediction_sample(api_url, num_points=100)

    if info_result["success"] and info_result["data"].get("status") == "loaded":
        info = info_result["data"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rmse = info.get('rmse_celsius', 0)
            st.metric("Test RMSE", f"{rmse:.2f} C")
        with col2:
            st.metric("Sequence Length", f"{info.get('sequence_length', 'N/A')} hrs")
        with col3:
            st.metric("Total Parameters", f"{info.get('total_parameters', 0):,}")
        with col4:
            st.metric("LSTM Layers", info.get('num_layers', 'N/A'))

        st.markdown("---")

        if predictions:
            st.markdown("### Recent Prediction Sample")
            st.markdown(f"Showing {len(predictions['actual'])} test set predictions "
                       f"(RMSE: **{predictions['rmse_celsius']:.2f} C**)")

            # Quick preview chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=predictions['actual'],
                name='Actual Temperature',
                line=dict(color='#6af5c8', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=predictions['predicted'],
                name='Predicted Temperature',
                line=dict(color='#7c6af5', width=2)
            ))
            fig.update_layout(
                title="Actual vs Predicted Temperature (Test Set Sample)",
                yaxis_title="Temperature (C)",
                **CHART_LAYOUT
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction data available. Train the model first.")
    else:
        st.markdown("""
        <div class="error-box">
        <h4>Model Not Loaded</h4>
        <p>The model has not been trained yet. Follow the steps below:</p>
        </div>
        """, unsafe_allow_html=True)
        st.code("""
# Step 1: Navigate to the backend directory
cd rnn_app/backend

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train the model
python app/train.py

# Step 4: Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Step 5: Start the frontend (separate terminal)
cd rnn_app/frontend
streamlit run streamlit_app.py
        """, language="bash")


# -- TAB 2: PREDICTIONS -------------------------------------------------------
with tab_predictions:
    st.markdown("### Actual vs Predicted Temperatures")
    st.markdown("""
    This chart compares the model's predictions against actual temperature values
    from the test set (the most recent portion of the dataset that the model
    never saw during training).
    """)

    num_points = st.slider("Data Points to Display", 100, 2000, 500, step=100)

    predictions_full = get_prediction_sample(api_url, num_points=num_points)

    if predictions_full:
        # Main prediction chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=predictions_full['actual'],
            name='Actual Temperature',
            line=dict(color='#6af5c8', width=1.5),
            opacity=0.8
        ))
        fig.add_trace(go.Scatter(
            y=predictions_full['predicted'],
            name='Predicted Temperature',
            line=dict(color='#7c6af5', width=1.5),
            opacity=0.8
        ))

        rmse = predictions_full['rmse_celsius']
        fig.update_layout(
            title=f"Temperature Forecast Results (RMSE: {rmse:.2f} C)",
            yaxis_title="Temperature (C)",
            height=500,
            **CHART_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

        # Error distribution
        import numpy as np
        actual = np.array(predictions_full['actual'])
        predicted = np.array(predictions_full['predicted'])
        errors = actual - predicted

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{rmse:.2f} C")
        with col2:
            st.metric("Mean Error", f"{np.mean(errors):.2f} C")
        with col3:
            st.metric("Max Error", f"{np.max(np.abs(errors)):.2f} C")
        with col4:
            st.metric("Std Error", f"{np.std(errors):.2f} C")

        # Error histogram
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            marker_color='#7c6af5',
            opacity=0.7,
            name='Prediction Error'
        ))
        fig_err.update_layout(
            title="Prediction Error Distribution",
            xaxis_title="Error (C)",
            yaxis_title="Count",
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_err, use_container_width=True)

    else:
        st.info("No prediction data available. Train the model first.")


# -- TAB 3: TRAINING ----------------------------------------------------------
with tab_training:
    st.markdown("### Training History")

    history = get_training_history(api_url)
    if history:
        epochs = list(range(1, len(history.get('loss', [])) + 1))

        # Loss chart
        fig_loss = go.Figure()
        if 'loss' in history:
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=history['loss'],
                name='Training Loss (MSE)',
                line=dict(color='#7c6af5', width=2)
            ))
        if 'val_loss' in history:
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=history['val_loss'],
                name='Validation Loss (MSE)',
                line=dict(color='#f56a6a', width=2, dash='dot')
            ))
        fig_loss.update_layout(
            title="Training & Validation Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Mean Squared Error (MSE)",
            height=500,
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # Training summary
        st.markdown("#### Training Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Epochs", len(epochs))
        m2.metric("Final Train Loss", f"{history.get('loss', [0])[-1]:.6f}")
        m3.metric("Best Val Loss", f"{min(history.get('val_loss', [0])):.6f}")
        m4.metric("Final Val Loss", f"{history.get('val_loss', [0])[-1]:.6f}")

        # Log scale option
        if st.checkbox("Show log scale"):
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(
                x=epochs, y=history['loss'],
                name='Training Loss', line=dict(color='#7c6af5', width=2)
            ))
            if 'val_loss' in history:
                fig_log.add_trace(go.Scatter(
                    x=epochs, y=history['val_loss'],
                    name='Validation Loss', line=dict(color='#f56a6a', width=2, dash='dot')
                ))
            fig_log.update_layout(
                title="Loss (Log Scale)",
                xaxis_title="Epoch",
                yaxis_title="MSE (log)",
                yaxis_type="log",
                **CHART_LAYOUT
            )
            st.plotly_chart(fig_log, use_container_width=True)

    else:
        st.info("Training history not available. Train the model first.")


# -- TAB 4: MODEL INFO --------------------------------------------------------
with tab_model:
    st.markdown("### Model Information")

    info_result = get_model_info(api_url)

    if info_result["success"]:
        info = info_result["data"]

        if info.get("status") == "not_loaded":
            st.warning("Model not loaded. Train the model first.")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sequence Length", f"{info.get('sequence_length', 'N/A')} hrs")
            with col2:
                st.metric("Hidden Units", info.get('hidden_units', 'N/A'))
            with col3:
                st.metric("Total Parameters", f"{info.get('total_parameters', 0):,}")
            with col4:
                rmse = info.get('rmse_celsius', 0)
                st.metric("Test RMSE", f"{rmse:.2f} C" if rmse else "N/A")

            st.markdown("---")

            col_arch, col_config = st.columns(2)

            with col_arch:
                st.markdown("### Layer Architecture")
                layers = info.get("layers", [])
                for i, layer in enumerate(layers):
                    arrow = ">" if i < len(layers) - 1 else ""
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>{layer['name']}</strong> — <em>{layer['type']}</em><br>
                        <small style="color: #7c6af5;">Parameters: {layer.get('parameters', 0):,}</small>
                    </div>
                    {"<div style='text-align:center; color:#7c6af5;'>&#x2193;</div>" if arrow else ""}
                    """, unsafe_allow_html=True)

            with col_config:
                st.markdown("### Hyperparameters")
                config = info.get("config", {})
                for key, value in config.items():
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>{key.replace('_', ' ').title()}</strong>: {value}
                    </div>
                    """, unsafe_allow_html=True)

            # Architecture image
            st.markdown("---")
            st.markdown("### Model Architecture Diagram")
            with st.spinner("Loading architecture diagram..."):
                arch_img = get_architecture_image(api_url)
            if arch_img:
                img_bytes = base64.b64decode(arch_img)
                st.image(img_bytes, use_column_width=False, width=500)
            else:
                st.info("Architecture diagram not available")
    else:
        st.error(f"Failed to fetch model info: {info_result.get('message')}")


# -- TAB 5: ABOUT -------------------------------------------------------------
with tab_about:
    st.markdown("### About This Application")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### What This App Does
        This application demonstrates **LSTM-based temperature forecasting**
        using the Jena Climate dataset:

        1. **Data Processing** — Load CSV, extract temperature, normalize
        2. **Sequence Creation** — Sliding window (5 days) to predict next hour
        3. **LSTM Model** — Two-layer LSTM learns temporal patterns
        4. **FastAPI Backend** — Serves predictions via REST API
        5. **Streamlit Frontend** — Interactive visualization dashboard

        #### The Jena Climate Dataset
        - **Source:** Max Planck Institute for Biogeochemistry
        - **Period:** January 2009 to December 2016
        - **Features:** 14 weather measurements (temperature, pressure,
          humidity, wind, etc.)
        - **Interval:** Every 10 minutes (~420,000 observations)
        - **Our focus:** Temperature column (`T (degC)`)
        - **Subsampling:** Every 6th row (hourly) for tractable training

        #### System Architecture
        ```
        You (browser)
              | HTTP
        Streamlit App (:8501)
              | HTTP requests
        FastAPI Server (:8000)
              | PyTorch inference
        LSTM Model (models/)
              |
        Jena Climate CSV (data/)
        ```
        """)

    with col2:
        st.markdown("""
        #### Technology Stack
        | Component | Technology |
        |-----------|------------|
        | Deep Learning | PyTorch LSTM |
        | Data Processing | pandas, scikit-learn |
        | Backend API | FastAPI + Uvicorn |
        | Frontend | Streamlit |
        | Visualization | Plotly |
        | Scaling | MinMaxScaler |

        #### Key Concepts
        - **LSTM** — Long Short-Term Memory networks handle sequential
          data by maintaining cell state across timesteps
        - **Sliding Window** — Use N past observations to predict the next
        - **Min-Max Scaling** — Normalize to [0,1] for efficient training
        - **RMSE** — Root Mean Squared Error measures prediction accuracy
          in the original units (degrees Celsius)
        - **Chronological Split** — Test set uses the latest data only
          (no data leakage from the future)

        #### Project Structure
        ```
        rnn_app/
        +-- backend/
        |   +-- app/
        |   |   +-- temperature_forecaster.py
        |   |   +-- train.py
        |   |   +-- main.py
        |   +-- data/
        |   |   +-- jena_climate_2009_2016.csv
        |   +-- models/
        |   +-- requirements.txt
        +-- frontend/
        |   +-- streamlit_app.py
        |   +-- requirements.txt
        +-- docs/
        |   +-- lstm_rnn_guide.html
        +-- README.md
        ```
        """)

    st.markdown("---")
    st.markdown("#### Quick Start")
    st.code("""
# Terminal 1: Backend
cd rnn_app/backend
pip install -r requirements.txt
python app/train.py                # Train the model
uvicorn app.main:app --reload      # Start API server

# Terminal 2: Frontend
cd rnn_app/frontend
pip install -r requirements.txt
streamlit run streamlit_app.py     # Start web UI
    """, language="bash")
