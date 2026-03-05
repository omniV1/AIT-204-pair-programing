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
    st.markdown("""
    <h2 style='text-align: center; background: linear-gradient(135deg, #7c6af5, #6af5c8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    LSTM & RNN Guide
    </h2>
    <p style='text-align: center; color: #aaaacc;'>
    Understanding Recurrent Neural Networks for Time-Series Temperature Forecasting<br>
    Applied to the Jena Climate Dataset
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Section 1: RNN Fundamentals ---
    st.markdown("### 1. RNN Fundamentals")
    st.markdown("""
    A **Recurrent Neural Network (RNN)** is a class of neural networks designed to process
    **sequential data** — data where order matters. Unlike feedforward networks that treat each
    input independently, RNNs maintain a **hidden state** that carries information from
    previous timesteps, creating a form of memory.
    """)

    st.markdown("""
    <div class="info-card">
    <strong style="color: #7c6af5;">Why Sequential Data Needs Special Treatment</strong><br><br>
    Consider predicting tomorrow's temperature. A standard neural network would look at today's
    temperature in isolation. But temperature follows patterns — it was warm yesterday, cooling
    today, so it might be cooler tomorrow. An RNN captures these <em>temporal dependencies</em>
    by maintaining state across timesteps.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    At each timestep *t*, an RNN cell takes two inputs: the current input *x_t* (e.g., today's
    temperature) and the hidden state from the previous timestep *h_{t-1}* (memory). It produces
    a new hidden state *h_t* (updated memory) using the equation:
    """)

    st.latex(r"h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)")

    st.code("""
       x_1          x_2          x_3          x_4
        |            |            |            |
   +----v----+  +----v----+  +----v----+  +----v----+
   |  RNN    |--|  RNN    |--|  RNN    |--|  RNN    |
   |  Cell   |  |  Cell   |  |  Cell   |  |  Cell   |
   +----+----+  +----+----+  +----+----+  +----+----+
        |            |            |            |
       h_1          h_2          h_3          h_4
                                               |
                                          +----v----+
                                          |  Output |
                                          |  Layer  |
                                          +----+----+
                                               |
                                             y_hat""", language="text")

    st.markdown("""
    The same weight matrices *W_hh* and *W_xh* are shared across all timesteps. This parameter
    sharing is what makes RNNs efficient for sequences of any length — the model size does not
    grow with sequence length.
    """)

    st.markdown("---")

    # --- Section 2: Vanishing Gradient ---
    st.markdown("### 2. The Vanishing Gradient Problem")
    st.markdown("""
    While RNNs are theoretically capable of learning long-range dependencies, in practice they struggle.
    During backpropagation through time (BPTT), gradients must flow through many timesteps. At each step,
    the gradient is multiplied by the weight matrix and passed through the tanh derivative.
    """)

    st.markdown("""
    <div class="info-card" style="border-color: rgba(245, 200, 106, 0.4); background: rgba(245, 200, 106, 0.05);">
    <strong style="color: #f5d76a;">The Core Problem</strong><br><br>
    When the tanh derivative (which is always &le; 1) and weight values are repeatedly multiplied
    across many timesteps, the gradient either <strong>vanishes</strong> (approaches 0, so the network
    cannot learn from distant past inputs) or <strong>explodes</strong> (grows extremely large, making
    training unstable).<br><br>
    For temperature forecasting with 120 hourly observations, a vanilla RNN would struggle to
    learn patterns from 5 days ago because the gradient signal would effectively disappear
    after passing through ~120 multiplication steps.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("This is precisely why **LSTMs** were invented.")

    st.markdown("---")

    # --- Section 3: LSTM Architecture ---
    st.markdown("### 3. LSTM Architecture")
    st.markdown("""
    The **Long Short-Term Memory (LSTM)** network, introduced by Hochreiter & Schmidhuber (1997),
    solves the vanishing gradient problem with a carefully designed gating mechanism. Instead of
    a single hidden state, an LSTM cell maintains two states: the **cell state** *c_t* (the long-term
    memory highway) and the **hidden state** *h_t* (the working memory and output).
    """)

    st.markdown("#### The Four Components of an LSTM Cell")

    st.markdown("""
    <div class="info-card" style="border-color: #7c6af5;">
    <strong style="color: #7c6af5;">1. Forget Gate</strong><br><br>
    Decides what information to <em>discard</em> from the cell state. The sigmoid function outputs
    values between 0 and 1 — a value of 0 means "completely forget this," and 1 means "completely
    keep this." For temperature forecasting, the forget gate might learn to discard information from
    unusually anomalous readings while retaining the general trend.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)")

    st.markdown("""
    <div class="info-card" style="border-color: #7c6af5;">
    <strong style="color: #7c6af5;">2. Input Gate</strong><br><br>
    Decides what <em>new</em> information to store in the cell state. It has two parts: a sigmoid
    layer decides <em>which</em> values to update, and a tanh layer creates a vector of <em>candidate</em>
    new values. Together, they determine what new information enters the cell state.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)")
    st.latex(r"\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)")

    st.markdown("""
    <div class="info-card" style="border-color: #7c6af5;">
    <strong style="color: #7c6af5;">3. Cell State Update</strong><br><br>
    Combines the forget gate and input gate to update the long-term memory. This is the key
    innovation: the cell state is updated through <em>addition</em>, not multiplication. Gradients
    can flow through the addition operation without vanishing, enabling the network to remember
    information across many timesteps.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t")

    st.markdown("""
    <div class="info-card" style="border-color: #7c6af5;">
    <strong style="color: #7c6af5;">4. Output Gate</strong><br><br>
    Decides what part of the cell state to <em>output</em> as the hidden state. The output gate
    filters the cell state to produce the hidden state, which serves as both the output at this
    timestep and the input to the next LSTM cell.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)")
    st.latex(r"h_t = o_t \odot \tanh(c_t)")

    st.code("""
            Cell State Highway (c_{t-1} ----------------> c_t)
                     |                           ^
                     |     +---------+           |
                     +---->|  Forget  |---- x ----+
                     |     |  Gate    |           |
                     |     +---------+           |
                     |                    +------+
                     |     +---------+    |      |
                     +---->|  Input   |-- x -- + --
                     |     |  Gate    |    |
                     |     +---------+    |
                     |     +---------+    |
                     +---->|Candidate |----+
                     |     |  (tanh)  |
                     |     +---------+
                     |                        +---------+
                     +----------------------->|  Output  |--> h_t
                                              |  Gate    |
                          [h_{t-1}, x_t]      +---------+""", language="text")

    st.markdown("""
    The cell state acts as a **highway** for gradient flow. During backpropagation, the gradient
    passes through the cell state update equation. Since *f_t* is controlled by a learned gate
    (not a fixed weight matrix), the network can learn to keep the forget gate close to 1 for
    important long-term information, allowing gradients to flow unimpeded across hundreds of timesteps.
    """)

    st.markdown("---")

    # --- Section 4: Time-Series Forecasting ---
    st.markdown("### 4. Time-Series Forecasting with LSTMs")
    st.markdown("""
    Time-series forecasting is the task of predicting future values based on past observations.
    Temperature is a classic example — it follows daily cycles (warm during day, cool at night),
    seasonal patterns (warm in summer, cold in winter), and weather-driven trends.
    """)

    st.markdown("#### Classification vs. Regression")
    st.markdown("""
    | Aspect | Text Generation (Classification) | Temperature Forecasting (Regression) |
    |--------|----------------------------------|--------------------------------------|
    | Input | Sequence of word IDs (discrete integers) | Sequence of temperature values (continuous floats) |
    | Embedding | Required (word ID to dense vector) | Not needed (values are already continuous) |
    | Output | Probability distribution over vocabulary | Single scalar value (predicted temperature) |
    | Loss Function | CrossEntropyLoss | Mean Squared Error (MSE) |
    | Evaluation | Accuracy, Perplexity | RMSE (Root Mean Squared Error) |
    """)

    st.markdown("""
    The LSTM processes the sequence of past temperatures and its final hidden state *h_T* encodes
    the temporal patterns. A linear layer then projects this encoding to a single predicted value.
    """)

    st.markdown("---")

    # --- Section 5: Sliding Window ---
    st.markdown("### 5. The Sliding Window Approach")
    st.markdown("""
    To train an LSTM for forecasting, we need to create input-output pairs from the raw time series.
    The **sliding window** (or rolling window) approach does this by moving a fixed-size window
    across the data:
    """)

    st.code("""
Time Series: [t1, t2, t3, t4, t5, t6, t7, t8, t9, ...]

Window size = 5:

  Window 1: [t1, t2, t3, t4, t5]  ->  target: t6
  Window 2: [t2, t3, t4, t5, t6]  ->  target: t7
  Window 3: [t3, t4, t5, t6, t7]  ->  target: t8
  Window 4: [t4, t5, t6, t7, t8]  ->  target: t9""", language="text")

    st.markdown("""
    Each window becomes one training sample. The input *X* is the window of past values, and
    the target *y* is the next value immediately following the window.

    In our implementation, we use a window of **120 hourly observations** (5 days). This choice
    captures **daily cycles** (the 24-hour temperature oscillation), **multi-day trends** (warming
    or cooling patterns over several days), and **weather patterns** (frontal systems that typically
    last 2-5 days). A window that is too short (e.g., 6 hours) would miss daily cycles. A window
    that is too long (e.g., 30 days) would add computational cost without proportional benefit,
    as very old data has diminishing predictive value for the next hour.
    """)

    st.markdown("---")

    # --- Section 6: Normalization ---
    st.markdown("### 6. Min-Max Normalization")
    st.markdown("""
    Neural networks learn most efficiently when input values are small and in a consistent range.
    Raw temperature values in the Jena dataset range from approximately -23 C to +37 C.
    **Min-Max scaling** transforms these to the range [0, 1]:
    """)

    st.latex(r"x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}")

    st.markdown("For the inverse transformation (converting predictions back to degrees C):")

    st.latex(r"x_{original} = x_{normalized} \times (x_{max} - x_{min}) + x_{min}")

    st.markdown("""
    Normalization provides three key benefits. First, **gradient stability** — large input values produce
    large gradients, which can cause unstable training, and normalized values keep gradients in a
    manageable range. Second, **faster convergence** — when inputs are on a similar scale, the loss
    surface is more uniform, and gradient descent converges faster. Third, **numerical precision** —
    Float32 arithmetic is more precise near 0 than at large magnitudes.
    """)

    st.markdown("#### Min-Max vs. Standard Scaling")
    st.markdown("""
    | Method | Formula | Range | Best For |
    |--------|---------|-------|----------|
    | Min-Max | (x - min) / (max - min) | [0, 1] | Bounded data, preserves distribution shape |
    | Standard (Z-score) | (x - mean) / std | Unbounded | Gaussian-distributed data, outlier-resistant |

    We chose Min-Max scaling because temperature has natural physical bounds and we want the
    output to be bounded, matching common neural network activation ranges.
    """)

    st.markdown("---")

    # --- Section 7: Training ---
    st.markdown("### 7. Training the Model")
    st.markdown("""
    Training optimizes the model's weights to minimize prediction error. Our training loop follows
    these steps for each epoch:

    **Step 1 — Forward pass:** Feed a batch of input sequences through the LSTM to get predictions.

    **Step 2 — Loss computation:** Calculate Mean Squared Error between predictions and actual values.
    """)
    st.latex(r"MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2")

    st.markdown("""
    **Step 3 — Backpropagation through time (BPTT):** Compute gradients of the loss with respect
    to all weights by unrolling the LSTM across timesteps.

    **Step 4 — Gradient clipping:** Cap gradient norms at 1.0 to prevent exploding gradients.

    **Step 5 — Weight update:** Apply the Adam optimizer to update weights.
    """)

    st.markdown("#### Training Strategies")

    st.markdown("""
    <div class="info-card" style="border-color: rgba(106, 245, 200, 0.4); background: rgba(106, 245, 200, 0.05);">
    <strong style="color: #6af5c8;">Early Stopping</strong><br><br>
    We monitor validation loss after each epoch. If it does not improve for 10 consecutive
    epochs, training stops early to prevent overfitting. The best model checkpoint (lowest
    validation loss) is restored.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="border-color: rgba(106, 245, 200, 0.4); background: rgba(106, 245, 200, 0.05);">
    <strong style="color: #6af5c8;">Learning Rate Scheduling</strong><br><br>
    We use <code>ReduceLROnPlateau</code> — if validation loss plateaus for 5 epochs,
    the learning rate is halved. This allows fine-grained optimization as the model
    approaches a minimum.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="border-color: rgba(106, 245, 200, 0.4); background: rgba(106, 245, 200, 0.05);">
    <strong style="color: #6af5c8;">Chronological Data Split</strong><br><br>
    Time-series data must be split <em>chronologically</em>, not randomly. Random splitting would
    leak future information into the training set (data leakage), producing misleadingly good
    results. We use the earliest 80% for training and the most recent 20% for testing.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Section 8: Evaluation ---
    st.markdown("### 8. Evaluation: RMSE")
    st.markdown("""
    **Root Mean Squared Error (RMSE)** is the standard evaluation metric for regression tasks.
    It measures the average magnitude of prediction errors:
    """)

    st.latex(r"RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}")

    st.markdown("""
    RMSE is measured in the same units as the target variable (degrees C), making it directly
    interpretable — an RMSE of 2.0 means "predictions are off by about 2 C on average." It also
    penalizes large errors more heavily than small ones due to the squaring, which is desirable
    because a 10 C error is much worse than ten 1 C errors.
    """)

    st.markdown("#### Interpreting RMSE")
    st.markdown("""
    | RMSE (C) | Interpretation |
    |----------|----------------|
    | < 1.0 | Excellent — predictions are very close to actual values |
    | 1.0 - 2.0 | Good — typical for hourly temperature prediction |
    | 2.0 - 4.0 | Moderate — room for improvement |
    | > 4.0 | Poor — model may need more capacity or data |
    """)

    st.markdown("---")

    # --- Section 9: Jena Dataset ---
    st.markdown("### 9. The Jena Climate Dataset")
    st.markdown("""
    The Jena Climate dataset is a widely used benchmark for time-series forecasting, collected
    by the weather station at the Max Planck Institute for Biogeochemistry in Jena, Germany.

    | Property | Value |
    |----------|-------|
    | Location | Jena, Germany (50.9 N, 11.6 E) |
    | Period | January 1, 2009 — December 31, 2016 |
    | Recording Interval | Every 10 minutes |
    | Total Observations | ~420,000 |
    | Number of Features | 14 |
    """)

    st.markdown("#### Features in the Dataset")
    st.markdown("""
    | Column | Description | Unit |
    |--------|-------------|------|
    | Date Time | Timestamp | DD.MM.YYYY HH:MM:SS |
    | p (mbar) | Atmospheric pressure | mbar |
    | **T (degC)** | **Air temperature (our target)** | **C** |
    | Tpot (K) | Potential temperature | K |
    | Tdew (degC) | Dew point temperature | C |
    | rh (%) | Relative humidity | % |
    | VPmax (mbar) | Saturation vapor pressure | mbar |
    | VPact (mbar) | Actual vapor pressure | mbar |
    | VPdef (mbar) | Vapor pressure deficit | mbar |
    | sh (g/kg) | Specific humidity | g/kg |
    | H2OC (mmol/mol) | Water vapor concentration | mmol/mol |
    | rho (g/m3) | Air density | g/m3 |
    | wv (m/s) | Wind speed | m/s |
    | max. wv (m/s) | Maximum wind speed | m/s |
    | wd (deg) | Wind direction | degrees |
    """)

    st.markdown("""
    The raw dataset at 10-minute intervals contains ~420,000 data points. Using all of them
    with a window of 720 timesteps (5 days at 10-min resolution) would create an enormous
    number of training sequences. By subsampling every 6th row, we get hourly data (~70,000
    points) with a window of 120 timesteps (still 5 days). This dramatically reduces
    computation while preserving the important temperature patterns.
    """)

    st.markdown("---")

    # --- Section 10: Implementation ---
    st.markdown("### 10. Our Implementation")
    st.markdown("#### Model Architecture")

    st.code("""
Input: (batch, 120, 1) -- 120 hours of normalized temperature
               |
        +------v------+
        |   LSTM       |  hidden_size=64
        |   Layer 1    |  processes all 120 timesteps
        +------+------+
               |
        +------v------+
        |   Dropout    |  p=0.2
        +------+------+
               |
        +------v------+
        |   LSTM       |  hidden_size=64
        |   Layer 2    |  returns final hidden state h_T
        +------+------+
               |
        +------v------+
        |   Dropout    |  p=0.2
        +------+------+
               |
        +------v------+
        |   Linear     |  64 -> 1
        +------+------+
               |
Output: (batch, 1) -- predicted temperature (normalized)""", language="text")

    st.markdown("#### PyTorch Implementation")
    st.code("""
class TemperatureLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_units=64,
                 num_layers=2, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1) -- normalized temperatures
        out, (h, c) = self.lstm(x)
        # h[-1]: final hidden state from last LSTM layer
        final_hidden = self.dropout(h[-1])
        return self.fc(final_hidden)  # (batch, 1)""", language="python")

    st.markdown("#### Data Pipeline")
    st.code("""
# 1. Load CSV
df = pd.read_csv('jena_climate_2009_2016.csv')

# 2. Subsample to hourly (every 6th row)
df = df.iloc[::6].reset_index(drop=True)

# 3. Extract temperature column
temperature = df['T (degC)'].values

# 4. Min-Max normalize to [0, 1]
scaler = MinMaxScaler()
normalized = scaler.fit_transform(temperature.reshape(-1, 1))

# 5. Create sliding window sequences
X, y = [], []
for i in range(120, len(normalized)):
    X.append(normalized[i-120:i])  # 120 past hours
    y.append(normalized[i])         # next hour

# 6. Chronological split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]""", language="python")

    st.markdown("---")

    # --- Section 11: Hyperparameters ---
    st.markdown("### 11. Hyperparameter Choices")
    st.markdown("""
    | Parameter | Value | Rationale |
    |-----------|-------|-----------|
    | Sequence length | 120 hours (5 days) | Captures daily cycles and multi-day weather trends |
    | Hidden units | 64 | Sufficient capacity for univariate forecasting without overfitting |
    | LSTM layers | 2 | Stacking enables hierarchical feature extraction (short-term + long-term patterns) |
    | Dropout | 0.2 | Mild regularization; prevents memorizing noise in temperature readings |
    | Batch size | 256 | Large enough for stable gradient estimates; small enough to fit in memory |
    | Learning rate | 0.001 | Standard Adam default; reduced automatically by scheduler on plateau |
    | Subsample step | 6 (hourly) | Reduces dataset from 420k to 70k points while preserving patterns |
    | Optimizer | Adam | Adaptive learning rates per parameter; robust to hyperparameter choices |
    """)

    st.markdown("---")

    # --- Section 12: Observations ---
    st.markdown("### 12. Observations and Results")
    st.markdown("""
    **Daily patterns are learned quickly.** The model captures the 24-hour temperature cycle within the
    first few epochs, as this is the strongest signal in the data.

    **Seasonal trends require more data.** The model sees seasonal variation across the 5+ years of data.
    The chronological test split means the test set contains weather patterns not seen in training
    (the most recent years).

    **Extreme temperatures are harder.** The model tends to under-predict temperature spikes and
    over-predict cold snaps, as these events are less frequent in the training data.

    **Subsampling to hourly is effective.** 10-minute resolution adds noise without meaningful additional
    information for next-hour prediction.

    **Two LSTM layers outperform one.** The second layer helps capture hierarchical patterns — intra-day
    variation on layer 1, multi-day trends on layer 2.
    """)

    st.markdown("#### Potential Improvements")
    st.markdown("""
    **Multivariate input** would include pressure, humidity, and wind speed as additional features.
    These variables influence temperature and could improve predictions.

    **Attention mechanisms** would allow the model to attend to specific past timesteps rather than
    relying solely on a compressed hidden state.

    **Longer prediction horizons** could predict 24 or 48 hours ahead instead of just the next hour,
    using a sequence-to-sequence architecture.

    **Ensemble methods** would train multiple models with different hyperparameters and average their
    predictions for more robust forecasting.

    **Transformer-based models** such as the Temporal Fusion Transformer can outperform LSTMs on
    complex time-series tasks.
    """)

    st.markdown("---")

    # --- Section 13: References ---
    st.markdown("### 13. References")
    st.markdown("""
    1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
    *Neural Computation*, 9(8), 1735-1780.

    2. Graves, A. (2012). *Supervised Sequence Labelling with Recurrent Neural Networks*. Springer.

    3. Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
    Chapter 6: Deep learning for text and sequences.

    4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
    Chapter 10: Sequence Modeling.

    5. Max Planck Institute for Biogeochemistry. Jena Climate Dataset.
    Department of Biogeochemical Integration.

    6. PyTorch Documentation. torch.nn.LSTM. PyTorch Foundation.
    """)

    st.markdown("""
    <div style="text-align: center; color: #aaaacc; margin-top: 20px; padding-top: 15px;
    border-top: 1px solid #333355; font-size: 0.9em;">
    AIT-204 — LSTM Temperature Forecasting with the Jena Climate Dataset
    </div>
    """, unsafe_allow_html=True)
