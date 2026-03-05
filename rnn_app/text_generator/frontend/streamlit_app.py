"""
streamlit_app.py - Streamlit Frontend for RNN Text Generator
=============================================================
This is the student-friendly web interface for the text generation API.

Run with:
    streamlit run streamlit_app.py

Configuration:
    - API URL is configurable in the sidebar (default: http://localhost:8000)
    - Requires the FastAPI backend to be running
    - Works with any backend that implements the /generate and /model/info endpoints

Note for students with alternative frontends:
    The backend API is language/framework agnostic. You can replace this
    Streamlit frontend with any HTTP client:
    - React (TypeScript/JavaScript)
    - Vue.js
    - Flutter (mobile)
    - Plain HTML + fetch()
    - Jupyter Notebook
    - Command-line curl

    All you need is to send:
        POST {API_URL}/generate
        Content-Type: application/json
        {"seed_text": "...", "num_words": 50, "temperature": 1.0}
"""

import streamlit as st
import requests
import json
import time
import base64
from io import BytesIO
import plotly.graph_objects as go

# --- PAGE CONFIGURATION -------------------------------------------------------
st.set_page_config(
    page_title="RNN Text Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Headers */
    h1, h2, h3 {
        color: #7c6af5 !important;
    }

    /* Generated text box */
    .generated-text-box {
        background: linear-gradient(135deg, #1e1e3a, #252545);
        border: 1px solid #7c6af5;
        border-radius: 12px;
        padding: 20px 25px;
        font-family: Georgia, serif;
        font-size: 1.1em;
        line-height: 1.8;
        color: #e0e0f0;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(124, 106, 245, 0.2);
    }

    /* Seed text highlighting */
    .seed-highlight {
        color: #7c6af5;
        font-weight: bold;
    }

    /* Generated continuation */
    .generated-highlight {
        color: #6af5c8;
    }

    /* Info cards */
    .info-card {
        background: rgba(124, 106, 245, 0.1);
        border: 1px solid rgba(124, 106, 245, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* Warning box */
    .warning-box {
        background: rgba(245, 200, 106, 0.1);
        border: 1px solid rgba(245, 200, 106, 0.4);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Success box */
    .success-box {
        background: rgba(106, 245, 194, 0.1);
        border: 1px solid rgba(106, 245, 194, 0.4);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Error box */
    .error-box {
        background: rgba(245, 106, 106, 0.1);
        border: 1px solid rgba(245, 106, 106, 0.4);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Temperature indicator */
    .temp-indicator {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
    }

    /* Code block styling */
    code {
        background: rgba(124, 106, 245, 0.2) !important;
        color: #c8c8ff !important;
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Streamlit metric cards */
    [data-testid="metric-container"] {
        background: rgba(124, 106, 245, 0.1);
        border: 1px solid rgba(124, 106, 245, 0.3);
        border-radius: 8px;
        padding: 10px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a, #1a1a2e);
        border-right: 1px solid rgba(124, 106, 245, 0.2);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #aaaacc;
    }

    .stTabs [aria-selected="true"] {
        color: #7c6af5 !important;
        border-bottom-color: #7c6af5 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c6af5, #5a4dd4);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #9080ff, #7c6af5);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 106, 245, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# --- CONSTANTS & DEFAULTS -----------------------------------------------------
DEFAULT_API_URL = "https://ait-204-pair-programing.onrender.com"
DEFAULT_SEED = "the cat sat on"
DEFAULT_NUM_WORDS = 50
DEFAULT_TEMPERATURE = 1.0


# --- HELPER FUNCTIONS ---------------------------------------------------------

def check_api_health(api_url: str) -> dict:
    """Check if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.status_code == 200:
            return {"status": "online", "data": response.json()}
        return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.ConnectionError:
        return {"status": "offline", "message": "Cannot connect to API"}
    except requests.Timeout:
        return {"status": "timeout", "message": "API request timed out"}


def generate_text(api_url: str, seed_text: str, num_words: int, temperature: float) -> dict:
    """Call the /generate endpoint and return the response."""
    try:
        response = requests.post(
            f"{api_url}/generate",
            json={
                "seed_text": seed_text,
                "num_words": num_words,
                "temperature": temperature
            },
            timeout=60  # Generation can take a while
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 503:
            return {
                "success": False,
                "error": "model_not_loaded",
                "message": "Model not trained yet. See instructions below."
            }
        else:
            return {
                "success": False,
                "error": "api_error",
                "message": response.json().get("detail", "Unknown error")
            }
    except requests.ConnectionError:
        return {
            "success": False,
            "error": "connection_error",
            "message": f"Cannot connect to API at {api_url}"
        }
    except Exception as e:
        return {"success": False, "error": "unknown", "message": str(e)}


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


def temperature_label(temp: float) -> str:
    """Return a descriptive label for a temperature value."""
    if temp < 0.7:
        return "Conservative"
    elif temp < 1.2:
        return "Balanced"
    else:
        return "Creative"


def format_generated_text(seed_text: str, full_text: str) -> str:
    """Format generated text with HTML highlighting for seed vs. generated parts."""
    seed_clean = seed_text.strip().lower()
    full_clean = full_text.strip().lower()

    if full_clean.startswith(seed_clean):
        seed_part = full_text[:len(seed_text)]
        generated_part = full_text[len(seed_text):]
        return (
            f'<span class="seed-highlight">{seed_part}</span>'
            f'<span class="generated-highlight">{generated_part}</span>'
        )
    return f'<span class="generated-highlight">{full_text}</span>'


# --- SIDEBAR ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    api_url = st.text_input(
        "API URL",
        value=DEFAULT_API_URL,
        help="URL of the FastAPI backend. Default: http://localhost:8000"
    )

    st.markdown("### Generation Settings")

    num_words = st.slider(
        "Words to Generate",
        min_value=10,
        max_value=500,
        value=DEFAULT_NUM_WORDS,
        step=10,
        help="How many new words the model generates"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.5,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help="Controls randomness: low=predictable, high=creative"
    )

    st.markdown(f"**Mode:** {temperature_label(temperature)}")

    # Temperature explanation
    with st.expander("What is Temperature?"):
        st.markdown("""
        Temperature controls how "random" the generation is:

        - **Low (0.3-0.7):** Very predictable, picks most likely words
        - **Medium (0.8-1.2):** Balanced creativity and coherence
        - **High (1.3-2.0):** Very creative, sometimes surprising

        *Mathematically:* `p_i = exp(log_p_i / T) / sum(exp(...))`
        """)

    st.markdown("---")

    # API Status
    st.markdown("### API Status")
    health = check_api_health(api_url)

    if health["status"] == "online":
        st.success("API Online")
    elif health["status"] == "offline":
        st.error("API Offline")
        st.caption(f"Start backend: `uvicorn app.main:app --reload`")
    else:
        st.warning(f"{health['status']}")

    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown(f"[API Docs]({api_url}/docs)")
    st.markdown(f"[ReDoc]({api_url}/redoc)")


# --- MAIN CONTENT -------------------------------------------------------------

st.markdown("""
<h1 style='text-align: center; background: linear-gradient(135deg, #7c6af5, #6af5c8);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em;'>
RNN Text Generator
</h1>
<p style='text-align: center; color: #aaaacc; font-size: 1.1em;'>
LSTM-powered language model for creative text generation
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -- TABS ----------------------------------------------------------------------
tab_generate, tab_batch, tab_model, tab_about = st.tabs([
    "Generate", "Variations", "Model Info", "About"
])


# -- TAB 1: GENERATE -----------------------------------------------------------
with tab_generate:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Enter Seed Text")
        seed_text = st.text_area(
            "Seed Text",
            value=DEFAULT_SEED,
            height=100,
            placeholder="Type your starting text here...",
            label_visibility="collapsed",
            help="The model will continue from this text"
        )

        generate_btn = st.button("Generate Text", use_container_width=True)

    with col2:
        st.markdown("### Current Settings")
        st.info(f"**Words:** {num_words}\n\n**Temperature:** {temperature}\n\n**Mode:** {temperature_label(temperature)}")

    # -- Generation Output -----------------------------------------------------
    st.markdown("---")

    if generate_btn:
        if not seed_text.strip():
            st.error("Please enter some seed text first!")
        else:
            with st.spinner("Generating text..."):
                start_time = time.time()
                result = generate_text(api_url, seed_text, num_words, temperature)
                elapsed = time.time() - start_time

            if result["success"]:
                data = result["data"]
                generated = data["generated_text"]

                # Display the generated text
                st.markdown("### Generated Text")
                formatted = format_generated_text(seed_text, generated)
                st.markdown(
                    f'<div class="generated-text-box">{formatted}</div>',
                    unsafe_allow_html=True
                )

                # Metadata row
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    word_count = len(generated.split())
                    st.metric("Total Words", word_count)
                with col_b:
                    st.metric("Characters", len(generated))
                with col_c:
                    st.metric("Gen. Time", f"{elapsed:.1f}s")
                with col_d:
                    st.metric("Temperature", temperature)

                # Copy button (text area for easy copying)
                with st.expander("Copy Generated Text"):
                    st.text_area("", value=generated, height=150, key="copy_area")

                # Legend
                st.markdown("""
                <div style="font-size: 0.85em; color: #888899; margin-top: 10px;">
                    <span style="color: #7c6af5; font-weight: bold;">Purple</span> = your seed text &nbsp;|&nbsp;
                    <span style="color: #6af5c8; font-weight: bold;">Cyan</span> = AI generated text
                </div>
                """, unsafe_allow_html=True)

            else:
                error_type = result.get("error", "unknown")
                if error_type == "model_not_loaded":
                    st.markdown("""
                    <div class="warning-box">
                    <h4>Model Not Trained Yet</h4>
                    <p>The API is running but no model has been trained. Follow these steps:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code("""
# Step 1: Navigate to the backend directory
cd rnn_app/backend

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train the model (uses sample text by default)
python app/train.py

# Step 4: Restart the API server
uvicorn app.main:app --reload

# Optional: Download a real book for better results
python app/train.py --download https://www.gutenberg.org/files/11/11-0.txt
                    """, language="bash")
                elif error_type == "connection_error":
                    st.markdown(f"""
                    <div class="error-box">
                    <h4>Cannot Connect to API</h4>
                    <p>Make sure the FastAPI backend is running at <code>{api_url}</code></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000",
                            language="bash")
                else:
                    st.error(f"Error: {result.get('message', 'Unknown error')}")

    else:
        # Show placeholder when not yet generated
        st.markdown("""
        <div class="generated-text-box" style="text-align: center; color: #555577; padding: 40px;">
            <p style="font-size: 1.5em;"></p>
            <p>Enter seed text and click <strong>Generate Text</strong> to see the LSTM output here.</p>
        </div>
        """, unsafe_allow_html=True)


# -- TAB 2: BATCH VARIATIONS ---------------------------------------------------
with tab_batch:
    st.markdown("### Generate Multiple Variations")
    st.markdown(
        "Generate several different continuations from the same seed to see how "
        "sampling randomness produces varied outputs."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        batch_seed = st.text_area(
            "Seed Text",
            value=DEFAULT_SEED,
            height=80,
            key="batch_seed"
        )
    with col2:
        num_variations = st.select_slider(
            "Number of Variations",
            options=[2, 3, 4, 5],
            value=3
        )
        batch_words = st.slider("Words Each", 20, 100, 30, key="batch_words")

    batch_btn = st.button("Generate Variations", use_container_width=True)

    if batch_btn:
        with st.spinner(f"Generating {num_variations} variations..."):
            results = []
            for i in range(num_variations):
                result = generate_text(api_url, batch_seed, batch_words, temperature)
                results.append(result)

        for i, result in enumerate(results):
            if result["success"]:
                text = result["data"]["generated_text"]
                with st.expander(f"Variation {i+1}", expanded=True):
                    formatted = format_generated_text(batch_seed, text)
                    st.markdown(
                        f'<div class="generated-text-box">{formatted}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.error(f"Variation {i+1} failed: {result.get('message', 'Error')}")

    st.markdown("---")
    st.markdown("### Temperature Comparison")
    st.markdown("See how different temperatures affect the same seed text:")

    compare_seed = st.text_input("Seed text for comparison", value=DEFAULT_SEED, key="compare_seed")
    compare_btn = st.button("Compare Temperatures", use_container_width=True)

    if compare_btn:
        temperatures = [0.5, 1.0, 1.5]
        labels = ["Conservative (0.5)", "Balanced (1.0)", "Creative (1.5)"]
        cols = st.columns(3)

        for col, temp, label in zip(cols, temperatures, labels):
            with col:
                st.markdown(f"**{label}**")
                with st.spinner("..."):
                    result = generate_text(api_url, compare_seed, 25, temp)
                if result["success"]:
                    text = result["data"]["generated_text"]
                    st.markdown(
                        f'<div class="generated-text-box" style="font-size:0.9em;">{text}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.error(result.get("message", "Error"))


# -- TAB 3: MODEL INFO ---------------------------------------------------------
with tab_model:
    st.markdown("### Model Information")

    info_result = get_model_info(api_url)

    if info_result["success"]:
        info = info_result["data"]

        if info.get("status") == "not_loaded":
            st.warning("Model not loaded. Train the model first.")
        else:
            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Vocabulary Size", f"{info.get('vocab_size', 'N/A'):,}")
            with col2:
                st.metric("Sequence Length", info.get('sequence_length', 'N/A'))
            with col3:
                st.metric("Total Parameters", f"{info.get('total_parameters', 0):,}")
            with col4:
                st.metric("Embedding Dim", info.get('config', {}).get('embedding_dim', 'N/A'))

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
                        <small style="color: #7c6af5;">Params: {layer.get('param_shapes', '')}</small>
                    </div>
                    {"<div style='text-align:center; color:#7c6af5;'>&#x2193;</div>" if arrow else ""}
                    """, unsafe_allow_html=True)

            with col_config:
                st.markdown("### Hyperparameters")
                config = info.get("config", {})
                for key, value in config.items():
                    if key != "vocab_size":
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
                st.image(img_bytes, use_column_width=False, width=400)
            else:
                st.info("Architecture diagram not available")

            # Training history
            st.markdown("---")
            st.markdown("### Training History")
            history = get_training_history(api_url)
            if history:
                epochs = list(range(1, len(history.get('loss', [])) + 1))
                chart_layout = dict(
                    plot_bgcolor='rgba(30,30,60,1)',
                    paper_bgcolor='rgba(20,20,40,1)',
                    font=dict(color='white'),
                    legend=dict(bgcolor='rgba(40,40,70,1)'),
                    xaxis_title="Epoch",
                )

                # Loss chart
                fig_loss = go.Figure()
                if 'loss' in history:
                    fig_loss.add_trace(go.Scatter(
                        x=epochs, y=history['loss'],
                        name='Training Loss', line=dict(color='#7c6af5', width=2)
                    ))
                if 'val_loss' in history:
                    fig_loss.add_trace(go.Scatter(
                        x=epochs, y=history['val_loss'],
                        name='Validation Loss', line=dict(color='#f56a6a', width=2, dash='dot')
                    ))
                fig_loss.update_layout(title="Loss Over Epochs", yaxis_title="Loss", **chart_layout)
                st.plotly_chart(fig_loss, use_container_width=True)

                # Accuracy chart
                fig_acc = go.Figure()
                if 'accuracy' in history:
                    fig_acc.add_trace(go.Scatter(
                        x=epochs, y=[v * 100 for v in history['accuracy']],
                        name='Training Accuracy', line=dict(color='#7c6af5', width=2)
                    ))
                if 'val_accuracy' in history:
                    fig_acc.add_trace(go.Scatter(
                        x=epochs, y=[v * 100 for v in history['val_accuracy']],
                        name='Validation Accuracy', line=dict(color='#f56a6a', width=2, dash='dot')
                    ))
                fig_acc.update_layout(title="Accuracy Over Epochs", yaxis_title="Accuracy (%)", **chart_layout)
                st.plotly_chart(fig_acc, use_container_width=True)

                # Summary stats
                st.markdown("#### Training Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Epochs", len(epochs))
                m2.metric("Best Val Loss", f"{min(history.get('val_loss', [0])):.4f}")
                m3.metric("Final Train Acc", f"{history.get('accuracy', [0])[-1]*100:.1f}%")
                m4.metric("Best Val Acc", f"{max(history.get('val_accuracy', [0]))*100:.1f}%")
            else:
                st.info("Training history not available. Train the model first.")
    else:
        st.error(f"Failed to fetch model info: {info_result.get('message')}")


# -- TAB 4: ABOUT --------------------------------------------------------------
with tab_about:
    st.markdown("### About This Application")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### What This App Does
        This application demonstrates **LSTM-based text generation**:

        1. **LSTM Model** learns patterns in text data
        2. **FastAPI Backend** serves the trained model via HTTP
        3. **Streamlit Frontend** (this page) provides the user interface

        #### System Architecture
        ```
        You (browser)
              | HTTP
        Streamlit App (:8501)
              | HTTP POST /generate
        FastAPI Server (:8000)
              | Keras inference
        LSTM Model (models/)
        ```

        #### Technology Stack
        | Component | Technology |
        |-----------|------------|
        | Deep Learning | PyTorch LSTM |
        | Backend API | FastAPI + Uvicorn |
        | Frontend | Streamlit |
        | Visualization | Plotly |
        """)

    with col2:
        st.markdown("""
        #### Building an Alternative Frontend

        The FastAPI backend accepts standard HTTP requests.
        You can build your own frontend using any technology!

        **API Contract (POST /generate):**
        ```json
        Request:
        {
            "seed_text": "string",
            "num_words": 50,
            "temperature": 1.0
        }

        Response:
        {
            "seed_text": "string",
            "generated_text": "string",
            "num_words_requested": 50,
            "temperature": 1.0,
            "model_info": {...}
        }
        ```

        **Alternative frontend ideas:**
        - React (TypeScript) SPA
        - Vue.js with Vuetify
        - Flutter (mobile app)
        - Jupyter Notebook widget
        - CLI tool (click/typer)
        """)

    st.markdown("---")
    st.markdown("""
    #### Learning Objectives
    This activity demonstrates:

    - **RNN/LSTM theory** -> working implementation (text_generator.py)
    - **ML pipeline**: data -> preprocessing -> sequences -> training -> inference
    - **API design**: RESTful endpoints with FastAPI and Pydantic validation
    - **Frontend integration**: Streamlit consuming a backend API
    - **Software architecture**: separation of concerns (data/model/api/ui)

    #### Project Structure
    ```
    rnn_app/
    ├── backend/
    │   ├── app/
    │   │   ├── text_generator.py    # Core LSTM logic
    │   │   ├── train.py             # Training script
    │   │   └── main.py              # FastAPI server
    │   ├── data/
    │   │   └── training_text.txt    # Training corpus
    │   ├── models/                  # Saved model files
    │   └── requirements.txt
    └── frontend/
        ├── streamlit_app.py         # This file
        └── requirements.txt
    ```
    """)

    # Quick start
    st.markdown("---")
    st.markdown("#### Quick Start")
    st.code("""
# Terminal 1: Backend
cd rnn_app/backend
pip install -r requirements.txt
python app/train.py          # Train the model
uvicorn app.main:app --reload  # Start API server

# Terminal 2: Frontend (this app)
cd rnn_app/frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
    """, language="bash")
