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
    st.markdown("""
    <h2 style='text-align: center; background: linear-gradient(135deg, #7c6af5, #6af5c8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    LSTM & RNN Guide
    </h2>
    <p style='text-align: center; color: #aaaacc;'>
    Understanding Recurrent Neural Networks for Character-Level Text Generation
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
    <div class="generated-text-box" style="border-color: #7c6af5;">
    <strong style="color: #7c6af5;">Why Sequential Data Needs Special Treatment</strong><br><br>
    Consider predicting the next word in a sentence. A standard neural network would look at each
    word in isolation. But language follows patterns — "the cat sat on the" strongly suggests "mat"
    or "chair" as the next word. An RNN captures these <em>sequential dependencies</em>
    by maintaining state across timesteps, letting past context inform future predictions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    At each timestep *t*, an RNN cell takes two inputs: the current input *x_t* (e.g., the current
    word embedding) and the hidden state from the previous timestep *h_{t-1}* (memory). It produces
    a new hidden state *h_t* (updated memory) using the equation:
    """)

    st.latex(r"h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)")

    st.code("""
       x_1          x_2          x_3          x_4
    ("the")      ("cat")      ("sat")      ("on")
        |            |            |            |
   +----v----+  +----v----+  +----v----+  +----v----+
   |  RNN    |--|  RNN    |--|  RNN    |--|  RNN    |
   |  Cell   |  |  Cell   |  |  Cell   |  |  Cell   |
   +----+----+  +----+----+  +----+----+  +----+----+
        |            |            |            |
       h_1          h_2          h_3          h_4
                                               |
                                          +----v----+
                                          | Softmax |
                                          |  Layer  |
                                          +----+----+
                                               |
                                          P(next word)""", language="text")

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
    <div class="generated-text-box" style="border-color: rgba(245, 200, 106, 0.4); background: linear-gradient(135deg, rgba(245, 200, 106, 0.05), rgba(245, 200, 106, 0.1));">
    <strong style="color: #f5d76a;">The Core Problem</strong><br><br>
    When the tanh derivative (which is always &le; 1) and weight values are repeatedly multiplied
    across many timesteps, the gradient either <strong>vanishes</strong> (approaches 0, so the network
    cannot learn from distant past inputs) or <strong>explodes</strong> (grows extremely large, making
    training unstable).<br><br>
    For text generation with a sequence length of 50 tokens, a vanilla RNN would struggle to
    learn that an opening quotation mark 40 words ago means a closing quote is needed, because
    the gradient signal would effectively disappear after passing through ~50 multiplication steps.
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
    <div class="generated-text-box" style="border-color: #7c6af5;">
    <strong style="color: #7c6af5;">1. Forget Gate</strong><br><br>
    Decides what information to <em>discard</em> from the cell state. The sigmoid function outputs
    values between 0 and 1 — a value of 0 means "completely forget this," and 1 means "completely
    keep this." For text generation, the forget gate might learn to discard information about a
    completed clause while retaining the overall topic and tone of the passage.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)")

    st.markdown("""
    <div class="generated-text-box" style="border-color: #7c6af5;">
    <strong style="color: #7c6af5;">2. Input Gate</strong><br><br>
    Decides what <em>new</em> information to store in the cell state. It has two parts: a sigmoid
    layer decides <em>which</em> values to update, and a tanh layer creates a vector of <em>candidate</em>
    new values. Together, they determine what new information enters the cell state.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)")
    st.latex(r"\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)")

    st.markdown("""
    <div class="generated-text-box" style="border-color: #7c6af5;">
    <strong style="color: #7c6af5;">3. Cell State Update</strong><br><br>
    Combines the forget gate and input gate to update the long-term memory. This is the key
    innovation: the cell state is updated through <em>addition</em>, not multiplication. Gradients
    can flow through the addition operation without vanishing, enabling the network to remember
    information across many timesteps.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t")

    st.markdown("""
    <div class="generated-text-box" style="border-color: #7c6af5;">
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

    # --- Section 4: Text Generation as Classification ---
    st.markdown("### 4. Text Generation as a Classification Task")
    st.markdown("""
    Text generation is fundamentally a **classification problem** — at each step, the model must
    choose the next word from a fixed vocabulary. This differs from regression tasks like temperature
    forecasting where the output is a continuous value.
    """)

    st.markdown("#### How It Differs from Regression")
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
    The LSTM processes the sequence of input word embeddings and its final hidden state *h_T*
    encodes the sequential context. A fully connected layer then projects this encoding to a
    vector of size *V* (the vocabulary size), and a softmax function converts it into a
    probability distribution over all possible next words.
    """)

    st.markdown("---")

    # --- Section 5: Word Embeddings ---
    st.markdown("### 5. Word Embeddings")
    st.markdown("""
    Words are discrete symbols — they have no inherent numerical relationship. The word "cat" is
    not mathematically closer to "dog" than to "refrigerator" in a raw integer encoding. **Word
    embeddings** solve this by mapping each word to a dense vector in a continuous space, where
    semantically similar words end up near each other.
    """)

    st.latex(r"\text{Embedding}: \mathbb{Z}^{|V|} \rightarrow \mathbb{R}^{d}")

    st.markdown("""
    In our model, the embedding dimension is **100**, meaning each word in the vocabulary is
    represented as a 100-dimensional vector. These vectors are learned during training — the
    model discovers which dimensions are useful for capturing word relationships. After training,
    words that appear in similar contexts (like "warm" and "hot," or "the" and "a") will have
    embedding vectors that are close together in this 100-dimensional space.

    The embedding layer is the first layer of our text generation model. It takes a sequence of
    integer word IDs and outputs a sequence of dense vectors, which the LSTM then processes.
    """)

    st.markdown("---")

    # --- Section 6: Tokenization ---
    st.markdown("### 6. Tokenization and Sequence Creation")
    st.markdown("""
    Before the LSTM can process text, the raw text must be converted into numerical sequences.
    This involves two steps: **tokenization** (mapping words to integers) and **sequence creation**
    (building input-output pairs for training).
    """)

    st.markdown("#### Tokenization")
    st.markdown("""
    The tokenizer builds a vocabulary from the training corpus, assigning each unique word an
    integer ID. Words that appear below a frequency threshold are mapped to a special unknown
    token. The tokenizer is saved as a pickle file so the same mapping can be used during inference.
    """)

    st.markdown("#### Sliding Window for Text")
    st.code("""
Text:    "the cat sat on the warm mat by the fire"
Tokens:  [4, 12, 87, 15, 4, 203, 156, 31, 4, 89]

Sequence length = 5:

  Input: [4, 12, 87, 15, 4]    ->  Target: 203  ("warm")
  Input: [12, 87, 15, 4, 203]  ->  Target: 156  ("mat")
  Input: [87, 15, 4, 203, 156] ->  Target: 31   ("by")
  Input: [15, 4, 203, 156, 31] ->  Target: 4    ("the")""", language="text")

    st.markdown("""
    Each window of *sequence_length* words becomes one training sample. The input *X* is the
    window of word IDs, and the target *y* is the single word ID that follows the window.
    Our model uses a sequence length of **50 words**, which provides enough context for the
    LSTM to learn sentence structure, paragraph flow, and stylistic patterns from the training
    corpus.
    """)

    st.markdown("---")

    # --- Section 7: Temperature Sampling ---
    st.markdown("### 7. Temperature Sampling")
    st.markdown("""
    During text generation, the model outputs a probability distribution over the entire vocabulary.
    Rather than always picking the most probable word (which produces repetitive, boring text), we
    use **temperature sampling** to control the randomness of the selection.
    """)

    st.markdown("The temperature *T* rescales the model's raw logits before applying softmax:")

    st.latex(r"P(w_i) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}")

    st.markdown("""
    **Low temperature (0.3-0.7)** sharpens the distribution, making the model strongly prefer
    high-probability words. The output is conservative, predictable, and grammatically safe, but
    can feel repetitive.

    **Medium temperature (0.8-1.2)** maintains the learned distribution roughly as-is, producing
    a balance of coherence and creativity. A temperature of 1.0 is the unmodified distribution.

    **High temperature (1.3-2.0+)** flattens the distribution, giving lower-probability words a
    better chance of being selected. The output becomes more creative and surprising, but risks
    incoherence as rare words are chosen more frequently.

    Temperature is the key creative control in text generation. It lets users dial between
    "safe and repetitive" and "wild and creative" without retraining the model.
    """)

    st.markdown("---")

    # --- Section 8: Model Architecture ---
    st.markdown("### 8. Our Model Architecture")

    st.code("""
Input: (batch, 50) -- sequence of 50 word IDs
               |
        +------v------+
        | Embedding    |  vocab_size -> 100 dims
        +------+------+
               |
        +------v------+
        |   LSTM       |  hidden_size=150
        |   Layer      |  processes all 50 timesteps
        +------+------+
               |
        +------v------+
        |   Dropout    |  p=0.2
        +------+------+
               |
        +------v------+
        |   Linear     |  150 -> vocab_size
        +------+------+
               |
        +------v------+
        |   Softmax    |  probability over vocabulary
        +------+------+
               |
Output: (batch, vocab_size) -- P(next word)""", language="text")

    st.markdown("#### PyTorch Implementation")
    st.code("""
class TextGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100,
                 lstm_units=150, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) -- word IDs
        embedded = self.embedding(x)       # (batch, seq_len, 100)
        out, (h, c) = self.lstm(embedded)  # process sequence
        final_hidden = self.dropout(h[-1]) # last hidden state
        return self.fc(final_hidden)       # (batch, vocab_size)""", language="python")

    st.markdown("---")

    # --- Section 9: Training ---
    st.markdown("### 9. Training the Model")
    st.markdown("""
    Training optimizes the model's weights to minimize prediction error. For text generation,
    we use **Cross-Entropy Loss**, which measures how well the predicted probability distribution
    matches the true next word:
    """)

    st.latex(r"\mathcal{L} = -\sum_{i=1}^{V} y_i \log(\hat{y}_i)")

    st.markdown("""
    where *y* is the one-hot encoded true next word and *y_hat* is the predicted probability
    distribution. Since only one word is correct, this simplifies to the negative log probability
    of the correct word.

    The training loop follows these steps for each epoch:

    **Step 1 — Forward pass:** Feed a batch of input sequences through the embedding and LSTM
    to get probability distributions over the vocabulary.

    **Step 2 — Loss computation:** Calculate Cross-Entropy Loss between predictions and actual
    next words.

    **Step 3 — Backpropagation through time (BPTT):** Compute gradients of the loss with respect
    to all weights by unrolling the LSTM across timesteps.

    **Step 4 — Gradient clipping:** Cap gradient norms at 1.0 to prevent exploding gradients.

    **Step 5 — Weight update:** Apply the Adam optimizer to update weights.
    """)

    st.markdown("#### Training Strategies")

    st.markdown("""
    <div class="generated-text-box" style="border-color: rgba(106, 245, 200, 0.4); background: linear-gradient(135deg, rgba(106, 245, 200, 0.03), rgba(106, 245, 200, 0.08));">
    <strong style="color: #6af5c8;">Early Stopping</strong><br><br>
    We monitor validation loss after each epoch. If it does not improve for 10 consecutive
    epochs, training stops early to prevent overfitting. The best model checkpoint (lowest
    validation loss) is restored. This prevents the model from simply memorizing the training
    text rather than learning generalizable language patterns.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="generated-text-box" style="border-color: rgba(106, 245, 200, 0.4); background: linear-gradient(135deg, rgba(106, 245, 200, 0.03), rgba(106, 245, 200, 0.08));">
    <strong style="color: #6af5c8;">Learning Rate Scheduling</strong><br><br>
    We use <code>ReduceLROnPlateau</code> — if validation loss plateaus for 5 epochs,
    the learning rate is halved. This allows fine-grained optimization as the model
    approaches a minimum, preventing it from overshooting good parameter configurations.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="generated-text-box" style="border-color: rgba(106, 245, 200, 0.4); background: linear-gradient(135deg, rgba(106, 245, 200, 0.03), rgba(106, 245, 200, 0.08));">
    <strong style="color: #6af5c8;">Validation Split</strong><br><br>
    10% of the training data is held out for validation. Unlike time-series data which must be
    split chronologically, text data can be shuffled before splitting since we are learning
    general language patterns rather than temporal trends.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Section 10: Evaluation ---
    st.markdown("### 10. Evaluation Metrics")
    st.markdown("""
    Text generation models are evaluated differently from regression models. The two primary
    metrics are **accuracy** and **perplexity**.

    **Accuracy** measures the percentage of time the model's highest-probability prediction
    matches the actual next word. For language modeling, even 20-30% accuracy can produce
    reasonable text, because many positions have multiple valid continuations.

    **Perplexity** is the exponentiation of the cross-entropy loss:
    """)

    st.latex(r"\text{Perplexity} = e^{\mathcal{L}} = e^{-\frac{1}{N}\sum \log P(w_i)}")

    st.markdown("""
    Perplexity can be interpreted as the effective number of equally likely choices the model
    considers at each step. A perplexity of 50 means the model is, on average, as uncertain as
    if it were choosing uniformly among 50 words. Lower perplexity indicates a more confident
    and accurate model.
    """)

    st.markdown("---")

    # --- Section 11: Hyperparameters ---
    st.markdown("### 11. Hyperparameter Choices")
    st.markdown("""
    | Parameter | Value | Rationale |
    |-----------|-------|-----------|
    | Sequence length | 50 words | Long enough to capture sentence structure and paragraph context |
    | Embedding dim | 100 | Standard size for moderate vocabularies; captures word relationships |
    | LSTM units | 150 | Sufficient capacity for language patterns without excessive overfitting |
    | Dropout | 0.2 | Mild regularization; prevents memorizing specific phrases verbatim |
    | Batch size | 128 | Balances gradient stability with memory constraints |
    | Learning rate | 0.001 | Standard Adam default; reduced automatically by scheduler on plateau |
    | Epochs | 100 (max) | Early stopping typically halts training well before this limit |
    | Vocab size | 4,167 | All unique words in the training corpus above frequency threshold |
    | Validation split | 10% | Enough data to reliably estimate generalization performance |
    """)

    st.markdown("---")

    # --- Section 12: Observations ---
    st.markdown("### 12. Observations and Results")
    st.markdown("""
    **Common words and patterns are learned first.** The model quickly learns high-frequency
    words like "the," "and," and "of," along with basic subject-verb-object structure. This
    produces grammatically plausible but generic text within the first few epochs.

    **Style and vocabulary emerge with more training.** As training progresses, the model picks
    up stylistic patterns from the training corpus — characteristic phrases, sentence lengths,
    and thematic vocabulary. The generated text begins to feel more like the source material.

    **Temperature is the key creative control.** Low temperature (0.5) produces safe, repetitive
    text that closely mirrors the most common patterns. High temperature (1.5+) introduces variety
    but risks incoherence. A temperature around 0.8-1.0 typically produces the best balance.

    **Vocabulary size affects quality.** A larger vocabulary allows more expressive text but makes
    the classification problem harder (more classes to predict). Rare words are difficult for the
    model to learn because they appear in few training examples.

    **Overfitting manifests as memorization.** Without dropout and early stopping, the model
    begins reproducing exact passages from the training text rather than generating novel
    continuations. The gap between training loss and validation loss is the key indicator.
    """)

    st.markdown("#### Potential Improvements")
    st.markdown("""
    **Multi-layer LSTM** would stack two or more LSTM layers for hierarchical feature extraction —
    lower layers learning local syntax and upper layers capturing longer-range semantic patterns.

    **Attention mechanisms** would allow the model to attend to specific past tokens rather than
    relying solely on a compressed hidden state, improving coherence over longer generated passages.

    **Subword tokenization** (such as Byte-Pair Encoding) would handle rare and out-of-vocabulary
    words gracefully by breaking them into known subword units, dramatically reducing the unknown
    token rate.

    **Beam search** decoding would maintain multiple candidate sequences and select the one with
    the highest overall probability, producing more globally coherent text than greedy or purely
    sampled generation.

    **Transformer-based models** such as GPT use self-attention to process all tokens in parallel,
    achieving superior performance on text generation compared to sequential LSTM processing.
    """)

    st.markdown("---")

    # --- Section 13: References ---
    st.markdown("### 13. References")
    st.markdown("""
    1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
    *Neural Computation*, 9(8), 1735-1780.

    2. Graves, A. (2013). Generating Sequences with Recurrent Neural Networks.
    *arXiv preprint arXiv:1308.0850*.

    3. Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
    Chapter 8: Text generation with LSTM.

    4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
    Chapter 10: Sequence Modeling.

    5. Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space.
    *arXiv preprint arXiv:1301.3781*.

    6. PyTorch Documentation. torch.nn.LSTM & torch.nn.Embedding. PyTorch Foundation.
    """)

    st.markdown("""
    <div style="text-align: center; color: #aaaacc; margin-top: 20px; padding-top: 15px;
    border-top: 1px solid #333355; font-size: 0.9em;">
    AIT-204 — LSTM Text Generation
    </div>
    """, unsafe_allow_html=True)
