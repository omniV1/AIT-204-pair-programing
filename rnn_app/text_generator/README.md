# RNN Text Generator

An LSTM-based text generation application with a FastAPI backend and Streamlit frontend. Built as a learning project for AIT-204 to demonstrate recurrent neural networks, REST API design, and frontend/backend integration.

---

## Quick Start

### Step 1 — Install dependencies

```bash
# Backend
cd rnn_app/backend
pip install -r requirements.txt

# Frontend (separate terminal)
cd rnn_app/frontend
pip install -r requirements.txt
```

### Step 2 — Train the model

```bash
cd rnn_app/backend

# Train on the built-in sample text
python app/train.py

# Or train on a Project Gutenberg book (recommended for better results)
python app/train.py --download https://www.gutenberg.org/files/11/11-0.txt
```

Training typically takes 5–20 minutes depending on your hardware and the dataset size. The trained model is saved to `rnn_app/backend/models/`.

### Step 3 — Run the application

Open two terminals:

**Terminal 1 — API server:**
```bash
cd rnn_app/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit frontend:**
```bash
cd rnn_app/frontend
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`.

---

## Directory Structure

```
rnn_app/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Package marker
│   │   ├── text_generator.py    # Core LSTM model class
│   │   ├── train.py             # Training script (run this first)
│   │   └── main.py              # FastAPI application and endpoints
│   ├── data/
│   │   └── training_text.txt    # Training corpus (created after training)
│   ├── models/
│   │   ├── model.pt             # Saved PyTorch model weights (created after training)
│   │   └── tokenizer.pkl        # Fitted tokenizer (created after training)
│   └── requirements.txt
└── frontend/
    ├── streamlit_app.py         # Streamlit web interface
    └── requirements.txt
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check if the API is running |
| POST | `/generate` | Generate text from a seed string |
| GET | `/model/info` | Get model architecture and hyperparameters |
| GET | `/model/architecture-image` | Get a base64-encoded model diagram |
| GET | `/training/history` | Get loss curves from the last training run |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | ReDoc API documentation |

### POST /generate — Request body

```json
{
    "seed_text": "the cat sat on",
    "num_words": 50,
    "temperature": 1.0
}
```

### POST /generate — Response body

```json
{
    "seed_text": "the cat sat on",
    "generated_text": "the cat sat on the mat and looked at the...",
    "num_words_requested": 50,
    "temperature": 1.0,
    "model_info": {
        "vocab_size": 3500,
        "sequence_length": 20
    }
}
```

---

## Hyperparameter Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `embedding_dim` | 128 | 64–256 | Size of word embedding vectors. Larger = richer word representations, slower training. |
| `lstm_units` | 256 | 64–512 | Number of LSTM memory units. Larger = more capacity, higher risk of overfitting. |
| `num_lstm_layers` | 2 | 1–4 | Stacked LSTM depth. Deeper models capture longer dependencies. |
| `dropout_rate` | 0.3 | 0.0–0.5 | Fraction of units randomly dropped during training. Reduces overfitting. |
| `sequence_length` | 20 | 10–50 | Number of words used as context window. Longer = more context, more memory needed. |
| `batch_size` | 64 | 16–256 | Samples per gradient update. Smaller = noisier but sometimes better generalization. |
| `epochs` | 50 | 10–200 | Training passes over the dataset. More epochs = better fit (risk of overfitting). |
| `learning_rate` | 0.001 | 1e-4–1e-2 | Step size for the Adam optimizer. |
| `temperature` (inference) | 1.0 | 0.1–2.5 | Sampling randomness. Low = predictable, high = creative. |

---

## Frontend Alternatives

The backend is a plain HTTP API. You are not required to use Streamlit. Any HTTP client can consume it.

### React (TypeScript)

```typescript
const response = await fetch("http://localhost:8000/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        seed_text: "the cat sat on",
        num_words: 50,
        temperature: 1.0,
    }),
});
const data = await response.json();
console.log(data.generated_text);
```

### Plain HTML + fetch()

```html
<!DOCTYPE html>
<html>
<body>
  <textarea id="seed"></textarea>
  <button onclick="generate()">Generate</button>
  <p id="output"></p>
  <script>
    async function generate() {
      const res = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed_text: document.getElementById("seed").value,
          num_words: 50,
          temperature: 1.0
        })
      });
      const data = await res.json();
      document.getElementById("output").textContent = data.generated_text;
    }
  </script>
</body>
</html>
```

### curl (command line)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "the cat sat on", "num_words": 50, "temperature": 1.0}'
```

### Python requests (Jupyter Notebook)

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "seed_text": "the cat sat on",
    "num_words": 50,
    "temperature": 1.0,
})
print(response.json()["generated_text"])
```

---

## Troubleshooting

### "Cannot connect to API"
- Make sure the FastAPI server is running: `uvicorn app.main:app --reload`
- Confirm the port is correct (default 8000) in the Streamlit sidebar
- Check for firewall rules blocking localhost connections

### "Model not trained yet" (503 error)
- Run `python app/train.py` from inside `rnn_app/backend/`
- Wait for training to complete before starting the API server
- If you restart the API, the model is loaded automatically from `models/`

### Training is very slow
- Reduce `epochs` in `train.py` (e.g., set to 10 for a quick test)
- Reduce `lstm_units` (e.g., 64 instead of 256)
- Use a smaller dataset for initial testing
- Enable GPU acceleration if available (PyTorch uses CUDA automatically when a GPU is detected)

### Generated text is repetitive or nonsensical
- Train for more epochs (50+ recommended)
- Use a larger or more varied training corpus
- Increase `lstm_units` or `embedding_dim`
- Try a temperature between 0.8 and 1.2 for the best balance

### Port already in use
```bash
# Find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn app.main:app --reload --port 8001
```

Then update the API URL in the Streamlit sidebar to match.

### Streamlit page not loading
- Confirm Streamlit is installed: `pip install streamlit`
- Make sure you are running from the `rnn_app/frontend/` directory
- Try `streamlit run streamlit_app.py --server.port 8502` if 8501 is taken
