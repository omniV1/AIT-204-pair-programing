# RNN Text Generator — Backend

A FastAPI backend that trains and serves an LSTM language model for character/word-level text generation. Built for the AIT-204 Deep Learning activity on Recurrent Neural Networks.

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
# Train using the included Alice in Wonderland sample text
python app/train.py

# Train with a custom text file
python app/train.py --text data/my_book.txt

# Download a full book from Project Gutenberg and train
python app/train.py --download https://www.gutenberg.org/files/11/11-0.txt

# Quick training run for testing (fewer epochs)
python app/train.py --epochs 20 --batch-size 64
```

Training artifacts are saved to `models/`:
- `model.pt` — PyTorch model weights (load with `model.load_state_dict(torch.load(...))`)
- `tokenizer.pkl` — Word-to-index mapping
- `config.json` — Hyperparameters used during training
- `training_history.json` — Loss/accuracy per epoch

### 3. Start the API server

```bash
# Development (auto-reload on file changes)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4. Test generation

```bash
# Using curl
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "alice was beginning", "num_words": 50, "temperature": 1.0}'

# Open the interactive docs
open http://localhost:8000/docs
```

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py          # Package marker
│   ├── text_generator.py    # Core ML module (TextGenerator class)
│   ├── train.py             # Training script (run this to train)
│   └── main.py              # FastAPI application (run this to serve)
├── data/
│   └── training_text.txt    # Sample training data (Alice in Wonderland)
├── models/                  # Auto-created after training
│   ├── model.pt
│   ├── tokenizer.pkl
│   ├── config.json
│   └── training_history.json
├── visualizations/          # Auto-created after training
│   └── training_history.png
├── requirements.txt
└── README.md
```

---

## Architecture Overview

The model uses a stacked LSTM architecture:

```
Input (sequence of word IDs)
        |
  [Embedding Layer]    — vocab_size x embedding_dim lookup table
        |              — converts integer IDs to dense float vectors
  [LSTM Layer 1]       — processes sequence, returns all hidden states
        |              — units=150, return_sequences=True
  [Dropout 20%]        — regularization
        |
  [LSTM Layer 2]       — deeper representation, returns final hidden state
        |              — units=150, return_sequences=False
  [Dropout 20%]        — regularization
        |
  [Dense + Softmax]    — projects to vocab_size probabilities
        |
Output (probability distribution over vocabulary)
```

**Training objective:** Minimize categorical cross-entropy between predicted and actual next word.

**Generation:** Auto-regressive sampling — each predicted word becomes part of the next input context.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check, model status |
| GET | `/health` | Lightweight health check |
| POST | `/generate` | Generate text from seed |
| POST | `/generate/batch` | Generate multiple variations |
| GET | `/model/info` | Model architecture and stats |
| GET | `/model/architecture-image` | Base64 PNG architecture diagram |
| GET | `/training/history` | Loss/accuracy history (JSON) |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc documentation |

### POST /generate

Request body:
```json
{
  "seed_text": "alice was beginning to",
  "num_words": 50,
  "temperature": 1.0
}
```

Response:
```json
{
  "seed_text": "alice was beginning to",
  "generated_text": "alice was beginning to feel very tired of sitting ...",
  "num_words_requested": 50,
  "temperature": 1.0,
  "model_info": {
    "vocab_size": 2847,
    "sequence_length": 50
  }
}
```

### POST /generate/batch

Generates multiple independent samples from the same seed:
```
POST /generate/batch?variations=5
```

---

## Hyperparameter Tuning Guide

### sequence_length (default: 50)
Controls how many words the model sees as context when predicting the next word.

| Value | Effect |
|-------|--------|
| 20-30 | Short context, trains faster, less coherent output |
| 50 | Balanced — good starting point |
| 80-100 | Long context, trains slower, more thematically consistent |

### embedding_dim (default: 100)
Dimensionality of the word vector space.

| Value | Effect |
|-------|--------|
| 50 | Compact, faster, less expressive |
| 100 | Good balance for small-medium vocabularies |
| 200-300 | Better for large vocabularies (10k+ words) |

### lstm_units (default: 150)
Number of memory cells in each LSTM layer — determines model capacity.

| Value | Effect |
|-------|--------|
| 64-128 | Small, fast, may underfit on complex text |
| 150-256 | Good for most use cases |
| 512+ | Large model, needs substantial training data |

### dropout_rate (default: 0.2)
Fraction of neurons randomly disabled during training to prevent overfitting.

| Value | Effect |
|-------|--------|
| 0.1 | Light regularization |
| 0.2-0.3 | Standard regularization |
| 0.5 | Heavy regularization (use if overfitting badly) |

### temperature (generation-time parameter)
Controls randomness of text generation. Not a training hyperparameter.

| Value | Effect |
|-------|--------|
| 0.5 | Conservative — high probability words are heavily favored |
| 1.0 | Balanced — samples from the model's true distribution |
| 1.5 | Creative — more varied and surprising output |
| 2.0+ | Very random — often incoherent |

### Example: faster training for experiments

```bash
python app/train.py \
  --epochs 30 \
  --batch-size 256 \
  --seq-len 30 \
  --lstm-units 128 \
  --embed-dim 64
```

---

## Getting Better Training Data

The quality of generated text is directly proportional to training data size and quality. A minimum of 100,000 words is recommended for coherent generation.

### Project Gutenberg (free, public domain)

```bash
# Alice in Wonderland (~26,000 words)
python app/train.py --download https://www.gutenberg.org/files/11/11-0.txt

# Pride and Prejudice (~122,000 words)
python app/train.py --download https://www.gutenberg.org/files/1342/1342-0.txt

# Sherlock Holmes — The Adventures (~100,000 words)
python app/train.py --download https://www.gutenberg.org/files/1661/1661-0.txt

# Frankenstein (~78,000 words)
python app/train.py --download https://www.gutenberg.org/files/84/84-0.txt

# Moby Dick (~220,000 words) — great for longer training
python app/train.py --download https://www.gutenberg.org/files/2701/2701-0.txt
```

### Combining multiple texts

```bash
# Manually concatenate files, then train
cat data/alice.txt data/pride.txt data/sherlock.txt > data/combined.txt
python app/train.py --text data/combined.txt
```

---

## Troubleshooting

### "No trained model found" on server startup

The model has not been trained yet. Run the training script first:
```bash
cd backend
python app/train.py
```

### Training is very slow

- Reduce `--epochs` for a quick test: `python app/train.py --epochs 10`
- Reduce `--lstm-units`: `--lstm-units 64`
- Reduce `--seq-len`: `--seq-len 20`
- Enable GPU: PyTorch detects CUDA automatically; on Apple Silicon install `torch` with MPS support

### Out of Memory during training

- Reduce `--batch-size`: try 32 or 64
- Reduce `--lstm-units`: try 64 or 128
- Use a smaller text file

### "Very few training sequences" warning

Your text file is too short. The model needs at least:
  - sequence_length + 1 tokens to create even one training example
  - Recommended minimum: 10,000 tokens (roughly 7,500 words)

### Generated text is repetitive or incoherent

- **Repetitive:** Increase temperature (try 1.2–1.5)
- **Incoherent:** Decrease temperature (try 0.7–0.9)
- **Both:** Model needs more training data or more epochs
- Check that training loss was actually decreasing (view `visualizations/training_history.png`)

### ImportError or ModuleNotFoundError

Make sure you are running from the `backend/` directory with the virtual environment activated:
```bash
cd rnn_app/backend
source venv/bin/activate
python app/train.py
```

### CORS errors in the frontend

The API allows all origins by default (`allow_origins=["*"]`). If you still see CORS errors, check that the frontend is pointing to the correct API URL and port.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models` | Directory containing trained model artifacts |

Set via shell:
```bash
export MODEL_DIR=/path/to/your/models
uvicorn app.main:app --reload
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | LSTM model (nn.Embedding, nn.LSTM, nn.Linear) |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server for FastAPI |
| `pydantic` | Request/response data validation |
| `numpy` | Numerical array operations |
| `matplotlib` | Training history plots |
| `python-dotenv` | Environment variable loading |
