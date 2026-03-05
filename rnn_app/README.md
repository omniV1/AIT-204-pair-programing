# RNN Applications — AIT-204

This directory contains two LSTM-based applications built with PyTorch, FastAPI, and Streamlit.

## Projects

### 1. [Text Generator](text_generator/)
LSTM-based text generation trained on literary text (Alice in Wonderland). Generates creative text continuations from seed phrases using temperature-controlled sampling.

- **Model:** 2-layer LSTM with word embeddings
- **Task:** Next-word prediction (classification)
- **Loss:** CrossEntropyLoss

### 2. [Temperature Forecaster](temperature_forecaster/)
LSTM-based temperature forecasting using the Jena Climate dataset (2009-2016). Predicts future temperature from 5 days of historical hourly observations.

- **Model:** 2-layer LSTM for time-series regression
- **Task:** Next-hour temperature prediction (regression)
- **Loss:** Mean Squared Error (MSE)
- **Metric:** RMSE (°C)

## Quick Start

Each project has its own backend and frontend. See the individual READMEs for setup instructions.

```bash
# Text Generator
cd rnn_app/text_generator/backend
pip install -r requirements.txt
python app/train.py
uvicorn app.main:app --reload --port 8000

# Temperature Forecaster
cd rnn_app/temperature_forecaster/backend
pip install -r requirements.txt
python app/train.py
uvicorn app.main:app --reload --port 8001
```

## Documentation

See `temperature_forecaster/docs/lstm_rnn_guide.html` for an educational guide on LSTM/RNN concepts, time-series forecasting, and the Jena Climate dataset.

## Directory Structure

```
rnn_app/
├── text_generator/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── text_generator.py
│   │   │   ├── train.py
│   │   │   └── main.py
│   │   ├── data/
│   │   ├── models/
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   └── README.md
├── temperature_forecaster/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── temperature_forecaster.py
│   │   │   ├── train.py
│   │   │   └── main.py
│   │   ├── data/
│   │   │   └── jena_climate_2009_2016.csv
│   │   ├── models/
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   ├── docs/
│   │   └── lstm_rnn_guide.html
│   └── README.md
└── README.md
```
