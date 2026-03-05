"""
temperature_forecaster.py - LSTM Temperature Forecasting Module
================================================================
This module implements the TemperatureForecaster class for time-series
temperature prediction using the Jena Climate dataset.

Pipeline:
1. Data loading (pandas CSV)
2. Min-Max normalization (sklearn)
3. Sliding-window sequence creation
4. LSTM model construction (LSTM -> LSTM -> Linear)
5. Training with MSE loss, Adam, early stopping
6. Evaluation with RMSE
7. Visualization (loss curves, actual vs predicted)
8. Model persistence (save/load weights + scaler)

THEORY -> CODE GUIDE:
Each method is annotated with the corresponding mathematical formulation.
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# ─── Default Hyperparameters ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "sequence_length": 120,     # 120 hourly observations = 5 days of context
    "hidden_units":    64,      # Hidden units per LSTM layer
    "num_layers":      2,       # Number of stacked LSTM layers
    "dropout_rate":    0.2,     # Dropout between LSTM layers
    "batch_size":      256,     # Samples per gradient update
    "epochs":          50,      # Maximum training epochs
    "learning_rate":   0.001,   # Adam learning rate
    "validation_split": 0.1,   # Fraction of training data for validation
    "subsample_step":  6,      # Take every Nth row (6 = hourly from 10-min data)
}


# ─── PyTorch Model ───────────────────────────────────────────────────────────

class TemperatureLSTM(nn.Module):
    """
    Stacked LSTM for temperature regression (time-series forecasting).

    THEORY -> CODE MAPPING:
    ─────────────────────────────────────────────────────────────────────
    Unlike text generation (discrete tokens → embedding → classification),
    temperature forecasting uses continuous values directly:

    Input: (batch, sequence_length, 1)
        A window of past temperature readings (already normalized to [0,1]).
        No embedding layer needed — values are already continuous floats.

    LSTM layers (stacked):
        Each LSTM cell maintains a hidden state h_t and cell state c_t:
            f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
            i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
            ĉ_t = tanh(W_c · [h_{t-1}, x_t] + b_c) # Candidate cell state
            c_t = f_t * c_{t-1} + i_t * ĉ_t         # New cell state
            o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # Output gate
            h_t = o_t * tanh(c_t)                     # Hidden state output

        Stacking two LSTM layers lets the network learn hierarchical
        temporal patterns — short-term fluctuations (layer 1) and
        longer-term trends (layer 2).

    Output (Linear):
        Projects the final hidden state h_T to a single scalar value:
        ŷ = W · h_T + b
        This is the predicted (normalized) temperature for the next timestep.

    Loss: MSE (Mean Squared Error) — standard for regression tasks.
        L = (1/N) Σ (y_i - ŷ_i)²
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, input_size: int = 1, hidden_units: int = 64,
                 num_layers: int = 2, dropout_rate: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, sequence_length, 1) — normalized temperature values

        Returns:
            (batch, 1) — predicted next temperature (normalized)
        """
        # LSTM processes the full sequence
        # out: (batch, seq_len, hidden_units) — all hidden states
        # h: (num_layers, batch, hidden_units) — final hidden state per layer
        out, (h, c) = self.lstm(x)

        # Take the final hidden state from the last LSTM layer
        # h[-1] shape: (batch, hidden_units)
        final_hidden = self.dropout(h[-1])

        # Project to single temperature prediction
        return self.fc(final_hidden)  # (batch, 1)


# ─── TemperatureForecaster ───────────────────────────────────────────────────

class TemperatureForecaster:
    """
    LSTM-based temperature forecasting system for the Jena Climate dataset.

    THEORY BACKGROUND:
    -----------------
    Time-series forecasting learns the mapping:
        ŷ_{t+1} = f(x_{t}, x_{t-1}, ..., x_{t-n+1})

    Given a window of n past temperature observations, the model predicts
    the next temperature value. This is a regression task (continuous output),
    unlike text generation which is classification (discrete word prediction).

    Key differences from text generation:
    1. No embedding layer — input is already continuous
    2. MSE loss instead of CrossEntropyLoss
    3. Output is a single scalar, not a probability distribution
    4. Data is normalized with Min-Max scaling, not tokenized
    5. Train/test split is chronological (no shuffling)
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.training_history = None
        self.test_results = None  # Stores actual vs predicted for API

        self.sequence_length = self.config["sequence_length"]
        self.hidden_units = self.config["hidden_units"]
        self.num_layers = self.config["num_layers"]
        self.dropout_rate = self.config["dropout_rate"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    # ─── PHASE 1: DATA LOADING & PREPROCESSING ──────────────────────────────

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load the Jena Climate dataset and extract the temperature column.

        THEORY: The Jena Climate dataset contains 14 weather features recorded
        every 10 minutes from 2009 to 2016 (~420k observations). We focus on
        the 'T (degC)' column for univariate temperature forecasting.

        Subsampling: The raw 10-minute intervals create an extremely large
        dataset. We subsample every 6th row to get hourly data (~70k points),
        which is still sufficient for learning temperature patterns while being
        computationally tractable.
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Raw dataset shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['Date Time'].iloc[0]} to {df['Date Time'].iloc[-1]}")

        # Subsample to hourly
        step = self.config["subsample_step"]
        df = df.iloc[::step].reset_index(drop=True)
        print(f"  After subsampling (every {step}th row): {df.shape}")

        # Extract temperature
        temperature = df['T (degC)'].values
        print(f"  Temperature range: {temperature.min():.2f}°C to {temperature.max():.2f}°C")
        print(f"  Temperature mean:  {temperature.mean():.2f}°C")

        self.raw_dates = df['Date Time'].values
        self.raw_temperature = temperature
        return df

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Min-Max normalization to scale temperature values to [0, 1].

        THEORY: Neural networks learn more efficiently when input values
        are in a small, consistent range. Min-Max scaling transforms:
            x_normalized = (x - x_min) / (x_max - x_min)

        This maps the minimum value to 0 and maximum to 1. The scaler
        object stores x_min and x_max so we can inverse-transform
        predictions back to actual temperature (°C).

        Why Min-Max over Standard scaling?
        - Temperature has natural bounds (physical constraints)
        - Min-Max preserves the original distribution shape
        - Output is bounded [0,1], matching common activation ranges
        """
        data_reshaped = data.reshape(-1, 1)
        normalized = self.scaler.fit_transform(data_reshaped).flatten()

        print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        print(f"  Scaler min={self.scaler.data_min_[0]:.2f}°C, "
              f"max={self.scaler.data_max_[0]:.2f}°C")
        return normalized

    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """Convert normalized values back to actual temperature (°C)."""
        return self.scaler.inverse_transform(
            normalized_data.reshape(-1, 1)
        ).flatten()

    def create_sequences(self, data: np.ndarray):
        """
        Create input-output pairs using a sliding window approach.

        THEORY:
        Given a time series [t₁, t₂, t₃, ..., tₙ] and window size w:
            X[0] = [t₁, t₂, ..., t_w]      →  y[0] = t_{w+1}
            X[1] = [t₂, t₃, ..., t_{w+1}]  →  y[1] = t_{w+2}
            ...

        Each input X[i] is a sequence of w consecutive temperature readings.
        The target y[i] is the temperature immediately following the window.
        The window slides one step at a time, creating overlapping sequences.

        With sequence_length=120 (hourly data), each input represents 5 days
        of past temperature, and the target is the next hour's temperature.
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i])

        X = np.array(X)
        y = np.array(y)

        # Reshape X to (samples, sequence_length, 1) for LSTM input
        X = X.reshape(X.shape[0], X.shape[1], 1)

        print(f"  Created {len(X)} sequences")
        print(f"  X shape: {X.shape}  (samples, sequence_length, features)")
        print(f"  y shape: {y.shape}  (samples,)")
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
        """
        Split data chronologically into training and test sets.

        THEORY: Time-series data MUST be split chronologically, not randomly.
        Random shuffling would leak future information into the training set,
        making the model appear better than it actually is (data leakage).

        The latest data goes to the test set because we want to evaluate
        how well the model predicts future (unseen) temperatures.

        We further split the training set to hold out validation data
        for early stopping and hyperparameter monitoring.
        """
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples:     {len(X_test)}")
        return X_train, X_test, y_train, y_test

    # ─── PHASE 2: MODEL CONSTRUCTION ─────────────────────────────────────────

    def build_model(self) -> TemperatureLSTM:
        """Construct the LSTM model and move it to the compute device."""
        self.model = TemperatureLSTM(
            input_size=1,
            hidden_units=self.hidden_units,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        print(f"\nModel architecture:\n{self.model}")
        print(f"\nTotal parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return self.model

    # ─── PHASE 3: TRAINING ───────────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              model_dir: str = 'models') -> dict:
        """
        Train the LSTM model with a manual PyTorch training loop.

        THEORY: Training minimises the MSE loss via gradient descent.
        For each batch:
          1. Forward pass: ŷ = model(X_batch)
          2. Loss:  L = MSE(ŷ, y_batch) = (1/N) Σ (y_i - ŷ_i)²
          3. Backward (BPTT): compute ∂L/∂W for all weights W
          4. Gradient clipping: ||g|| ≤ 1.0 (prevents exploding gradients)
          5. Weight update: W ← W - α · ∂L/∂W  (Adam adaptive step)

        MSE is preferred over CrossEntropyLoss because this is a regression
        task (predicting a continuous value, not a class).

        Callbacks:
          - Best-model saving: save weights when val_loss improves
          - Early stopping: stop if no improvement for 10 epochs
          - ReduceLROnPlateau: halve learning rate on val_loss plateau
        """
        os.makedirs(model_dir, exist_ok=True)
        best_path = os.path.join(model_dir, 'best_model.pt')

        # ── Train / Validation split ─────────────────────────────────────────
        val_split = self.config['validation_split']
        split_idx = int(len(X_train) * (1 - val_split))
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        # ── DataLoaders ──────────────────────────────────────────────────────
        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        train_dl = DataLoader(train_ds, batch_size=self.config['batch_size'],
                              shuffle=False)  # No shuffle for time-series
        val_dl = DataLoader(val_ds, batch_size=self.config['batch_size'])

        # ── Optimizer, loss, scheduler ───────────────────────────────────────
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6
        )

        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting training: {self.config['epochs']} max epochs, "
              f"batch_size={self.config['batch_size']}, "
              f"validation_split={val_split}")

        for epoch in range(self.config['epochs']):

            # ── Training pass ────────────────────────────────────────────────
            self.model.train()
            t_loss, t_total = 0.0, 0
            for X_b, y_b in train_dl:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device).unsqueeze(1)  # (batch, 1)

                optimizer.zero_grad()
                predictions = self.model(X_b)  # (batch, 1)
                loss = criterion(predictions, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                t_loss += loss.item() * len(y_b)
                t_total += len(y_b)

            train_loss = t_loss / t_total

            # ── Validation pass ───────────────────────────────────────────────
            self.model.eval()
            v_loss, v_total = 0.0, 0
            with torch.no_grad():
                for X_b, y_b in val_dl:
                    X_b = X_b.to(self.device)
                    y_b = y_b.to(self.device).unsqueeze(1)

                    predictions = self.model(X_b)
                    loss = criterion(predictions, y_b)
                    v_loss += loss.item() * len(y_b)
                    v_total += len(y_b)

            val_loss = v_loss / v_total

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1:3d}/{self.config['epochs']}  "
                  f"loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

            scheduler.step(val_loss)

            # ── Save best / early stopping ────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_path)
                print(f"  -> val_loss improved, saved best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(no improvement for 10 epochs)")
                    break

        # Restore best weights
        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device, weights_only=True)
        )
        self.training_history = history

        # Save training history
        history_path = os.path.join(model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in history.items()},
                f, indent=2
            )
        print(f"Training history saved to {history_path}")
        return history

    # ─── PHASE 4: EVALUATION ────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on the test set and compute RMSE.

        THEORY:
        RMSE (Root Mean Squared Error) is the standard metric for regression:
            RMSE = √( (1/N) Σ (y_i - ŷ_i)² )

        RMSE is in the same units as the target variable (°C), making it
        directly interpretable: "On average, predictions are off by X°C."

        We compute RMSE on both normalized and denormalized (actual °C) values.
        The denormalized RMSE is the meaningful metric for reporting.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained or loaded first.")

        self.model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions_norm = self.model(X_tensor).cpu().numpy().flatten()

        # Denormalize to actual °C
        actual_temp = self.inverse_transform(y_test)
        predicted_temp = self.inverse_transform(predictions_norm)

        # Compute RMSE
        rmse_normalized = np.sqrt(np.mean((y_test - predictions_norm) ** 2))
        rmse_celsius = np.sqrt(np.mean((actual_temp - predicted_temp) ** 2))

        self.test_results = {
            'actual': actual_temp.tolist(),
            'predicted': predicted_temp.tolist(),
            'rmse_normalized': float(rmse_normalized),
            'rmse_celsius': float(rmse_celsius),
            'num_samples': len(y_test),
        }

        print(f"\nTest Set Evaluation:")
        print(f"  RMSE (normalized): {rmse_normalized:.6f}")
        print(f"  RMSE (°C):         {rmse_celsius:.2f}°C")
        print(f"  Test samples:      {len(y_test)}")

        return self.test_results

    def predict(self, input_sequence: np.ndarray) -> float:
        """
        Predict the next temperature given a sequence of past values.

        Args:
            input_sequence: array of normalized temperature values,
                            shape (sequence_length,)

        Returns:
            Predicted temperature in °C (denormalized).
        """
        if self.model is None:
            raise RuntimeError("Model must be trained or loaded first.")

        self.model.eval()
        # Reshape to (1, sequence_length, 1)
        x = torch.tensor(
            input_sequence.reshape(1, -1, 1), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy().flatten()[0]

        return float(self.inverse_transform(np.array([pred_norm]))[0])

    # ─── PHASE 5: MODEL PERSISTENCE ──────────────────────────────────────────

    def save(self, save_dir: str = 'models') -> dict:
        """Save model weights, scaler, config, and test results to disk."""
        os.makedirs(save_dir, exist_ok=True)

        # Save model weights
        model_path = os.path.join(save_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)

        # Save scaler (needed for inverse transforms in production)
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save config
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Save test results for the API
        if self.test_results:
            results_path = os.path.join(save_dir, 'test_results.json')
            with open(results_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)

        print(f"Model saved to {save_dir}/")
        return {'model': model_path, 'scaler': scaler_path, 'config': config_path}

    def load(self, save_dir: str = 'models') -> None:
        """Load a previously saved model, scaler, and configuration."""
        # Load config
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path) as f:
            saved_config = json.load(f)
        self.config.update(saved_config)
        self.sequence_length = self.config['sequence_length']
        self.hidden_units = self.config['hidden_units']
        self.num_layers = self.config['num_layers']
        self.dropout_rate = self.config['dropout_rate']

        # Load scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load model
        model_path = os.path.join(save_dir, 'model.pt')
        self.model = TemperatureLSTM(
            input_size=1,
            hidden_units=self.hidden_units,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Load test results if available
        results_path = os.path.join(save_dir, 'test_results.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                self.test_results = json.load(f)

        # Load training history if available
        history_path = os.path.join(save_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                self.training_history = json.load(f)

        print(f"Model loaded from {save_dir}/")

    # ─── PHASE 6: VISUALIZATION ──────────────────────────────────────────────

    def plot_training_history(self, output_dir: str = 'visualizations') -> str:
        """
        Plot training and validation loss over epochs.

        THEORY: Monitoring loss curves reveals:
        - Underfitting: both losses are high and not decreasing
        - Overfitting: training loss decreases but val_loss increases
        - Good fit: both losses decrease and converge
        """
        if not self.training_history:
            raise RuntimeError("No training history. Train the model first.")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_history.png')

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#2e2e3e')

        epochs = range(1, len(self.training_history['loss']) + 1)
        ax.plot(epochs, self.training_history['loss'],
                label='Training Loss', color='#7c6af5', linewidth=2)
        ax.plot(epochs, self.training_history['val_loss'],
                label='Validation Loss', color='#f56a6a', linewidth=2, linestyle='--')

        ax.set_title('LSTM Training Loss (MSE)', fontsize=14,
                     fontweight='bold', color='white')
        ax.set_xlabel('Epoch', color='white')
        ax.set_ylabel('Mean Squared Error', color='white')
        ax.legend(facecolor='#2e2e3e', labelcolor='white')
        ax.grid(alpha=0.3, color='#555577')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#555577')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e2e')
        plt.close()
        print(f"Training plot saved to {output_path}")
        return output_path

    def plot_predictions(self, output_dir: str = 'visualizations',
                         num_points: int = 500) -> str:
        """
        Plot actual vs predicted temperatures on the test set.

        Shows a subset of predictions overlaid on actual temperatures
        to visually assess model performance.
        """
        if not self.test_results:
            raise RuntimeError("No test results. Evaluate the model first.")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'predictions.png')

        actual = np.array(self.test_results['actual'][:num_points])
        predicted = np.array(self.test_results['predicted'][:num_points])

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#2e2e3e')

        ax.plot(actual, label='Actual Temperature', color='#6af5c8',
                linewidth=1.5, alpha=0.8)
        ax.plot(predicted, label='Predicted Temperature', color='#7c6af5',
                linewidth=1.5, alpha=0.8)

        rmse = self.test_results['rmse_celsius']
        ax.set_title(f'Actual vs Predicted Temperature (RMSE: {rmse:.2f}°C)',
                     fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel('Time Step (hours)', color='white')
        ax.set_ylabel('Temperature (°C)', color='white')
        ax.legend(facecolor='#2e2e3e', labelcolor='white')
        ax.grid(alpha=0.3, color='#555577')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#555577')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e2e')
        plt.close()
        print(f"Prediction plot saved to {output_path}")
        return output_path

    def get_model_info(self) -> dict:
        """Return model statistics as a dict (for the API /model/info endpoint)."""
        if self.model is None:
            return {"status": "not_loaded"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)

        layers = []
        for name, module in self.model.named_children():
            param_count = sum(p.numel() for p in module.parameters())
            layers.append({
                "name": name,
                "type": type(module).__name__,
                "parameters": param_count,
            })

        info = {
            "status": "loaded",
            "sequence_length": self.sequence_length,
            "hidden_units": self.hidden_units,
            "num_layers": self.num_layers,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layers": layers,
            "config": self.config,
        }

        if self.test_results:
            info["rmse_celsius"] = self.test_results['rmse_celsius']

        return info
