"""
train.py - Jena Climate Temperature Forecasting Training Orchestrator
=====================================================================
Run this script to train the LSTM temperature forecasting model.

Usage:
    python train.py                              # Use default settings
    python train.py --epochs 30                  # Custom epochs
    python train.py --data data/jena_climate_2009_2016.csv  # Custom path
    python train.py --hidden-units 128           # More hidden units
    python train.py --help                       # Show all options

Pipeline:
  1. Load Jena Climate CSV with pandas
  2. Extract and subsample temperature column (hourly)
  3. Normalize with Min-Max scaling
  4. Create sliding-window sequences (5 days → predict next hour)
  5. Split chronologically (80% train, 20% test)
  6. Build LSTM model
  7. Train with early stopping and LR scheduling
  8. Evaluate on test set (RMSE)
  9. Save model artifacts
  10. Generate training and prediction visualizations
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.temperature_forecaster import TemperatureForecaster, DEFAULT_CONFIG


def main():
    """Main training pipeline for Jena Climate temperature forecasting."""
    parser = argparse.ArgumentParser(
        description='Train the LSTM temperature forecasting model on Jena Climate data'
    )
    parser.add_argument('--data', type=str,
                        default='data/jena_climate_2009_2016.csv',
                        help='Path to the Jena Climate CSV file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save model artifacts')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Training batch size')
    parser.add_argument('--seq-len', type=int, default=DEFAULT_CONFIG['sequence_length'],
                        help='Sequence length (past observations window)')
    parser.add_argument('--hidden-units', type=int, default=DEFAULT_CONFIG['hidden_units'],
                        help='Number of LSTM hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=DEFAULT_CONFIG['num_layers'],
                        help='Number of stacked LSTM layers')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='Adam optimizer learning rate')
    parser.add_argument('--subsample', type=int, default=DEFAULT_CONFIG['subsample_step'],
                        help='Subsample step (6=hourly from 10-min data)')

    args = parser.parse_args()

    # ── STEP 1: Initialize forecaster with config ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Initializing Temperature Forecaster")
    print("=" * 60)
    config = {
        'sequence_length': args.seq_len,
        'hidden_units': args.hidden_units,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'subsample_step': args.subsample,
    }
    forecaster = TemperatureForecaster(config=config)
    print(f"Config: {forecaster.config}")

    # ── STEP 2: Load dataset ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading Jena Climate Dataset")
    print("=" * 60)
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        print("Please download the Jena Climate dataset and place it in the data/ directory.")
        print("  Expected file: data/jena_climate_2009_2016.csv")
        sys.exit(1)

    df = forecaster.load_data(args.data)

    # ── STEP 3: Normalize temperature ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Normalizing Temperature Data (Min-Max Scaling)")
    print("=" * 60)
    normalized = forecaster.normalize(forecaster.raw_temperature)

    # ── STEP 4: Create sequences ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Creating Training Sequences (Sliding Window)")
    print("=" * 60)
    print(f"  Window size: {args.seq_len} hourly observations ({args.seq_len/24:.1f} days)")
    X, y = forecaster.create_sequences(normalized)

    # ── STEP 5: Split data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Splitting Data (Chronological)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = forecaster.split_data(X, y)

    # ── STEP 6: Build model ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Building LSTM Model")
    print("=" * 60)
    forecaster.build_model()

    # ── STEP 7: Train ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Training Model")
    print("=" * 60)
    forecaster.train(X_train, y_train, model_dir=args.model_dir)

    # ── STEP 8: Evaluate ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Evaluating on Test Set")
    print("=" * 60)
    results = forecaster.evaluate(X_test, y_test)

    # ── STEP 9: Save model ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Saving Model Artifacts")
    print("=" * 60)
    saved = forecaster.save(save_dir=args.model_dir)
    print(f"Saved: {saved}")

    # ── STEP 10: Generate visualizations ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 10: Generating Visualizations")
    print("=" * 60)
    forecaster.plot_training_history(output_dir='visualizations')
    forecaster.plot_predictions(output_dir='visualizations')

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  RMSE: {results['rmse_celsius']:.2f}°C")
    print(f"  Test samples: {results['num_samples']}")
    print(f"  Model saved to: {args.model_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Start the API server:  uvicorn app.main:app --reload")
    print(f"  2. Open Streamlit UI:     streamlit run ../frontend/streamlit_app.py")
    print(f"  3. Or test via curl:")
    print(f'       curl http://localhost:8000/model/info')


if __name__ == '__main__':
    main()
