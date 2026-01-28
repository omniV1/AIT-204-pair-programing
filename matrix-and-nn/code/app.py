import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from multilayer_perceptron import MultiLayerPerceptron
import kagglehub
from kagglehub import KaggleDatasetAdapter

# -----------------------------
# Load dataset
# -----------------------------
file_path = "all_seasons.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "justinas/nba-players-data",
    file_path
)

# -----------------------------
# Select 100 players from 5-year window
# -----------------------------
df["season_start_year"] = df["season"].str.split("-").str[0].astype(int)
start_year = 1996
end_year = 2000
df_window = df[(df["season_start_year"] >= start_year) & (df["season_start_year"] <= end_year)]
df_pool = df_window.sample(100, random_state=42).reset_index(drop=True)
df = df_pool

# -----------------------------
# DATA CLEANING & PREPARATION
# -----------------------------
features = ["age", "player_height", "player_weight", "draft_round", "player_name", "team_abbreviation"]
df_clean = df[features].copy()

df_clean["draft_round"] = pd.to_numeric(df_clean["draft_round"], errors="coerce")
df_clean["draft_round"] = df_clean["draft_round"].fillna(df_clean["draft_round"].max() + 1)

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df_clean["height_norm"] = normalize(df_clean["player_height"])
df_clean["weight_norm"] = normalize(df_clean["player_weight"])
PRIME_AGE = 27
df_clean["age_prime_score"] = 1 - abs(df_clean["age"] - PRIME_AGE) / df_clean["age"].max()
df_clean["age_prime_score"] = df_clean["age_prime_score"].clip(0, 1)
df_clean["draft_score"] = 1 - normalize(df_clean["draft_round"])

# Player suitability
df_clean["suitability_score"] = (
    0.35 * df_clean["height_norm"] +
    0.25 * df_clean["weight_norm"] +
    0.25 * df_clean["age_prime_score"] +
    0.15 * df_clean["draft_score"]
)

# Weak supervision labels
TOP_PERCENT = 0.25
threshold = df_clean["suitability_score"].quantile(1 - TOP_PERCENT)
df_clean["optimal_label"] = (df_clean["suitability_score"] >= threshold).astype(int)

# Features for ANN
feature_cols = ["age", "player_height", "player_weight", "draft_round"]
X = df_clean[feature_cols].values
y = df_clean["optimal_label"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding
y_onehot = np.zeros((y.size, 2))
y_onehot[np.arange(y.size), y] = 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Streamlit app
# =========================
def main():
    st.title("Artificial Neural Networks")
    st.write("""
    Demonstration of a multilayer perceptron (MLP) trained on NBA player stats.
    """)

    st.sidebar.header("Data & Model Settings")
    test_size = st.sidebar.slider("Test Split %", 10, 50, 20, 5) / 100
    random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
        value=0.01
    )
    n_iterations = st.sidebar.slider("Iterations", 100, 2000, 500, 100)

    st.header("Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(X))
    with col2:
        st.metric("Test Split", f"{test_size*100:.0f}%")

    # Train button
    if st.button("Train Model", type="primary"):
        input_size = X_train.shape[1]
        hidden_size = 8  # single hidden layer
        output_size = 2

        mlp_model = MultiLayerPerceptron(input_size, hidden_size, output_size)
        st.subheader("Training Progress")
        mlp_model.train(X_train, y_train, epochs=n_iterations, learning_rate=learning_rate)
        st.success("Training complete!")

        # Predictions
        y_pred_train = mlp_model.predict(X_train)
        y_pred_test = mlp_model.predict(X_test)
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        train_acc = np.mean(y_pred_train == y_train_labels)
        test_acc = np.mean(y_pred_test == y_test_labels)

        st.header("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
        with col2:
            st.metric("Test Accuracy", f"{test_acc*100:.2f}%")

        # Top 5 players
        y_prob = mlp_model.forward(X_train)[:, 1]
        top5_idx = np.argsort(y_prob)[-5:][::-1]
        st.header("Optimal Team (Top 5 Players)")
        st.dataframe(df_clean.iloc[top5_idx][["player_name", "team_abbreviation", "age", "player_height", "player_weight"]])

if __name__ == "__main__":
    main()
