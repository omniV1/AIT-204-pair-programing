import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from multilayer_perceptron import MultiLayerPerceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "all_seasons.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "justinas/nba-players-data",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

# -----------------------------
# Select 100 players from 5-year window
# -----------------------------

# Convert the season to starting year as integer
df["season_start_year"] = df["season"].str.split("-").str[0].astype(int)

# Define 5-year window
start_year = 1996
end_year = 2000

# Filter dataset to 5-year window
df_window = df[(df["season_start_year"] >= start_year) & (df["season_start_year"] <= end_year)]

# Randomly select 100 players
df_pool = df_window.sample(100, random_state=42).reset_index(drop=True)

# Use df_pool for all further processing
df = df_pool

# -----------------------------
# DATA CLEANING & PREPARATION
# -----------------------------

# Keep only rows with required numeric data
# Keep numeric features AND name/abbreviation for display
features = ["age", "player_height", "player_weight", "draft_round", "player_name", "team_abbreviation"]
df_clean = df[features].copy()


# Handle draft_round
df_clean["draft_round"] = pd.to_numeric(df_clean["draft_round"], errors="coerce")
df_clean["draft_round"] = df_clean["draft_round"].fillna(df_clean["draft_round"].max() + 1)

# Normalization
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


# -----------------------------
# FINAL DATASET FOR ANN
# -----------------------------

X = df_clean[
    ["height_norm", "weight_norm", "age_prime_score", "draft_score"]
].values

y = df_clean["optimal_label"].values

print("Class distribution:")
print(df_clean["optimal_label"].value_counts())

df_clean.head()


# ==========================================
# SIDEBAR - DATA SETTINGS
# ==========================================
st.sidebar.header("1. Data Settings")
test_size = st.sidebar.slider("Test Split %", 10, 50, 20, 5) / 100
random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

# ==========================================
# SIDEBAR - MODEL SETTINGS
# ==========================================
st.sidebar.header("2. Model Settings")
learning_rate = st.sidebar.select_slider(
"Learning Rate",
options=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
value=0.01
)
n_iterations = st.sidebar.slider("Iterations", 100, 2000, 500, 100)



# ==========================================
# PREPARE FEATURES AND LABELS
# ==========================================
# Select features you want to use (physical + draft info)
feature_cols = ["age", "player_height", "player_weight", "draft_round"]
X = df_clean[feature_cols].values
y = df_clean["optimal_label"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert labels to one-hot encoding for softmax
y_onehot = np.zeros((y.size, 2))
y_onehot[np.arange(y.size), y] = 1

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=test_size, random_state=int(42), stratify=y
)


def main():
    # ==========================================
    # HOME SCREEN / WELCOME
    # ==========================================
    st.title("Artificial Neural Networks")
    st.write("""
    Welcome to the Artificial Neural Network app! This application demonstrates
    an artificial neural network trained on a database of basketball player stats.
    """)

    st.info("""
    **Getting Started:**
    1. Set the train test split
    2. Set the learning rate of the model
    3. Set the iteration count of the training
    4. Set the gradient descent mode of the model
    5. Click run model to see the output results!
    """)

    # ==========================================
    # LOAD AND VALIDATE DATA
    # ==========================================

    st.header("1. Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(X))
        #st.metric("Training Samples", len(X_train))
    with col2:
        #st.metric("Test Samples", len(X_test))
        st.metric("Test Split", f"{test_size*100:.0f}%")
    

    # Train button
    if st.button("Train Model", type="primary"):
        # ==========================================
        # INITIALIZE AND TRAIN THE MLP
        # ==========================================
        input_size = X_train.shape[1]
        hidden_size = 8  # tune this
        output_size = 2  # binary classification

        mlp_model = MultiLayerPerceptron(input_size, hidden_size, output_size)

        st.subheader("Training Progress")
        mlp_model.train(X_train, y_train, epochs=n_iterations, learning_rate=learning_rate)

        st.success("Training complete!")


        # Predictions
        y_pred_train = mlp_model.predict(X_train)
        y_pred_test = mlp_model.predict(X_test)

        # Convert back to single class labels
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        # Accuracy metrics
        train_acc = np.mean(y_pred_train == y_train_labels)
        test_acc = np.mean(y_pred_test == y_test_labels)

        st.header("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
        with col2:
            st.metric("Test Accuracy", f"{test_acc*100:.2f}%")


        # Use predicted probabilities for ranking
        y_prob = mlp_model.forward(X_train)[:, 1]  # probability of class 1
        top5_idx = np.argsort(y_prob)[-5:][::-1]   # indices of top 5

        st.header("Optimal Team (Top 5 Players)")
        st.dataframe(df_clean.iloc[top5_idx][["player_name", "team_abbreviation", "age", "player_height", "player_weight"]])

           
if __name__ == "__main__":
    main()


