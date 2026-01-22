"""

Owen Lindsey & Tyler Friesen\

Professor Artzi

AIT-204 

01/21/2026

Linear Regresssion 

This application was created by us following this starter code: https://padlet.com/isac_artzi/ait-204-t1-background-math-and-gradient-based-learning-yqjc6gj44bkjpzbx/wish/9kmlZVr4AXwNZpgV

Synthetic data was generated using the following code: https://github.com/isac-artzi/AIT-204-Topic1.git

""" 

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="Linear Regression", 
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Sidebar
# ===============================

st.sidebar.title("Linear Regression")

### ===============================
### Helper Functions
### ===============================

def load_csv_from_upload(uploaded_file):
    """
    Load data from an uploaded CSV file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        X: Feature matrix (n_samples, 1)
        y: Target vector (n_samples)
        y_true: True values without noise (or y if y_true not present)
    """
    df = pd.read_csv(uploaded_file)

    # Check required columns
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("CSV must contain 'x' and 'y' columns")

    X = df["x"].values.reshape(-1, 1)
    y = df["y"].values

    # y_true is optional - use y if not present
    if "y_true" in df.columns:
        y_true = df["y_true"].values
    else:
        y_true = y.copy()

    return X, y, y_true

def load_csv_data(file_path):
    """ 
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file to load.

    Returns:
        X: Feature matrix (n_samples, 1)
        y: Target vector (n_samples) 
        y_true: True values without noise
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    X = df["x"].values.reshape(-1, 1)
    y = df["y"].values
    y_true = df["y_true"].values
    return X, y, y_true

### ===============================
### Training / Testing Split
### ===============================

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Args:
        X: Feature matrix (n_samples, 1)
        y: Target vector (n_samples)
        test_size (float): The proportion of the data to include in the testing set.
        random_state (int): The random seed to use for the split.

    Returns:
        X_train, y_train, X_test, y_test
    """
    np.random.seed(random_state)  # Set seed for reproducibility
    indices = np.random.permutation(len(X))  # Store the shuffled indices
    
    split_index = int(len(X) * (1 - test_size))
    
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test



### ===============================
### Linear Regression Model
### ===============================

class LinearRegression:
    """
    Linear Regression model.

    Attributes:
        learning_rate (float): The learning rate for the gradient descent.
        n_iterations (int): The number of iterations to run the gradient descent.
        weights (np.ndarray): The weights of the model.
        bias (float): The bias of the model.
        losses (list): The losses of the model.
        weights_history (list): The history of the weights.
        gradients_history (list): The history of the gradients.

    Methods:
        fit(X, y, batch_size=None): Fit the linear regression model to the training data.
        predict(X): Predict the target values for the given features.
        get_losses(): Get the losses of the model.
        get_weights(): Get the weights of the model.
        get_bias(): Get the bias of the model.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = [] 
        self.weights_history = []
        self.gradients_history = []

    def fit(self, X, y, batch_size=None):
        """
        Fit the linear regression model to the training data.

        Args:
            batch_size (int): The number of samples to use for each batch. If None, all samples are used.
        """
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0

        if batch_size is None:
            batch_size = n_samples

        for iteration in range(self.n_iterations):
            # Shuffle data each iteration (important for mini-batch/SGD)
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                batch_len = end_idx - start_idx
                
                ### Forward Pass
                y_pred_batch = X_batch @ self.weights + self.bias
                
                dw = -(2 / batch_len) * X_batch.T @ (y_batch - y_pred_batch)
                db = -(2 / batch_len) * np.sum(y_batch - y_pred_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
            ### Forward Pass for final loss on full dataset
            y_pred_full = X @ self.weights + self.bias
            loss = np.mean((y_pred_full - y) ** 2)
            self.losses.append(loss)
            self.weights_history.append(self.weights.copy())
            self.gradients_history.append({'dw': dw.copy(), 'db': db})

    def predict(self, X):
        return X @ self.weights + self.bias
    
    def get_losses(self):
        return self.losses
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
### ===============================
### Numerical Gradient Verification
### ===============================

def numerical_gradient_check(X, y, weights, bias, epsilon=1e-7):
    """
    Verify analytical gradients by comparing with numerical approximations.
    
    Uses finite differences: df/dx ≈ (f(x+ε) - f(x-ε)) / (2ε)
    
    Args:
        X: Feature matrix
        y: Target vector
        weights: Current weights
        bias: Current bias
        epsilon: Small value for finite difference
    
    Returns:
        Dictionary with analytical and numerical gradients
    """
    y = y.reshape(-1, 1)
    n_samples = len(X)
    
    # Compute analytical gradients
    y_pred = X @ weights + bias
    dw_analytical = -(2 / n_samples) * X.T @ (y - y_pred)
    db_analytical = -(2 / n_samples) * np.sum(y - y_pred)
    
    # Compute numerical gradient for weight
    weights_plus = weights.copy()
    weights_plus[0, 0] += epsilon
    loss_plus_w = np.mean((X @ weights_plus + bias - y) ** 2)
    
    weights_minus = weights.copy()
    weights_minus[0, 0] -= epsilon
    loss_minus_w = np.mean((X @ weights_minus + bias - y) ** 2)
    
    dw_numerical = (loss_plus_w - loss_minus_w) / (2 * epsilon)
    
    # Compute numerical gradient for bias
    loss_plus_b = np.mean((X @ weights + (bias + epsilon) - y) ** 2)
    loss_minus_b = np.mean((X @ weights + (bias - epsilon) - y) ** 2)
    
    db_numerical = (loss_plus_b - loss_minus_b) / (2 * epsilon)
    
    return {
        "dw_analytical": dw_analytical[0, 0],
        "dw_numerical": dw_numerical,
        "dw_difference": abs(dw_analytical[0, 0] - dw_numerical),
        "db_analytical": db_analytical,
        "db_numerical": db_numerical,
        "db_difference": abs(db_analytical - db_numerical)
    }


### ===============================
### Evaluation Metrics
### ===============================

def calculate_metrics(y_true, y_pred):
    """
    Calculate the evaluation metrics for the linear regression model.

    Args:
        y_true: True values (n_samples)
        y_pred: Predicted values (n_samples)

    Returns:
        mse: Mean squared error
        rmse: Root mean squared error
        mae: Mean absolute error
        r2: R-squared score
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    return { 
             "mse": mse, 
             "rmse": rmse, 
             "mae": mae, 
             "r2": r2 
    }


### ==============================
### Visualization
### ==============================

def plot_losses(losses):
    """
    Plot the training loss over iterations.
    """
    fig = px.line(x=range(len(losses)), y=losses, 
                  title="Training Loss",
                  labels={"x": "Iteration", "y": "Loss (MSE)"})
    st.plotly_chart(fig)


def plot_weights_history(weights_history):
    """
    Plot the weights of the linear regression model over iterations.
    """
    # Extract scalar weight values from the array history
    weights = [w[0][0] for w in weights_history]
    fig = px.line(x=range(len(weights)), y=weights, 
                  title="Weight Over Iterations",
                  labels={"x": "Iteration", "y": "Weight"})
    st.plotly_chart(fig)


def plot_gradients_history(gradients_history):
    """
    Plot the gradients of the linear regression model over iterations.
    """
    dw_values = [g['dw'][0][0] for g in gradients_history]
    db_values = [g['db'] for g in gradients_history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(dw_values))), y=dw_values, 
                             mode='lines', name='dw (weight gradient)'))
    fig.add_trace(go.Scatter(x=list(range(len(db_values))), y=db_values, 
                             mode='lines', name='db (bias gradient)'))
    fig.update_layout(title="Gradients Over Iterations",
                      xaxis_title="Iteration", yaxis_title="Gradient Value")
    st.plotly_chart(fig)

def plot_predictions(X, y_true, y_pred, title="Predictions vs Actual"):
    """
    Scatter plot of actual data with regression line.
    """
    X_flat = X.flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_flat, y=y_true, mode='markers', name='Actual'))
    
    # Sort for line plot
    idx = np.argsort(X_flat)
    fig.add_trace(go.Scatter(x=X_flat[idx], y=y_pred_flat[idx], 
                             mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig)


def plot_residuals(y_true, y_pred):
    """
    Plot residuals vs predicted values.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Residual Plot", 
                      xaxis_title="Predicted", yaxis_title="Residual")
    st.plotly_chart(fig)


def plot_train_val_split(X_train, y_train, X_val, y_val):
    """
    Visualize the train/validation split.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train.flatten(), y=y_train, 
                             mode='markers', name='Training'))
    fig.add_trace(go.Scatter(x=X_val.flatten(), y=y_val, 
                             mode='markers', name='Validation'))
    fig.update_layout(title="Train/Validation Split",
                      xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig)

### ==============================
### Main Function
### ==============================

def main():
    # ==========================================
    # HOME SCREEN / WELCOME
    # ==========================================
    st.title("Linear Regression with Gradient Descent")
    st.write("""
    Welcome to the Linear Regression Analysis App! This application demonstrates
    key concepts in machine learning including gradient descent optimization,
    train/test splitting, and model evaluation.
    """)

    st.info("""
    **Getting Started:**
    1. Upload your CSV file using the sidebar (must contain 'x' and 'y' columns)
    2. Adjust the model parameters as needed
    3. Click 'Train Model' to see the results
    """)

    # ==========================================
    # SIDEBAR - DATA UPLOAD
    # ==========================================
    st.sidebar.header("1. Upload Data")

    st.sidebar.markdown("[Generate a dataset](https://gcuswe2023-2025-br7euxcjtrtjg7qsxcd6yh.streamlit.app/)")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain 'x' and 'y' columns. 'y_true' column is optional."
    )

    st.sidebar.caption("Required columns: `x`, `y`")
    st.sidebar.caption("Optional column: `y_true` (true values without noise)")

    # ==========================================
    # SIDEBAR - DATA SETTINGS
    # ==========================================
    st.sidebar.header("2. Data Settings")
    test_size = st.sidebar.slider("Test Split %", 10, 50, 20, 5) / 100
    random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

    # ==========================================
    # SIDEBAR - MODEL SETTINGS
    # ==========================================
    st.sidebar.header("3. Model Settings")
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
        value=0.01
    )
    n_iterations = st.sidebar.slider("Iterations", 100, 2000, 500, 100)

    batch_mode = st.sidebar.selectbox(
        "Gradient Descent Mode",
        ["Full Batch", "Mini-Batch", "Stochastic (SGD)"]
    )

    batch_size = None  # Full batch
    if batch_mode == "Mini-Batch":
        batch_size = st.sidebar.slider("Batch Size", 8, 64, 32, 8)
    elif batch_mode == "Stochastic (SGD)":
        batch_size = 1

    # ==========================================
    # CHECK IF DATA IS UPLOADED
    # ==========================================
    if uploaded_file is None:
        st.warning("Please upload a CSV file using the sidebar to begin.")

        # Show example format
        st.subheader("Expected CSV Format")
        example_df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.1, 4.2, 5.8, 8.1, 9.9],
            "y_true": [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        st.dataframe(example_df, use_container_width=True)
        st.caption("Note: 'y_true' column is optional (used for comparing noisy vs true values)")
        return

    # ==========================================
    # LOAD AND VALIDATE DATA
    # ==========================================
    try:
        X, y, y_true = load_csv_from_upload(uploaded_file)
        st.sidebar.success(f"Loaded {len(X)} samples!")
    except ValueError as e:
        st.error(f"Error loading CSV: {e}")
        return
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return

    # Split data
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    st.header("1. Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(X))
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Test Samples", len(X_test))
        st.metric("Test Split", f"{test_size*100:.0f}%")
    
    plot_train_val_split(X_train, y_train, X_test, y_test)

    # Train button
    if st.button("Train Model", type="primary"):
        # Show initial guess before training
        st.header("2. Initial Model (Before Training)")
        init_model = LinearRegression(learning_rate=learning_rate, n_iterations=1)
        init_model.fit(X_train, y_train, batch_size=batch_size)
        y_pred_init = init_model.predict(X_train)
        plot_predictions(X_train, y_train, y_pred_init, title="Initial Random Guess")
        
        # Train full model
        st.header("3. Training Progress")
        model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
        
        with st.spinner("Training..."):
            model.fit(X_train, y_train, batch_size=batch_size)
        
        st.success("Training complete!")
        
        # Results
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Loss", "Predictions", "Residuals", "Parameters", "Metrics", "Gradient Verification", "Gradient Exploration"
        ])
        
        with tab1:
            plot_losses(model.get_losses())
            col1, col2 = st.columns(2)
            col1.metric("Initial Loss", f"{model.losses[0]:.4f}")
            col2.metric("Final Loss", f"{model.losses[-1]:.4f}")
        
        with tab2:
            plot_predictions(X_train, y_train, y_pred_train, "Training Set Predictions")
            plot_predictions(X_test, y_test, y_pred_test, "Test Set Predictions")
        
        with tab3:
            plot_residuals(y_train, y_pred_train)
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                plot_weights_history(model.weights_history)
            with col2:
                plot_gradients_history(model.gradients_history)
            
            st.write(f"**Final Weight:** {model.get_weights()[0][0]:.4f}")
            st.write(f"**Final Bias:** {model.get_bias():.4f}")
        
        with tab5:
            train_metrics = calculate_metrics(y_train, y_pred_train)
            test_metrics = calculate_metrics(y_test, y_pred_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Metrics")
                st.metric("MSE", f"{train_metrics['mse']:.4f}")
                st.metric("RMSE", f"{train_metrics['rmse']:.4f}")
                st.metric("MAE", f"{train_metrics['mae']:.4f}")
                st.metric("R²", f"{train_metrics['r2']:.4f}")
            with col2:
                st.subheader("Test Metrics")
                st.metric("MSE", f"{test_metrics['mse']:.4f}")
                st.metric("RMSE", f"{test_metrics['rmse']:.4f}")
                st.metric("MAE", f"{test_metrics['mae']:.4f}")
                st.metric("R²", f"{test_metrics['r2']:.4f}")
            
        with tab6:
            st.subheader("Numerical Gradient Verification")
            st.write("Comparing analytical gradients with numerical approximations (finite differences)")

            grad_check = numerical_gradient_check(X_train, y_train, model.weights, model.bias)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Weight Gradient (dw)**")
                st.write(f"Analytical: {grad_check['dw_analytical']:.8f}")
                st.write(f"Numerical:  {grad_check['dw_numerical']:.8f}")
                st.write(f"Difference: {grad_check['dw_difference']:.2e}")
            with col2:
                st.write("**Bias Gradient (db)**")
                st.write(f"Analytical: {grad_check['db_analytical']:.8f}")
                st.write(f"Numerical:  {grad_check['db_numerical']:.8f}")
                st.write(f"Difference: {grad_check['db_difference']:.2e}")
    
            if grad_check['dw_difference'] < 1e-5 and grad_check['db_difference'] < 1e-5:
                st.success("✓ Gradients verified! Analytical and numerical gradients match.")
            else:
                st.warning("⚠ Gradient mismatch detected. Check your implementation.")
        with tab7:
            st.subheader("Gradient Explosion Demonstration")
            st.write("What happens when the learning rate is too large?")
    
            explosion_model = LinearRegression(learning_rate=1.0, n_iterations=50)
            explosion_model.fit(X_train, y_train, batch_size=batch_size)

            col1, col2 = st.columns(2)
            with col1:
                plot_losses(explosion_model.get_losses())
            with col2:
                plot_weights_history(explosion_model.weights_history)

            st.warning("""
            **Observations:**
            - With a large learning rate, gradients cause parameters to overshoot
            - Loss may oscillate wildly or diverge to infinity (NaN)
            - Weights can explode to very large values
            - This is why learning rate tuning is critical!
            """)
            
if __name__ == "__main__":
    main()


