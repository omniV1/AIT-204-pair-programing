import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
    # LOAD AND VALIDATE DATA
    # ==========================================
    X = 0
    y = 0
    # Split data
    #X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    st.header("1. Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", X)
        #st.metric("Training Samples", len(X_train))
    with col2:
        #st.metric("Test Samples", len(X_test))
        st.metric("Test Split", f"{test_size*100:.0f}%")
    
    #plot_train_val_split(X_train, y_train, X_test, y_test)

    # Train button
    if st.button("Train Model", type="primary"):
        # Show initial guess before training
        st.header("2. Initial Model (Before Training)")
        #init_model = LinearRegression(learning_rate=learning_rate, n_iterations=1)
        #init_model.fit(X_train, y_train, batch_size=batch_size)
        y_pred_init = init_model.predict(X_train)
        #plot_predictions(X_train, y_train, y_pred_init, title="Initial Random Guess")
        
        # Train full model
        st.header("3. Training Progress")
        #model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
        
        st.success("Training complete!")
        
        # Results
        #y_pred_train = model.predict(X_train)
        #y_pred_test = model.predict(X_test)
        
        # Tabs for different visualizations
        '''tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Loss", "Predictions", "Residuals", "Parameters", "Metrics", "Gradient Verification", "Gradient Exploration"
        ])
        
        
        
        
        with tab4:
            col1, col2 = st.columns(2)
            
        
        with tab5:
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Metrics")
                st.metric("MSE ")
                st.metric("RMSE")
            with col2:
                st.subheader("Test Metrics")
                st.metric("MSE")
            
        with tab6:
            st.subheader("Numerical Gradient Verification")
            st.write("Comparing analytical gradients with numerical approximations (finite differences)")


            col1, col2 = st.columns(2)
            with col1:
                st.write("**Weight Gradient (dw)**")
            with col2:
                st.write("**Bias Gradient (db)**")
    
            st.warning("""
            **Observations:**
            - With a large learning rate, gradients cause parameters to overshoot
            - Loss may oscillate wildly or diverge to infinity (NaN)
            - Weights can explode to very large values
            - This is why learning rate tuning is critical!
            """)'''
            
if __name__ == "__main__":
    main()


