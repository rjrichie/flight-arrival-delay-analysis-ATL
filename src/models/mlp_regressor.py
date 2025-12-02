import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
from pathlib import Path
import time


def load_engineered_data(filepath):
    """
    Load the dataset for modeling.
    """
    df = pd.read_csv(filepath)
    return df


def prepare_data_for_modeling(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : array-like
        Train and test splits
    """
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize features by removing the mean and scaling to unit variance.
    MLPs are sensitive to feature scaling, so this is critical.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
        
    Returns:
    --------
    X_train_scaled, X_test_scaled : array-like
        Scaled features
    scaler : StandardScaler
        Fitted scaler object
    """
    print("\nSCALING FEATURES")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Mean (train): {X_train_scaled.mean():.6f}")
    print(f"  Std Dev (train): {X_train_scaled.std():.6f}")
    print(f"  ✓ Features scaled successfully")
    
    return X_train_scaled, X_test_scaled, scaler


def train_mlp_regressor(X_train, y_train, hidden_layer_sizes=(100, 50), 
                        activation='relu', solver='adam', alpha=0.0001,
                        batch_size='auto', learning_rate='constant',
                        learning_rate_init=0.001, max_iter=150,
                        random_state=42, verbose=True, early_stopping=False,
                        validation_fraction=0.1):
    """
    Train a Multi-Layer Perceptron Regressor for delay prediction.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should be scaled)
    y_train : array-like
        Training target (continuous delay values)
    hidden_layer_sizes : tuple
        Number of neurons in each hidden layer
        e.g., (100, 50) means 2 hidden layers with 100 and 50 neurons
    activation : str
        Activation function ('relu', 'tanh', 'logistic')
    solver : str
        Weight optimization solver ('adam', 'sgd', 'lbfgs')
    alpha : float
        L2 regularization parameter
    batch_size : int or 'auto'
        Size of minibatches for stochastic optimizers
    learning_rate : str
        Learning rate schedule ('constant', 'invscaling', 'adaptive')
    learning_rate_init : float
        Initial learning rate
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed
    verbose : bool
        Whether to print progress messages
    early_stopping : bool
        Whether to use early stopping to terminate training when validation score is not improving
    validation_fraction : float
        Proportion of training data to set aside as validation set for early stopping
        
    Returns:
    --------
    MLPRegressor
        Trained model
    """
    print("TRAINING MLP REGRESSOR")
    
    print(f"\nModel parameters:")
    print(f"  - hidden_layer_sizes: {hidden_layer_sizes}")
    print(f"  - activation: {activation}")
    print(f"  - solver: {solver}")
    print(f"  - alpha (L2 penalty): {alpha}")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - learning_rate_init: {learning_rate_init}")
    print(f"  - max_iter: {max_iter}")
    print(f"  - early_stopping: {early_stopping}")
    if early_stopping:
        print(f"  - validation_fraction: {validation_fraction}")
    
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=10  # Number of iterations with no improvement to wait before stopping
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {training_time:.2f} seconds")
    print(f"  Number of iterations: {model.n_iter_}")
    print(f"  Loss: {model.loss_:.4f}")
    
    return model


def tune_mlp_regressor_hyperparams(X_train, y_train, cv=3, n_jobs=-1, verbose=1):
    """
    Perform hyperparameter tuning for MLP Regressor using GridSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should be scaled)
    y_train : array-like
        Training target
    cv : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
        
    Returns:
    --------
    best_model : MLPRegressor
        Best model from grid search
    best_params : dict
        Best parameters found
    """
    print("STARTING HYPERPARAMETER TUNING (MLP REGRESSOR)")
    print("Note: This may take several minutes...\n")
    
    model = MLPRegressor(max_iter=300, random_state=42, early_stopping=True, n_iter_no_change=20)
    
    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(32,), (50,), (32, 16), (50, 25)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"\n✓ Hyperparameter tuning complete in {tuning_time:.2f} seconds")
    print(f"\nBest Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    print(f"\nBest Negative MSE Score: {grid_search.best_score_:.4f}")
    print(f"Best RMSE Score: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_regressor(model, X_train, y_train, X_test, y_test):
    """
    Evaluate MLP Regressor performance.
    
    Parameters:
    -----------
    model : MLPRegressor
        Trained model
    X_train, y_train : Training data
    X_test, y_test : Test data
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print("MLP REGRESSION MODEL EVALUATION")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTraining Set Performance:")
    print(f"  MSE:  {train_mse:.4f}")
    print(f"  RMSE: {train_rmse:.4f} minutes")
    print(f"  MAE:  {train_mae:.4f} minutes")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  MSE:  {test_mse:.4f}")
    print(f"  RMSE: {test_rmse:.4f} minutes")
    print(f"  MAE:  {test_mae:.4f} minutes")
    print(f"  R²:   {test_r2:.4f}")
    
    # Check for overfitting
    overfit_rmse = train_rmse - test_rmse
    overfit_r2 = train_r2 - test_r2
    
    print("\nOverfitting Analysis:")
    print(f"  RMSE difference (train - test): {overfit_rmse:.4f}")
    print(f"  R² difference (train - test): {overfit_r2:.4f}")
    
    if abs(overfit_rmse) < 1.0 and abs(overfit_r2) < 0.05:
        print("  Status: Good generalization ✓")
    elif overfit_rmse < -2.0 or overfit_r2 < -0.1:
        print("  Status: Possible underfitting (test performance better than train)")
    else:
        print("  Status: Some overfitting detected")
    
    metrics = {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'y_test_pred': y_test_pred,
        'y_train_pred': y_train_pred
    }
    
    return metrics


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : MLPRegressor
        Trained model
    filepath : str or Path
        Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"\n✓ Model saved to: {filepath}")


def save_scaler(scaler, filepath):
    """
    Save fitted scaler to disk.
    
    Parameters:
    -----------
    scaler : StandardScaler
        Fitted scaler
    filepath : str or Path
        Path to save the scaler
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, filepath)
    print(f"✓ Scaler saved to: {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the saved model
        
    Returns:
    --------
    MLPRegressor
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"✓ Model loaded from: {filepath}")
    return model


def load_scaler(filepath):
    """
    Load fitted scaler from disk.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the saved scaler
        
    Returns:
    --------
    StandardScaler
        Loaded scaler
    """
    scaler = joblib.load(filepath)
    print(f"✓ Scaler loaded from: {filepath}")
    return scaler
