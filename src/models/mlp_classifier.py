import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
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
        Target column name (binary: 0=on-time, 1=delayed)
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
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    # Show class distribution
    train_class_dist = pd.Series(y_train).value_counts()
    test_class_dist = pd.Series(y_test).value_counts()
    
    print(f"\n  Training set class distribution:")
    print(f"    Class 0 (On-time): {train_class_dist.get(0, 0):,} ({train_class_dist.get(0, 0)/len(y_train)*100:.2f}%)")
    print(f"    Class 1 (Delayed): {train_class_dist.get(1, 0):,} ({train_class_dist.get(1, 0)/len(y_train)*100:.2f}%)")
    
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


def train_mlp_classifier(X_train, y_train, hidden_layer_sizes=(100, 50),
                         activation='relu', solver='adam', alpha=0.0001,
                         batch_size='auto', learning_rate='constant',
                         learning_rate_init=0.001, max_iter=150,
                         random_state=42, verbose=True, early_stopping=False,
                         validation_fraction=0.1):
    """
    Train a Multi-Layer Perceptron Classifier for delay detection (binary).
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should be scaled)
    y_train : array-like
        Training target (binary: 0=on-time, 1=delayed)
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
    MLPClassifier
        Trained model
    """
    print("TRAINING MLP CLASSIFIER")
    
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
    
    model = MLPClassifier(
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
    model.fit(X_train, y_train, sample_weight=sample_weights) # WE BALANCE THE IMBALANCED CLASS
    training_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {training_time:.2f} seconds")
    print(f"  Number of iterations: {model.n_iter_}")
    print(f"  Loss: {model.loss_:.4f}")
    
    return model


def tune_mlp_classifier_hyperparams(X_train, y_train, cv=3, n_jobs=-1, verbose=1):
    """
    Perform hyperparameter tuning for MLP Classifier using GridSearchCV.
    
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
    best_model : MLPClassifier
        Best model from grid search
    best_params : dict
        Best parameters found
    """
    print("STARTING HYPERPARAMETER TUNING (MLP CLASSIFIER)")
    print("Note: This may take several minutes...\n")
    
    model = MLPClassifier(max_iter=300, random_state=42, early_stopping=True)
    
    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(50,), (50, 100)],
        'activation': ['relu'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',  # Optimize for F1 score
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    sample_weights = class_weights_array[y_train.astype(int)]
    start_time = time.time()
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    tuning_time = time.time() - start_time
    
    print(f"\n✓ Hyperparameter tuning complete in {tuning_time:.2f} seconds")
    print(f"\nBest Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    print(f"\nBest F1 Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_classifier(model, X_train, y_train, X_test, y_test):
    """
    Evaluate MLP Classifier performance.
    
    Parameters:
    -----------
    model : MLPClassifier
        Trained model
    X_train, y_train : Training data
    X_test, y_test : Test data
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print("MLP CLASSIFICATION MODEL EVALUATION")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print("\nTraining Set Performance:")
    print(f"  Accuracy:  {train_accuracy:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  ROC-AUC:   {test_roc_auc:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"  True Negatives:  {cm[0,0]:,}")
    print(f"  False Positives: {cm[0,1]:,}")
    print(f"  False Negatives: {cm[1,0]:,}")
    print(f"  True Positives:  {cm[1,1]:,}")
    
    # Classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['On-time', 'Delayed'],
                                zero_division=0))
    
    # Overfitting check
    overfit_accuracy = train_accuracy - test_accuracy
    overfit_f1 = train_f1 - test_f1
    
    print("Overfitting Analysis:")
    print(f"  Accuracy difference (train - test): {overfit_accuracy:.4f}")
    print(f"  F1 difference (train - test): {overfit_f1:.4f}")
    
    if abs(overfit_accuracy) < 0.02 and abs(overfit_f1) < 0.02:
        print("  Status: Good generalization ✓")
    elif overfit_accuracy < -0.05 or overfit_f1 < -0.05:
        print("  Status: Possible underfitting (test performance better than train)")
    else:
        print("  Status: Some overfitting detected")
    
    metrics = {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'confusion_matrix': cm,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }
    
    return metrics


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : MLPClassifier
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
    MLPClassifier
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
