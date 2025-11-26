import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
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
    Split data into train and test sets. (Random Forest => Random State)
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


def train_random_forest_regressor(X_train, y_train, n_estimators=100, 
                                   max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, random_state=42, 
                                   n_jobs=-1, verbose=1):
    """
    Train a Random Forest Regressor for delay prediction.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target (continuous delay values)
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of trees
    min_samples_split : int
        Minimum samples required to split a node
    min_samples_leaf : int
        Minimum samples required at a leaf node
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs (-1 uses all cores)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    RandomForestRegressor
        Trained model
    """
    print("TRAINING RANDOM FOREST REGRESSOR")
    
    print(f"\nModel parameters:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - max_depth: {max_depth}")
    print(f"  - min_samples_split: {min_samples_split}")
    print(f"  - min_samples_leaf: {min_samples_leaf}")
    print(f"  - random_state: {random_state}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {training_time:.2f} seconds")
    
    return model


def train_random_forest_classifier(X_train, y_train, n_estimators=100,
                                    max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, class_weight=None,
                                    random_state=42, n_jobs=-1, verbose=1):
    """
    Train a Random Forest Classifier for delay detection (binary).
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target (binary: 0=on-time, 1=delayed)
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of trees
    min_samples_split : int
        Minimum samples required to split a node
    min_samples_leaf : int
        Minimum samples required at a leaf node
    class_weight : str, dict, or None
        Weights associated with classes. Use 'balanced' to handle class imbalance
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs (-1 uses all cores)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    print("TRAINING RANDOM FOREST CLASSIFIER")
    
    print(f"\nModel parameters:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - max_depth: {max_depth}")
    print(f"  - min_samples_split: {min_samples_split}")
    print(f"  - min_samples_leaf: {min_samples_leaf}")
    print(f"  - class_weight: {class_weight}")
    print(f"  - random_state: {random_state}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {training_time:.2f} seconds")
    
    return model


def evaluate_regressor(model, X_train, y_train, X_test, y_test):
    """
    Evaluate Random Forest Regressor performance.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    X_train, y_train : Training data
    X_test, y_test : Test data
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print("REGRESSION MODEL EVALUATION")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTraining Set Performance:")
    print(f"  RMSE: {train_rmse:.4f} minutes")
    print(f"  MAE:  {train_mae:.4f} minutes")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:.4f} minutes")
    print(f"  MAE:  {test_mae:.4f} minutes")
    print(f"  R²:   {test_r2:.4f}")
    
    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'y_test_pred': y_test_pred,
        'y_train_pred': y_train_pred
    }
    
    return metrics


def evaluate_classifier(model, X_train, y_train, X_test, y_test):
    """
    Evaluate Random Forest Classifier performance.
        
    Return:
    dict
        Dictionary containing evaluation metrics
    """
    print("CLASSIFICATION MODEL EVALUATION")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
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
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['On-time', 'Delayed']))
    
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


def get_feature_importance(model, feature_names, top_n=20):
    """
    Extract and sort feature importances.
    
    Parameters:
    -----------
    model : RandomForestRegressor or RandomForestClassifier
        Trained model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print(feature_importance_df.head(top_n).to_string(index=False))
    
    return feature_importance_df


def save_model(model, filepath):
    """
    Save trained model to disk.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"\n Model saved to: {filepath}")


def load_model(filepath):
    """
    Load trained model
    """
    model = joblib.load(filepath)
    return model


def train_and_evaluate_regressor(
    df,
    feature_cols,
    target_col,
    test_size=0.2,
    random_state=42,
    model_params=None,
    save_path=None,
    top_n_features=20,
):
    """
    Convenience pipeline for Random Forest regression.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe containing features and target
    feature_cols : list
        List of column names to use as features
    target_col : str
        Name of the target column (continuous)
    test_size : float
        Fraction of data used for testing
    random_state : int
        Random seed for reproducibility
    model_params : dict or None
        Additional parameters passed to `train_random_forest_regressor`
    save_path : str or Path or None
        If provided, saves trained model to this path
    top_n_features : int
        How many top features to return in importance DataFrame

    Returns
    -------
    dict
        Contains keys: model, metrics, feature_importances, X_train, X_test, y_train, y_test
    """
    if model_params is None:
        model_params = {}

    # Split data
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    # Train
    model = train_random_forest_regressor(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        **model_params
    )

    # Evaluate
    metrics = evaluate_regressor(model, X_train, y_train, X_test, y_test)

    # Feature importance
    try:
        importance_df = get_feature_importance(model, feature_cols, top_n=top_n_features)
    except Exception:
        importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': np.zeros(len(feature_cols))})

    # Save model if path provided
    if save_path:
        save_model(model, save_path)

    return {
        'model': model,
        'metrics': metrics,
        'feature_importances': importance_df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }
