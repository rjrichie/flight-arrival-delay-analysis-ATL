
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
from pathlib import Path
import time
#from tabulate import tabulate # Used for printing clean feature tables


def load_engineered_data(filepath):
    """
    Load the dataset for modeling.
    """
    df = pd.read_csv(filepath)
    return df


def prepare_data_for_modeling(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """
    Split data into train and test sets (non-stratified for consistency).
    """
    X = df[feature_cols]
    y = df[target_col]
    
    # NOTE: Using stratify=y is generally recommended for imbalanced data,
    # but the existing RF notebook did not use it. We maintain consistency.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nSTARTING DATA SPLIT")
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, C=1.0, penalty='l2', class_weight=None, solver='saga', random_state=42, n_jobs=-1, verbose=0, max_iter=5000):
    """
    Train a Logistic Regression model for classification.
    
    Parameters:
    - class_weight: Use 'balanced' to handle class imbalance.
    """
    print("\n" + "=" * 60)
    print("STARTING LOGISTIC REGRESSION TRAINING")
    print("=" * 60)
    start_time = time.time()

    # Logistic Regression does not have 'regressor' or 'classifier' distinction,
    # it is inherently a classifier.
    model = LogisticRegression(
        C=C, # Inverse of regularization strength
        penalty=penalty,
        class_weight=class_weight,
        solver=solver, # 'saga' handles both L1/L2 and works well with large datasets
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        max_iter=max_iter # Increase max_iter for convergence
    )
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"✓ Model trained in: {end_time - start_time:.2f} seconds")
    
    return model


def tune_logistic_regression_hyperparams(X_train, y_train, cv=5, n_jobs=-1, verbose=1):
    """
    Perform hyperparameter tuning for Logistic Regression using GridSearchCV.
    Returns the best estimator and best parameters.
    """
    print("\n" + "=" * 60)
    print("STARTING HYPERPARAMETER TUNING (LOGISTIC REGRESSION)")
    print("=" * 60)

    model = LogisticRegression(max_iter=5000, solver='saga')

    # Define parameter grid to search over
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',   # can use 'roc_auc', 'accuracy', etc.
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters Found:")
    print(grid_search.best_params_)
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_classifier(model, X_test, y_test, X_train=None, y_train=None, threshold=0.5):
    """
    Evaluate the classifier on the test set.
    """
    print("\n" + "=" * 60)
    print("CLASSIFIER EVALUATION (LOGISTIC REGRESSION)")
    print("=" * 60)
    """
    # Predict probabilities (needed for ROC AUC)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Predict classes using the standard threshold
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    # --- Test Metrics ---
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
    
    metrics = {
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
    """
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


def get_feature_coefficients(model, feature_names, top_n=20):
    """
    Display and return top logistic regression coefficients.
    """
    # For binary classification, model.coef_ is a 2D array, we flatten it.
    coef = model.coef_.flatten() 
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Abs(Coefficient)': np.abs(coef)
    }).sort_values('Abs(Coefficient)', ascending=False)
    
    # Use to_string() for clean printing without external libraries
    top_coef_string = coef_df.head(top_n).to_string(index=False, float_format="%.4f")
    
    print("\n" + "=" * 60)
    print(f"TOP {top_n} MOST INFLUENTIAL FEATURES (by Absolute Coefficient)")
    print("=" * 60)
    print(top_coef_string)
    
    return coef_df


def save_model(model, filepath):
    """
    Save the trained model to disk.
    """
    joblib.dump(model, filepath)
    print(f"\n✓ Model successfully saved to: {filepath}")
