import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import joblib
from pathlib import Path
import json
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FlightDelayPredictor:
    """
    XGBoost model for flight delay prediction - supports binary classification and regression.
    """
    
    def __init__(self, task_type='regression', config=None, balancing=None):
        """
        Initialize model for flight delay prediction.
        
        Args:
            task_type (str): 'regression' (delay minutes) or 'classification' (delayed/not-delayed)
            config (dict): Model hyperparameters
            balancing (dict): Balancing configuration for classification
        """
        self.task_type = task_type.lower()
        if self.task_type not in ['regression', 'classification']:
            raise ValueError("task_type must be 'regression' or 'classification'")
        
        # Default configurations optimized for flight delay prediction
        self.default_config_reg = {
            'n_estimators': 1000,
            'max_depth': 8,  # Slightly deeper for complex delay patterns
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'mae'
        }
        
        self.default_config_clf = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'logloss',
            'scale_pos_weight': 1.0
        }
        
        self.config = config or (self.default_config_reg if self.task_type == 'regression' 
                                else self.default_config_clf)
        self.balancing = balancing or {'method': 'none'}  # 'none', 'class_weight', 'smote', 'undersample'
        
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self.target_name = None
        self._last_X_eval = None  # Store last X for probability calculations
        
    def load_data(self, data_path, target_col=None, classification_threshold=15):
        """Load flight delay dataset and prepare target variable."""
        print("📊 Loading and exploring data...")
        
        df = pd.read_csv(data_path)
        
        # Auto-detect or create target column
        if target_col is None:
            if self.task_type == 'regression':
                if 'Arrival Delay (Minutes)' in df.columns:
                    target_col = 'Arrival Delay (Minutes)'
                elif 'Departure Delay (Minutes)' in df.columns:
                    target_col = 'Departure Delay (Minutes)'
                else:
                    raise ValueError("No regression target column found")
            else:  # classification
                if 'Is_Delayed' in df.columns:
                    target_col = 'Is_Delayed'
                elif 'Arrival Delay (Minutes)' in df.columns:
                    print(f"Creating binary classification target with threshold {classification_threshold} minutes")
                    target_col = 'Is_Delayed'
                    df[target_col] = (df['Arrival Delay (Minutes)'] > classification_threshold).astype(int)
                else:
                    raise ValueError("No suitable classification target column found")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Remove Date column - it's already captured in seasonal features
        if 'Date (YYYY-MM-DD)' in df.columns:
            df = df.drop(columns=['Date (YYYY-MM-DD)'])
            print("✓ Removed Date column (redundant - already captured in seasonal features)")
        
        # Separate features and target
        self.feature_columns = [col for col in df.columns if col != target_col]
        X = df[self.feature_columns]
        y = df[target_col]
        
        # DEBUG: Check data types before validation
        print(f"✓ Feature dtypes after preprocessing:")
        for col in X.columns:
            print(f"  - {col}: {X[col].dtype}")
        
        # Validate all features are numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"⚠️  Warning: Non-numeric columns found: {non_numeric}")
            print("Converting to numeric...")
            for col in non_numeric:
                # Try to convert to numeric, fill NaN with 0 for safety
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                print(f"  - Converted {col} to numeric")
        
        # Double-check after conversion
        non_numeric_after = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_after:
            print(f"❌ Still non-numeric columns after conversion: {non_numeric_after}")
            print("❌ Dropping non-numeric columns...")
            X = X.select_dtypes(include=[np.number])
            self.feature_columns = [col for col in X.columns]
        
        # Validate classification target is binary
        if self.task_type == 'classification':
            # Delete Arrival Delay (Minutes) if it exists to avoid leakage
            if 'Arrival Delay (Minutes)' in X.columns:
                X = X.drop(columns=['Arrival Delay (Minutes)'])
                # Update feature columns
                self.feature_columns = [col for col in X.columns]

            unique_classes = np.unique(y)
            if len(unique_classes) != 2:
                raise ValueError(f"Classification requires binary target. Found {len(unique_classes)} classes: {unique_classes}")
            print(f"✓ Binary classification: Class 0 (Not Delayed), Class 1 (Delayed)")
        
        print(f"✓ Loaded {self.task_type} dataset: {df.shape[0]:,} rows, {len(self.feature_columns)} features")
        print(f"✓ Target: {target_col}")
        
        if self.task_type == 'regression':
            print(f"✓ Delay range: [{y.min():.1f}, {y.max():.1f}] minutes")
            print(f"✓ Mean delay: {y.mean():.2f} minutes")
        else:
            class_counts = y.value_counts().sort_index()
            delay_ratio = class_counts[1] / class_counts[0] if 1 in class_counts else 0
            print(f"✓ Class distribution: Not Delayed (0): {class_counts.get(0, 0):,}, Delayed (1): {class_counts.get(1, 0):,}")
            print(f"✓ Delay ratio: {delay_ratio:.3f}")
        
        return X, y

    def apply_balancing(self, X, y):
        """
        Apply balancing techniques for imbalanced flight delay classification.
        
        Args:
            X, y: Original features and target
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        if self.task_type != 'classification' or self.balancing['method'] == 'none':
            return X, y
        
        method = self.balancing['method']
        print(f"✓ Applying balancing method: {method}")
        
        if method == 'class_weight':
            # Calculate class weights for flight delays
            class_counts = Counter(y)
            total = sum(class_counts.values())
            weights = {cls: total / count for cls, count in class_counts.items()}
            self.config['scale_pos_weight'] = weights[1] / weights[0]  # For binary classification
            print(f"✓ Class weights - Not Delayed: {weights[0]:.2f}, Delayed: {weights[1]:.2f}")
            return X, y
            
        elif method == 'smote':
            smote_params = self.balancing.get('params', {'random_state': 42, 'k_neighbors': 5})
            smote = SMOTE(**smote_params)
            X_bal, y_bal = smote.fit_resample(X, y)
            print(f"✓ After SMOTE - Not Delayed: {Counter(y_bal)[0]:,}, Delayed: {Counter(y_bal)[1]:,}")
            return X_bal, y_bal
            
        elif method == 'undersample':
            rus_params = self.balancing.get('params', {'random_state': 42})
            rus = RandomUnderSampler(**rus_params)
            X_bal, y_bal = rus.fit_resample(X, y)
            print(f"✓ After Undersampling - Not Delayed: {Counter(y_bal)[0]:,}, Delayed: {Counter(y_bal)[1]:,}")
            return X_bal, y_bal
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Split flight delay data into training and testing sets.
        
        Args:
            X, y: Features and target
            test_size (float): Proportion for test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True,
            stratify=y if self.task_type == 'classification' else None
        )
        
        print(f"✓ Train-test split: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test samples")
        
        if self.task_type == 'classification':
            train_delay_ratio = np.sum(y_train) / len(y_train)
            test_delay_ratio = np.sum(y_test) / len(y_test)
            print(f"✓ Delay ratio - Train: {train_delay_ratio:.3f}, Test: {test_delay_ratio:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model for flight delay prediction.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
        """
        print(f"\n🚀 Training {self.task_type} model for flight delay prediction...")
        
        # Apply balancing to training data (for classification only)
        X_train_bal, y_train_bal = self.apply_balancing(X_train, y_train)
        
        # Initialize appropriate model
        if self.task_type == 'regression':
            self.model = XGBRegressor(**self.config)
        else:
            self.model = XGBClassifier(**self.config)
        
        # Prepare evaluation sets
        eval_set = [(X_train_bal, y_train_bal)]
        eval_names = ['train']
        
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append('val')
        
        # Train model
        self.model.fit(
            X_train_bal, y_train_bal,
            eval_set=eval_set,
            verbose=100  # Print every 100 iterations
        )
        
        self.is_trained = True
        print(f"✓ Model trained successfully!")
        print(f"✓ Best iteration: {self.model.best_iteration}")
        print(f"✓ Best score: {self.model.best_score:.4f}")
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
            
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Store for potential probability calculations
        self._last_X_eval = X.copy()
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Input features
            
        Returns:
            np.array: Prediction probabilities for class 1 (Delayed)
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def predict_with_threshold(self, X, threshold=0.5):
        """
        Make classification predictions with custom threshold.
        Useful for adjusting precision/recall trade-off.
        
        Args:
            X: Input features
            threshold (float): Classification threshold (0-1)
            
        Returns:
            np.array: Binary predictions
        """
        if self.task_type != 'classification':
            raise ValueError("Threshold prediction only available for classification")
        
        probas = self.predict_proba(X)
        return (probas[:, 1] > threshold).astype(int)
    
    def evaluate(self, y_true, y_pred, dataset_name="Test"):
        """
        Evaluate model performance with flight delay specific metrics.
        
        Args:
            y_true, y_pred: True and predicted values
            dataset_name (str): Name for display
            
        Returns:
            dict: Performance metrics
        """
        print(f"\n{'='*60}")
        print(f"📊 {dataset_name.upper()} SET PERFORMANCE")
        print(f"{'='*60}")
        
        if self.task_type == 'regression':
            return self._evaluate_regression(y_true, y_pred, dataset_name)
        else:
            return self._evaluate_classification(y_true, y_pred, dataset_name)
    
    def _evaluate_regression(self, y_true, y_pred, dataset_name):
        """Evaluate regression performance for delay minutes."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Flight delay specific metrics
        within_5_min = np.mean(np.abs(y_true - y_pred) <= 5) * 100
        within_15_min = np.mean(np.abs(y_true - y_pred) <= 15) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Within_5min_%': within_5_min,
            'Within_15min_%': within_15_min
        }
        
        print(f"Mean Absolute Error (MAE):     {mae:.2f} minutes")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
        print(f"R² Score:                      {r2:.4f}")
        print(f"Explained Variance:            {r2*100:.2f}%")
        print(f"Predictions within 5 minutes:  {within_5_min:.1f}%")
        print(f"Predictions within 15 minutes: {within_15_min:.1f}%")
        
        return metrics
    
    def _evaluate_classification(self, y_true, y_pred, dataset_name):
        """Evaluate classification performance for delay prediction."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # ROC-AUC
        if self._last_X_eval is not None and len(self._last_X_eval) == len(y_true):
            try:
                y_pred_proba = self.predict_proba(self._last_X_eval)
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                roc_auc = None
        else:
            roc_auc = None
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc
        }
        
        print(f"Accuracy:                      {accuracy:.4f}")
        print(f"Precision (Delayed):           {precision:.4f}")
        print(f"Recall (Delayed):              {recall:.4f}")
        print(f"F1 Score:                      {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC:                       {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"               No Delay  Delay")
        print(f"Actual No Delay  {tn:>6}   {fp:>6}")
        print(f"Actual Delay     {fn:>6}   {tp:>6}")
        
        # Additional business metrics
        delay_detection_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        
        print(f"\nBusiness Metrics:")
        print(f"Delay Detection Rate: {delay_detection_rate:.1f}%")
        print(f"False Alarm Rate:     {false_alarm_rate:.1f}%")
        
        return metrics
    
    def get_feature_importance(self, importance_type='weight', top_n=15):
        """
        Get feature importance scores for flight delay prediction.
        
        Args:
            importance_type (str): Type of importance ('weight', 'gain', 'cover')
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        print(f"\n📈 TOP {top_n} FEATURE IMPORTANCES:")
        print("=" * 40)
        for i, row in importance_df.iterrows():
            print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
            
        return importance_df
    
    def save_model(self, model_path, feature_path=None):
        """
        Save the trained model and feature columns.
        
        Args:
            model_path (str): Path to save model
            feature_path (str): Path to save feature columns (optional)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save feature columns
        if feature_path:
            Path(feature_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.feature_columns, feature_path)
            print(f"✓ Feature columns saved to: {feature_path}")
        
        # Save metadata
        metadata = {
            'task_type': self.task_type,
            'config': self.config,
            'balancing': self.balancing,
            'target_name': self.target_name,
            'feature_count': len(self.feature_columns),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        config_path = str(Path(model_path).with_suffix('.json'))
        with open(config_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {config_path}")
    
    @classmethod
    def load_model(cls, model_path, feature_path=None):
        """
        Load a trained flight delay prediction model.
        
        Args:
            model_path (str): Path to saved model
            feature_path (str): Path to saved feature columns
            
        Returns:
            FlightDelayPredictor: Loaded model instance
        """
        # Load metadata
        config_path = str(Path(model_path).with_suffix('.json'))
        with open(config_path, 'r') as f:
            metadata = json.load(f)
        
        instance = cls(
            task_type=metadata['task_type'],
            config=metadata['config'],
            balancing=metadata['balancing']
        )
        
        instance.model = joblib.load(model_path)
        instance.is_trained = True
        instance.target_name = metadata['target_name']
        
        if feature_path:
            instance.feature_columns = joblib.load(feature_path)
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Task type: {metadata['task_type']}")
        print(f"✓ Trained on: {metadata['timestamp']}")
        
        return instance
    
    def get_model_info(self):
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "status": "trained",
            "task_type": self.task_type,
            "target_name": self.target_name,
            "feature_count": len(self.feature_columns),
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "balancing_method": self.balancing['method']
        }
        
        return info


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Regression (predict delay minutes)
    print("🔧 REGRESSION EXAMPLE")
    reg_model = FlightDelayPredictor(task_type='regression')
    # X_reg, y_reg = reg_model.load_data('flight_data_engineered.csv')
    # X_train, X_test, y_train, y_test = reg_model.train_test_split(X_reg, y_reg)
    # reg_model.train(X_train, y_train, X_test, y_test)
    # predictions_reg = reg_model.predict(X_test)
    # metrics_reg = reg_model.evaluate(y_test, predictions_reg, "Regression Test")
    
    # Example 2: Classification (predict delayed/not-delayed)
    print("\n\n🔧 CLASSIFICATION EXAMPLE")
    clf_model = FlightDelayPredictor(
        task_type='classification',
        balancing={'method': 'class_weight'}  # Handle imbalanced delays
    )
    # X_clf, y_clf = clf_model.load_data('flight_data_engineered.csv', classification_threshold=15)
    # X_train, X_test, y_train, y_test = clf_model.train_test_split(X_clf, y_clf)
    # clf_model.train(X_train, y_train, X_test, y_test)
    # predictions_clf = clf_model.predict(X_test)
    # metrics_clf = clf_model.evaluate(y_test, predictions_clf, "Classification Test")
    
    # Get feature importance
    # importance = clf_model.get_feature_importance(top_n=10)
    
    # Save model
    # clf_model.save_model('flight_delay_classifier.pkl', 'feature_columns.pkl')