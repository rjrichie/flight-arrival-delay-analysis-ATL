# Flight Delay Analysis - CS4641 Project

## Project Overview
Machine learning project analyzing flight delays using data from the Bureau of Transportation Statistics. Implements multiple classification and regression models to predict flight delays and identify key contributing factors.

## Final Report
The final report can be accessed through [here](https://rjrichie.github.io/ml-flight-arrival-delay/).

## Project Structure

### Root Directory
- `README.md`: Project documentation and directory structure
- `Dataset_BTS.zip`: Raw flight data from Bureau of Transportation Statistics
- `requirements.txt`: Python package dependencies
- `.gitignore`: Git ignore rules
- `.gitattributes`: Git LFS configuration for large files

### `/notebooks/`: Jupyter notebooks for analysis (run in order)
- `01_exploratory_data_analysis.ipynb`: Initial data exploration, statistical analysis, and quality assessment
- `02_data_preprocessing.ipynb`: Data cleaning, missing value handling, and outlier treatment
- `03_feature_engineering.ipynb`: Feature creation, categorical encoding, and feature selection
- `04_model_training_random_forest.ipynb`: Random Forest model training and evaluation (classification & regression)
- `05_model_training_logistic_regression.ipynb`: Logistic Regression model training and evaluation (classification)
- `06_model_training_mlp.ipynb`: Neural Network model training and evaluation (classification & regression)
- `07_model_training_XGBoost.ipynb`: XGBoost Gradient Boosting model training and evaluation (classification & regression)
- `draft.ipynb`: Scratch notebook for testing and experimental queries (not part of main workflow)

### `/src/`: Source code modules

#### `/src/data/`: Data processing scripts and utilities
- `extract_data.py`: Extract Dataset_BTS.zip and combine monthly CSV files into single dataset
- `eda_utils.py`: Utility functions for exploratory data analysis (statistics, visualization, quality checks)
- `preprocessing.py`: Data cleaning functions (missing values, duplicates, outliers, validation)
- `feature_engineering.py`: Feature creation and encoding pipeline (temporal, operational, categorical features)
  
#### `/src/models/`: Model training and evaluation scripts
- `random_forest.py`: Random Forest classifier and regressor implementation with hyperparameter tuning
- `logistic_regression.py`: Logistic Regression model with class balancing and regularization
- `mlp_classifier.py`: Multi-layer Perceptron classifier implementation for binary classification
- `mlp_regressor.py`: Multi-layer Perceptron regressor implementation for delay prediction
- `XGBoost.py`: XGBoost Gradient Boosting classifier and regressor with early stopping

### `/data/`: Processed datasets
- `raw/`: Unprocessed data extracted from Dataset_BTS.zip
- `processed/`: Cleaned and preprocessed datasets ready for modeling
  - `flight_delays_engineered.csv`: Feature-engineered dataset for classification tasks
  - `flight_delays_engineered_regression.csv`: Feature-engineered dataset for regression tasks

### `/results/`: Output files and visualizations

#### Saved Models (Root level)
- `logistic_regression_balanced_model.joblib`: Trained Logistic Regression model with class balancing
- `mlp_classifier_model.joblib`: Trained Multi-layer Perceptron classifier
- `mlp_classifier_model_scaler.joblib`: StandardScaler for MLP classifier feature normalization
- `mlp_regressor_model.joblib`: Trained Multi-layer Perceptron regressor
- `mlp_regressor_scaler.joblib`: StandardScaler for MLP regressor feature normalization

#### Performance Reports (Root level)
- `model_performance_summary.csv`: Logistic Regression classifier performance metrics (balanced)
- `model_performance_summary_logistic_regression.csv`: Detailed Logistic Regression metrics and hyperparameters
- `model_performance_summary_random_forest.csv`: Detailed Random Forest classification metrics
- `model_performance_summary_mlp.csv`: Detailed Neural Network performance metrics (classification & regression)
- `rf_regression_cv_results.csv`: Random Forest regression cross-validation results

#### `/results/models/`: Additional model artifacts
- `XGBoost_performance_report.json`: XGBoost model performance metrics and configuration details
- `best_hyperparameters.json`: Optimal hyperparameters found during XGBoost tuning

#### `/results/figures/`: Generated plots and visualizations

**Data Exploration & Preprocessing:**
- `correlation_matrix.png`: Feature correlation heatmap showing relationships between variables
- `data_quality_after_preprocessing.png`: Data quality visualizations after cleaning operations
- `delay_features_distribution_log.png`: Distribution of delay-related features using log scale
- `distributions.png`: General feature distributions from exploratory data analysis
- `missing_values_analysis.png`: Missing value patterns and visualization before preprocessing
- `time_features_distribution.png`: Temporal feature distributions (hour, day, month patterns)
- `regression_target_analysis.png`: Analysis of continuous delay target variable for regression

**Feature Engineering:**
- `feature_engineering_analysis.png`: Comprehensive 6-panel visualization of engineered features
- `classification_vs_regression_comparison.png`: Comparison of data distributions for classification vs regression datasets

**Model Performance Visualizations:**

*Random Forest:*
- `rf_classification_results.png`: Random Forest confusion matrix and ROC curve for classification
- `rf_tuned_classification_results.png`: Performance metrics after hyperparameter tuning
- `rf_top15_feature_importances.png`: Top 15 most important features from Random Forest model
- `rf_predicted_probability_distribution.png`: Distribution of predicted probabilities
- `rf_regression_results.png`: Random Forest regression performance analysis
- `rf_regression_actual_vs_predicted.png`: Scatter plot comparing actual vs predicted delay values

*Logistic Regression:*
- `lr_classification_results.png`: Logistic Regression confusion matrix, ROC curve, and metrics

*Multi-layer Perceptron:*
- `mlp_classifier_output.png`: MLP classifier training history and learning curves
- `mlp_classifier_results.png`: MLP classifier confusion matrix and performance metrics
- `mlp_regressor_results.png`: MLP regressor actual vs predicted plot and residual analysis

*XGBoost:*
- `XGBoost_performance.png`: Comprehensive XGBoost performance dashboard (classification & regression metrics, feature importance, ROC curve, actual vs predicted)

## Setup and Installation

```bash
# Clone the repository
git clone <repository-url>
cd CS4641_flight_delay_analysis

# Install dependencies
pip install -r requirements.txt

# Extract and prepare data
python src/data/extract_data.py
```
