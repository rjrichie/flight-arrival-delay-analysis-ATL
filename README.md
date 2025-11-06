# Flight Delay Analysis - CS4641 Project

## Project Structure

### Root Directory
- `README.md`: Project documentation and directory structure
- `Dataset_BTS.zip`: Raw flight data from Bureau of Transportation Statistics
- `requirements.txt`: Python package dependencies
- `.gitignore`: Git ignore rules
- `.gitattributes`: Tells Git to handle specific large files differently than regular code files

### `/notebooks/`: Jupyter notebooks for analysis (run in order)
- `01_exploratory_data_analysis.ipynb`: Initial data exploration, statistical analysis, and quality assessment
- `02_data_preprocessing.ipynb`: Data cleaning, missing value handling, and outlier treatment
- `03_feature_engineering.ipynb`: Feature creation, categorical encoding, and feature selection
- `04_model_training_random_forest.ipynb`: Random Forest model training and evaluation (classification)
- `05_model_training_logistic_regression.ipynb`: Logistic Regression model training and evaluation (classification)
- `draft.ipynb`: Scratch notebook for testing and queries (not part of main workflow)

### `/src/`: Source code modules
- `/src/data/`: Data processing scripts and utilities
  - `extract_data.py`: Extract Dataset_BTS.zip and combine monthly CSV files into one dataset
  - `eda_utils.py`: Utility functions for exploratory data analysis (statistics, visualization, quality checks)
  - `preprocessing.py`: Data cleaning functions (missing values, duplicates, outliers, validation)
  - `feature_engineering.py`: Feature creation and encoding pipeline (temporal, operational, categorical)
  
- `/src/models/`: Model training and evaluation scripts
  - `random_forest.py`: Random Forest model training, evaluation, and utility functions
  - `logistic_regression.py`: Logistic Regression model training, evaluation, and hyperparameter tuning

### `/results/`: Output files and visualizations

#### Root level files:
- `logistic_regression_balanced_model.joblib`: Saved Logistic Regression model (balanced)
- `model_performance_summary.csv`: Balanced logistic regression model metrics
- `model_performance_summary_logistic_regression.csv`: Logistic Regression specific metrics
- `model_performance_summary_random_forest.csv`: Random Forest specific metrics
- `preprocessing_summary.csv`: Summary statistics from data cleaning
- `preprocessing_summary.txt`: Detailed preprocessing report

#### `/results/figures/`: Generated plots and visualizations
- `correlation_matrix.png`: Feature correlation heatmap
- `data_quality_after_preprocessing.png`: Data quality visualizations post-cleaning
- `delay_features_distribution_log.png`: Distribution of delay-related features (log scale)
- `distributions.png`: General feature distributions from EDA
- `feature_engineering_analysis.png`: Feature engineering insights (6 subplots)
- `lr_classification_results.png`: Logistic Regression confusion matrix and ROC curve
- `missing_values_analysis.png`: Missing value patterns visualization
- `rf_classification_results.png`: Random Forest confusion matrix and ROC curve (original model)
- `rf_predicted_probability_distribution.png`: Prediction probability distribution
- `rf_regression_results.png`: Random Forest regression results (if regression task was performed)
- `rf_top15_feature_importances.png`: Top 15 feature importances (Random Forest)
- `rf_tuned_classification_results.png`: Tuned Random Forest performance
- `time_features_distribution.png`: Temporal feature distributions