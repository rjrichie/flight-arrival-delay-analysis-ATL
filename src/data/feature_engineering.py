import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


# Canonical date column used throughout the codebase.
# `eda_utils.load_data()` is responsible for creating this column in-memory.
CANONICAL_DATE = 'Date (YYYY-MM-DD)'


def create_basic_features(df):
    """
    Create only essential features: IsWeekend, Is_Holiday_Period, Season, Is_Delayed
    
    Args:
        df: Input DataFrame (cleaned data)
    
    Returns:
        DataFrame with basic features added
    """
    print("\n" + "=" * 60)
    print("CREATING BASIC FEATURES")
    print("=" * 60)

    # Expect the canonical date column created by eda_utils.load_data()
    date_col = CANONICAL_DATE
    if date_col not in df.columns:
        raise KeyError(
            f"Expected column '{CANONICAL_DATE}' in DataFrame. "
            "Ensure you loaded the data via eda_utils.load_data() which creates the canonical date column."
        )
    # Ensure date column is datetime (coerce invalids to NaT)
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    df['DayOfWeek'] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # 1. IsWeekend
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    print("✓ Created IsWeekend")
    
    # 2. Is_Holiday_Period (vectorized)
    # Create boolean masks for known holiday windows and combine them.
    # This avoids an expensive row-wise apply and uses vectorized operations.
    m = df['Month']
    d = df['Day']

    mask_thanksgiving = (m == 11) & (d.between(22, 28))
    mask_christmas_newyear = ((m == 12) & (d >= 20)) | ((m == 1) & (d <= 5))
    mask_july4 = (m == 7) & (d.between(1, 7))
    mask_memorial = (m == 5) & (d >= 25)
    mask_labor = (m == 9) & (d.between(1, 7))

    df['Is_Holiday_Period'] = (
        mask_thanksgiving
        | mask_christmas_newyear
        | mask_july4
        | mask_memorial
        | mask_labor
    ).astype(int)
    print("✓ Created Is_Holiday_Period")

    # 3. Season (vectorized)
    conditions = [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])]
    choices = ['Winter', 'Spring', 'Summer']
    df['Season'] = np.select(conditions, choices, default='Fall')
    print("✓ Created Season")
    
    # 4. Is_Delayed (binary: >15 minutes)
    df['Is_Delayed'] = (df['Arrival Delay (Minutes)'] > 15).astype(int)
    print("✓ Created Is_Delayed")
    
    print(f"\nFeature summary:")
    print(f"  - IsWeekend: {df['IsWeekend'].sum():,} ({df['IsWeekend'].mean()*100:.2f}%)")
    print(f"  - Is_Holiday_Period: {df['Is_Holiday_Period'].sum():,} ({df['Is_Holiday_Period'].mean()*100:.2f}%)")
    print(f"  - Season: {df['Season'].value_counts().to_dict()}")
    print(f"  - Is_Delayed: {df['Is_Delayed'].sum():,} ({df['Is_Delayed'].mean()*100:.2f}%)")
    
    return df


def encode_categorical_features(df):
    """
    Encode categorical features for modeling
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with encoded features, and label encoders
    """
    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 60)
    
    df_encoded = df.copy()
    label_encoders = {}
    
    # One-Hot Encode Carrier Code
    print("\nOne-hot encoding Carrier Code...")
    df_encoded = pd.get_dummies(df_encoded, columns=['Carrier Code'], prefix='Carrier', dtype=int)
    carrier_cols = [col for col in df_encoded.columns if col.startswith('Carrier_')]
    print(f"✓ Created {len(carrier_cols)} carrier dummy variables")
    
    # Label Encode Origin Airport
    print(f"\nLabel encoding Origin Airport...")
    le_airport = LabelEncoder()
    df_encoded['Origin_Airport_Encoded'] = le_airport.fit_transform(df_encoded['Origin Airport'])
    label_encoders['Origin Airport'] = le_airport
    print(f"✓ Encoded {df_encoded['Origin Airport'].nunique()} unique airports")
    
    # Label Encode Season
    print(f"\nLabel encoding Season...")
    le_season = LabelEncoder()
    df_encoded['Season_Encoded'] = le_season.fit_transform(df_encoded['Season'])
    label_encoders['Season'] = le_season
    print(f"✓ Encoded Season: {dict(zip(le_season.classes_, le_season.transform(le_season.classes_)))}")
    
    print(f"\n✓ Encoding complete. Total columns: {df_encoded.shape[1]}")
    
    return df_encoded, label_encoders


def select_features_for_modeling(df):
    """
    Select features for modeling (simple version)
    
    Args:
        df: Input DataFrame
    
    Returns:
        List of feature column names
    """
    print("\n" + "=" * 60)
    print("SELECTING FEATURES FOR MODELING")
    print("=" * 60)
    
    # Features to EXCLUDE
    exclude_features = [
        # Targets (for regression and classification)
        'Arrival Delay (Minutes)',
        'Is_Delayed',  # Classification target
        
        # Target leakage (known only after flight)
        'Actual Arrival Time',
        'Actual Elapsed Time (Minutes)',
        'Wheels-on Time',
        'Taxi-In time (Minutes)',
        'Delay Carrier (Minutes)',
        'Delay Weather (Minutes)',
        'Delay National Aviation System (Minutes)',
        'Delay Security (Minutes)',
        'Delay Late Aircraft Arrival (Minutes)',
        
        # Original categorical (already encoded)
        'Origin Airport',
        'Season',

        # Identifiers, too specific so exclude
        'Tail Number',
        'Flight Number',
        'Date (YYYY-MM-DD)',
        # 'Date (MM/DD/YYYY)',
        
        # Time strings
        'Scheduled Arrival Time',
        
        # Temporary columns
        'Month', 'Day', 'DayOfWeek',
    ]
    
    # Select features
    feature_cols = [col for col in df.columns if col not in exclude_features]
    
    print(f"\n✓ Total features: {len(feature_cols)}")
    print(f"\nFeatures for modeling:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2}. {col}")
    
    return feature_cols


def feature_engineering_pipeline(input_path, output_path):
    """
    Simple feature engineering pipeline
    
    Args:
        input_path: Path to cleaned data CSV
        output_path: Path to save engineered data CSV
    
    Returns:
        DataFrame with engineered features, feature columns, encoders
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING PIPELINE (SIMPLIFIED)")
    print("=" * 80)
    
    # Load cleaned data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Create basic features
    df = create_basic_features(df)
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(df)
    
    # Select features
    feature_cols = select_features_for_modeling(df_encoded)
    
    # Save
    print("\n" + "=" * 60)
    print("SAVING ENGINEERED DATA")
    print("=" * 60)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save: Date + Features + Targets (Arrival Delay, Is_Delayed)
    # Order: Date first, then all features, then regression target, then classification target
    cols_to_save = ['Date (YYYY-MM-DD)'] + feature_cols + ['Arrival Delay (Minutes)', 'Is_Delayed']
    df_to_save = df_encoded[cols_to_save]
    
    df_to_save.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Shape: {df_to_save.shape}")
    print(f"✓ Column order: 1 date + {len(feature_cols)} features + 2 targets")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Regression target: Arrival Delay (Minutes)")
    print(f"  - Classification target: Is_Delayed")
    
    print("\n" + "=" * 80)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    
    return df_to_save, feature_cols, label_encoders


# Example usage
if __name__ == "__main__":
    input_file = "../data/processed/flight_delays_cleaned.csv"
    output_file = "../data/processed/flight_delays_engineered.csv"
    
    df_engineered, features, encoders = feature_engineering_pipeline(input_file, output_file)
    
    print(f"\n✓ Engineered dataset: {df_engineered.shape}")
    print(f"✓ Total features: {len(features)}")