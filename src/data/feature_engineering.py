import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


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
    
    df = df.copy()
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date (MM/DD/YYYY)']):
        df['Date (MM/DD/YYYY)'] = pd.to_datetime(df['Date (MM/DD/YYYY)'])
    
    # Extract Month and Day for feature creation
    df['Month'] = df['Date (MM/DD/YYYY)'].dt.month
    df['Day'] = df['Date (MM/DD/YYYY)'].dt.day
    df['DayOfWeek'] = df['Date (MM/DD/YYYY)'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # 1. IsWeekend
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    print("✓ Created IsWeekend")
    
    # 2. Is_Holiday_Period
    def is_holiday(month, day):
        if month == 11 and 22 <= day <= 28:  # Thanksgiving
            return 1
        elif (month == 12 and day >= 20) or (month == 1 and day <= 5):  # Christmas/New Year
            return 1
        elif month == 7 and 1 <= day <= 7:  # July 4th
            return 1
        elif month == 5 and day >= 25:  # Memorial Day
            return 1
        elif month == 9 and 1 <= day <= 7:  # Labor Day
            return 1
        return 0
    
    df['Is_Holiday_Period'] = df.apply(lambda x: is_holiday(x['Month'], x['Day']), axis=1)
    print("✓ Created Is_Holiday_Period")
    
    # 3. Season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season)
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
        # Target
        'Arrival Delay (Minutes)',
        
        # Target leakage (known only after flight)
        'Actual Arrival Time',
        'Actual Elapsed Time (Minutes)',
        'Wheels-on Time',
        'Taxi-In time (Minutes)',
        
        # Original categorical (already encoded)
        'Origin Airport',
        'Season',

        # Identifiers, too specific so exclude
        'Tail Number',
        'Flight Number',
        'Date (MM/DD/YYYY)',
        
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
    
    df_encoded.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Shape: {df_encoded.shape}")
    
    print("\n" + "=" * 80)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    
    return df_encoded, feature_cols, label_encoders


# Example usage
if __name__ == "__main__":
    input_file = "../data/processed/flight_delays_cleaned.csv"
    output_file = "../data/processed/flight_delays_engineered.csv"
    
    df_engineered, features, encoders = feature_engineering_pipeline(input_file, output_file)
    
    print(f"\n✓ Engineered dataset: {df_engineered.shape}")
    print(f"✓ Total features: {len(features)}")