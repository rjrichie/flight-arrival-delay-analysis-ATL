import pandas as pd
import numpy as np
from pathlib import Path


# Date column names (canonical vs legacy)
CANONICAL_DATE = 'Date (YYYY-MM-DD)'
LEGACY_DATE = 'Date (MM/DD/YYYY)'


def _choose_date_col(df):
    """Return the preferred date column name present in df.

    Prefers CANONICAL_DATE and falls back to LEGACY_DATE for backward compatibility.
    Raises KeyError if neither is present.
    """
    if CANONICAL_DATE in df.columns:
        return CANONICAL_DATE
    if LEGACY_DATE in df.columns:
        return LEGACY_DATE
    raise KeyError(f"No date column found. Expected '{CANONICAL_DATE}' or '{LEGACY_DATE}'")

def remove_invalid_rows(df):
    """
    Remove rows where MOST critical columns are missing (specifically the 11 problematic rows)
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with invalid rows removed
    """
    print("REMOVING INVALID ROWS")

    
    initial_len = len(df)
    
    # Define the critical columns that identify problematic rows
    # These are the columns that are missing in those 11 rows
    # Choose which date column is present (prefer canonical, fall back to legacy)
    date_col = None
    if 'Date (YYYY-MM-DD)' in df.columns:
        date_col = 'Date (YYYY-MM-DD)'
    elif 'Date (MM/DD/YYYY)' in df.columns:
        date_col = 'Date (MM/DD/YYYY)'

    critical_cols = [
        'Flight Number',
    ]

    if date_col:
        critical_cols.append(date_col)

    critical_cols += [
        'Origin Airport',
        'Scheduled Arrival Time',
        'Actual Arrival Time',
        'Scheduled Elapsed Time (Minutes)',
        'Actual Elapsed Time (Minutes)',
        'Arrival Delay (Minutes)',
        'Wheels-on Time',
        'Taxi-In time (Minutes)',
        'Delay Carrier (Minutes)',
        'Delay Weather (Minutes)',
        'Delay National Aviation System (Minutes)',
        'Delay Security (Minutes)',
        'Delay Late Aircraft Arrival (Minutes)'
    ]

    # Count how many critical columns are missing per row. Use reindex so missing
    # columns (if any) appear as NaN instead of raising KeyError.
    missing_count = df.reindex(columns=critical_cols).isna().sum(axis=1)
    
    # Remove rows where more than 10 critical columns are missing
    # (the 11 problematic rows have all 15 critical columns missing)
    df_clean = df[missing_count < 10].copy()
    
    removed = initial_len - len(df_clean)
    print(f"Removed {removed:,} rows with {len(critical_cols)} critical columns missing ({removed/initial_len*100:.4f}%)")
    print(f"Remaining: {len(df_clean):,} rows")
    
    return df_clean

def handle_missing_values(df, strategy='median'):
    """
    Handle missing values based on column type
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('median', 'mean', 'mode', 'drop')
    
    Returns:
        DataFrame with missing values handled
    """

    print("HANDLING MISSING VALUES")

    
    df = df.copy()
    
    # Check initial missing values
    initial_missing = df.isna().sum().sum()
    print(f"Initial missing values: {initial_missing:,}")
    
    # Numeric columns - fill with median or mean
    # Include both standard int/float AND nullable integer types (Int16, Int32)
    numeric_cols = df.select_dtypes(include=[
        'int8', 'int16', 'int32', 'int64', 
        'Int8', 'Int16', 'Int32', 'Int64',  # Nullable integers
        'float16', 'float32', 'float64'
    ]).columns
    
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            if strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mean':
                fill_value = df[col].mean()
            else:
                continue
            
            df[col] = df[col].fillna(fill_value)
            print(f"  - {col}: Filled {missing_count} values with {strategy} ({fill_value:.2f})")
    
    # Datetime columns - skip (they're handled separately)
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"  - {col}: Skipping {missing_count} missing datetime values (will drop rows)")
    
    # Categorical and string columns - fill with mode or 'Unknown'
    cat_cols = df.select_dtypes(include=['category', 'string', 'object']).columns
    
    for col in cat_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            # Skip time columns (keep as-is or drop later)
            if 'Time' in col:
                print(f"  - {col}: Skipping {missing_count} missing time values")
                continue
            
            # For Tail Number, fill with 'Unknown'
            if col == 'Tail Number':
                df[col] = df[col].fillna('Unknown')
                print(f"  - {col}: Filled {missing_count} values with 'Unknown'")
            else:
                # For other categoricals, use mode if available
                if len(df[col].mode()) > 0:
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
                    print(f"  - {col}: Filled {missing_count} values with mode ('{mode_value}')")
                else:
                    df[col] = df[col].fillna('Unknown')
                    print(f"  - {col}: Filled {missing_count} values with 'Unknown'")
    
    # Final missing value count
    final_missing = df.isna().sum().sum()
    print(f"\n Final missing values: {final_missing:,}")
    print(f"Reduced missing values by {initial_missing - final_missing:,}")
    
    return df

def validate_data_types(df):
    """
    Ensure correct data types and fix inconsistencies
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with validated data types
    """
    print("VALIDATING DATA TYPES")
    
    df = df.copy()
    changes_made = 0
    
    # Ensure Date column is datetime (prefer canonical name)
    date_col = _choose_date_col(df)

    if date_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            print(f"✓ Converted '{date_col}' to datetime")
            changes_made += 1
    
    # Ensure numeric columns are numeric (with nullable integer support)
    numeric_cols = {
        'Flight Number': 'Int32',
        'Scheduled Elapsed Time (Minutes)': 'Int16',
        'Actual Elapsed Time (Minutes)': 'Int16',
        'Arrival Delay (Minutes)': 'Int16',
        'Taxi-In time (Minutes)': 'Int16',
        'Delay Carrier (Minutes)': 'Int16',
        'Delay Weather (Minutes)': 'Int16',
        'Delay National Aviation System (Minutes)': 'Int16',
        'Delay Security (Minutes)': 'Int16',
        'Delay Late Aircraft Arrival (Minutes)': 'Int16'
    }
    
    for col, dtype in numeric_cols.items():
        if col in df.columns:
            current_dtype = str(df[col].dtype)
            if current_dtype != dtype:
                # Convert to numeric first, then to desired type
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'Int' in dtype:  # Nullable integer
                    df[col] = df[col].astype(dtype)
                print(f"✓ Converted '{col}' from {current_dtype} to {dtype}")
                changes_made += 1
    
    # Ensure categorical columns are proper type
    cat_cols = ['Carrier Code', 'Origin Airport']
    for col in cat_cols:
        if col in df.columns:
            if df[col].dtype != 'category':
                df[col] = df[col].astype('category')
                print(f"Converted '{col}' to category")
                changes_made += 1
    
    # Ensure time columns are string type
    time_cols = ['Scheduled Arrival Time', 'Actual Arrival Time', 'Wheels-on Time']
    for col in time_cols:
        if col in df.columns:
            if df[col].dtype != 'string':
                df[col] = df[col].astype('string')
                print(f"Converted '{col}' to string")
                changes_made += 1
    
    # Ensure Tail Number is string
    if 'Tail Number' in df.columns and df['Tail Number'].dtype != 'string':
        df['Tail Number'] = df['Tail Number'].astype('string')
        print(f"Converted 'Tail Number' to string")
        changes_made += 1
    
    if changes_made == 0:
        print("All data types are correct")
    else:
        print(f"\nData type validation complete ({changes_made} changes made)")
    
    return df

def handle_outliers(df, columns=None, method='cap', threshold=1.5):
    """
    Handle outliers in numeric columns
    
    Args:
        df: Input DataFrame
        columns: List of columns to check (None = all numeric)
        method: 'cap' (cap at threshold), 'remove' (remove outliers), or 'keep' (do nothing)
        threshold: IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
        DataFrame with outliers handled
    """

    print(f"HANDLING OUTLIERS (method={method})")

    
    if method == 'keep':
        print("Keeping outliers as-is (Random Forest is robust to outliers)")
        return df
    
    df = df.copy()
    
    if columns is None:
        # Get all numeric columns including nullable integers
        columns = df.select_dtypes(include=[
            'int8', 'int16', 'int32', 'int64',
            'Int8', 'Int16', 'Int32', 'Int64',
            'float16', 'float32', 'float64'
        ]).columns.tolist()
    
    outlier_summary = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Calculate outlier bounds using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers = outliers_mask.sum()
        
        if outliers > 0:
            if method == 'cap':
                # Cap outliers at bounds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"  - {col}: Capped {outliers} outliers ({outliers/len(df)*100:.2f}%)")
            elif method == 'remove':
                # Track for removal (will remove after loop)
                outlier_summary.append(outliers_mask)
                print(f"  - {col}: Marked {outliers} outliers for removal")
    
    # Remove all rows with outliers (if method='remove')
    if method == 'remove' and outlier_summary:
        # Combine all outlier masks
        combined_mask = pd.concat(outlier_summary, axis=1).any(axis=1)
        initial_len = len(df)
        df = df[~combined_mask]
        removed = initial_len - len(df)
        print(f"\n Removed {removed:,} rows with outliers ({removed/initial_len*100:.2f}%)")
    
    print(f"Outlier handling complete")
    
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with duplicates removed
    """
    print("REMOVING DUPLICATES")
    
    initial_len = len(df)
    df = df.drop_duplicates()
    removed = initial_len - len(df)
    
    print(f"Removed {removed:,} duplicate rows ({removed/initial_len*100:.2f}%)")
    print(f"Remaining: {len(df):,} rows")
    
    return df

def filter_by_date_range(df, start_date=None, end_date=None):
    """
    Filter data by date range
    
    Args:
        df: Input DataFrame
        start_date: Start date (string 'YYYY-MM-DD' or datetime)
        end_date: End date (string 'YYYY-MM-DD' or datetime)
    
    Returns:
        Filtered DataFrame
    """
    print("FILTERING BY DATE RANGE")
    
    df = df.copy()
    initial_len = len(df)
    
    # Determine which date column to use
    try:
        date_col = _choose_date_col(df)
    except KeyError:
        print("Date column not found, skipping date filter")
        return df

    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df[date_col] >= start_date]
        print(f"Filtered from: {start_date.date()}")

    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df[date_col] <= end_date]
        print(f"Filtered to: {end_date.date()}")
    
    removed = initial_len - len(df)
    print(f"Removed {removed:,} rows outside date range")
    print(f"Remaining: {len(df):,} rows")
    
    return df

def prepare_regression_target(df, target_col='Arrival Delay (Minutes)', 
                             cap_min=-60, cap_max=300):
    """
    Prepare regression target by removing extreme outliers.
    
    Args:
        df: Input DataFrame
        target_col: Name of delay column
        cap_min: Minimum acceptable delay (e.g., -60 = 1 hour early)
        cap_max: Maximum acceptable delay (e.g., 300 = 5 hours late)
    
    Returns:
        DataFrame with extreme outliers removed
    """
    print(f"\n{'='*60}")
    print("PREPARING REGRESSION TARGET")
    print(f"{'='*60}")
    
    df = df.copy()
    initial_len = len(df)
    
    # Show distribution before cleaning
    print(f"\nOriginal '{target_col}' statistics:")
    print(df[target_col].describe())
    
    # Count extremes
    too_early = (df[target_col] < cap_min).sum()
    too_late = (df[target_col] > cap_max).sum()
    
    print(f"\nExtreme outliers detected:")
    print(f"  Too early (< {cap_min} min): {too_early:,} ({too_early/initial_len*100:.4f}%)")
    print(f"  Too late  (> {cap_max} min): {too_late:,} ({too_late/initial_len*100:.4f}%)")
    
    # Remove extremes
    df = df[(df[target_col] >= cap_min) & (df[target_col] <= cap_max)]
    
    removed = initial_len - len(df)
    print(f"\n✓ Removed {removed:,} extreme outliers ({removed/initial_len*100:.4f}%)")
    
    # Show final distribution
    print(f"\nCleaned '{target_col}' statistics:")
    print(df[target_col].describe())
    
    print(f"\nDelay categories:")
    print(f"  Early (< 0 min):      {(df[target_col] < 0).sum():,}")
    print(f"  On-time (0-15 min):   {((df[target_col] >= 0) & (df[target_col] <= 15)).sum():,}")
    print(f"  Delayed (> 15 min):   {(df[target_col] > 15).sum():,}")
    
    return df

def preprocessing_pipeline(input_path, output_path, 
                          remove_invalid=True,
                          handle_missing=True,
                          remove_dupes=True,
                          validate_dtypes=True,
                          handle_outliers_method='keep',
                          date_range=None):
    """
    Complete preprocessing pipeline
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        remove_invalid: Whether to remove invalid rows
        handle_missing: Whether to handle missing values
        remove_dupes: Whether to remove duplicates
        validate_dtypes: Whether to validate data types
        handle_outliers_method: How to handle outliers ('cap', 'remove', 'keep')
        date_range: Tuple of (start_date, end_date) to filter
    
    Returns:
        Cleaned DataFrame
    """
    print("STARTING DATA PREPROCESSING PIPELINE")
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    # --- Ensure canonical date column exists in-memory (do NOT modify files on disk) ---
    # If the CSV still contains the legacy header, convert it to the canonical
    # name and drop the legacy header. Otherwise ensure the canonical column is
    # parsed as datetime.
    if 'Date (MM/DD/YYYY)' in df.columns:
        df[CANONICAL_DATE] = pd.to_datetime(df['Date (MM/DD/YYYY)'], errors='coerce')
        df = df.drop(columns=['Date (MM/DD/YYYY)'])
        print(f"Created in-memory canonical date column '{CANONICAL_DATE}' from legacy header and dropped legacy column")
    elif CANONICAL_DATE in df.columns:
        df[CANONICAL_DATE] = pd.to_datetime(df[CANONICAL_DATE], errors='coerce')

    # Move canonical date column to second position for consistency with eda_utils
    if CANONICAL_DATE in df.columns:
        cols = list(df.columns)
        if cols[1] != CANONICAL_DATE:
            cols = [c for c in cols if c != CANONICAL_DATE]
            cols.insert(1, CANONICAL_DATE)
            df = df.loc[:, cols]
            print(f"Moved '{CANONICAL_DATE}' to column position 2 (in-memory)")
    # ------------------------------------------------------------------------------
    
    # Step 1: Remove invalid rows
    if remove_invalid:
        df = remove_invalid_rows(df)
    
    # Step 2: Filter by date range (if specified)
    if date_range:
        start_date, end_date = date_range
        df = filter_by_date_range(df, start_date, end_date)
    
    # Step 3: Handle missing values
    if handle_missing:
        df = handle_missing_values(df, strategy='median')
    
    # Step 4: Remove duplicates
    if remove_dupes:
        df = remove_duplicates(df)
    
    # Step 5: Validate data types
    if validate_dtypes:
        df = validate_data_types(df)
    
    # Step 6: Handle outliers
    if handle_outliers_method != 'keep':
        df = handle_outliers(df, method=handle_outliers_method)
    
    # Save cleaned data
    print("SAVING CLEANED DATA")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"Final shape: {len(df):,} rows, {len(df.columns)} columns")
    
    # Summary
    print("PREPROCESSING COMPLETE")
    print(f"Output file: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def preprocessing_pipeline_regression(input_path, output_path, 
                                     remove_invalid=True,
                                     handle_missing=True,
                                     remove_dupes=True,
                                     validate_dtypes=True,
                                     handle_outliers_method='keep',
                                     regression_cap_min=-60,
                                     regression_cap_max=300,
                                     date_range=None):
    """
    Preprocessing pipeline specifically for REGRESSION tasks.
    
    Identical to preprocessing_pipeline() but adds regression target cleaning.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        remove_invalid: Whether to remove invalid rows
        handle_missing: Whether to handle missing values
        remove_dupes: Whether to remove duplicates
        validate_dtypes: Whether to validate data types
        handle_outliers_method: How to handle outliers ('cap', 'remove', 'keep')
        regression_cap_min: Minimum delay cap for regression (default: -60)
        regression_cap_max: Maximum delay cap for regression (default: 300)
        date_range: Tuple of (start_date, end_date) to filter
    
    Returns:
        Cleaned DataFrame for regression
    """
    print("=" * 80)
    print("STARTING REGRESSION PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Ensure canonical date column exists
    if 'Date (MM/DD/YYYY)' in df.columns:
        df[CANONICAL_DATE] = pd.to_datetime(df['Date (MM/DD/YYYY)'], errors='coerce')
        df = df.drop(columns=['Date (MM/DD/YYYY)'])
        print(f"Created canonical date column '{CANONICAL_DATE}'")
    elif CANONICAL_DATE in df.columns:
        df[CANONICAL_DATE] = pd.to_datetime(df[CANONICAL_DATE], errors='coerce')

    # Move canonical date column to second position
    if CANONICAL_DATE in df.columns:
        cols = list(df.columns)
        if cols[1] != CANONICAL_DATE:
            cols = [c for c in cols if c != CANONICAL_DATE]
            cols.insert(1, CANONICAL_DATE)
            df = df.loc[:, cols]
    
    # Step 1: Remove invalid rows
    if remove_invalid:
        df = remove_invalid_rows(df)
    
    # Step 2: Filter by date range (if specified)
    if date_range:
        start_date, end_date = date_range
        df = filter_by_date_range(df, start_date, end_date)
    
    # Step 3: Handle missing values
    if handle_missing:
        df = handle_missing_values(df, strategy='median')
    
    # Step 4: Remove duplicates
    if remove_dupes:
        df = remove_duplicates(df)
    
    # Step 5: Validate data types
    if validate_dtypes:
        df = validate_data_types(df)
    
    # Step 6: REGRESSION-SPECIFIC - Prepare regression target
    df = prepare_regression_target(
        df, 
        target_col='Arrival Delay (Minutes)',
        cap_min=regression_cap_min,
        cap_max=regression_cap_max
    )
    
    # Step 7: Handle outliers (optional, usually keep for regression)
    if handle_outliers_method != 'keep':
        df = handle_outliers(df, method=handle_outliers_method)
    
    # Save cleaned data
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA (REGRESSION)")
    print("=" * 60)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Final shape: {len(df):,} rows, {len(df.columns)} columns")
    
    # Summary
    print("\n" + "=" * 80)
    print("REGRESSION PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# Example usage (keep existing, add new example)
if __name__ == "__main__":
    # CLASSIFICATION preprocessing (existing)
    input_file = '../../data/processed/flight_delays_combined.csv'
    output_file_classification = '../../data/processed/flight_delays_cleaned.csv'
    
    df_clean = preprocessing_pipeline(
        input_path=input_file,
        output_path=output_file_classification,
        remove_invalid=True,
        handle_missing=True,
        remove_dupes=True,
        validate_dtypes=True,
        handle_outliers_method='keep',
        date_range=None
    )
    
    print("\n✓ Classification preprocessing complete!")
    
    # REGRESSION preprocessing (NEW)
    output_file_regression = '../../data/processed/flight_delays_cleaned_regression.csv'
    
    df_clean_regression = preprocessing_pipeline_regression(
        input_path=input_file,
        output_path=output_file_regression,
        remove_invalid=True,
        handle_missing=True,
        remove_dupes=True,
        validate_dtypes=True,
        handle_outliers_method='keep',
        regression_cap_min=-60,
        regression_cap_max=300,
        date_range=None
    )
    
    print("\n✓ Regression preprocessing complete!")