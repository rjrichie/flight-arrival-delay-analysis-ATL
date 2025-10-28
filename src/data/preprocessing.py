import pandas as pd
import numpy as np
from pathlib import Path

def remove_invalid_rows(df):
    """
    Remove rows where MOST critical columns are missing (specifically the 11 problematic rows)
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with invalid rows removed
    """
    print("\n" + "=" * 60)
    print("REMOVING INVALID ROWS")
    print("=" * 60)
    
    initial_len = len(df)
    
    # Define the critical columns that identify problematic rows
    # These are the columns that are missing in those 11 rows
    critical_cols = [
        'Flight Number',
        'Date (MM/DD/YYYY)',
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
    
    # Count how many critical columns are missing per row
    missing_count = df[critical_cols].isna().sum(axis=1)
    
    # Remove rows where more than 10 critical columns are missing
    # (the 11 problematic rows have all 15 critical columns missing)
    df_clean = df[missing_count < 10].copy()
    
    removed = initial_len - len(df_clean)
    print(f"✓ Removed {removed:,} rows with {len(critical_cols)} critical columns missing ({removed/initial_len*100:.4f}%)")
    print(f"✓ Remaining: {len(df_clean):,} rows")
    
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
    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)
    
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
    print(f"\n✓ Final missing values: {final_missing:,}")
    print(f"✓ Reduced missing values by {initial_missing - final_missing:,}")
    
    return df

def validate_data_types(df):
    """
    Ensure correct data types and fix inconsistencies
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with validated data types
    """
    print("\n" + "=" * 60)
    print("VALIDATING DATA TYPES")
    print("=" * 60)
    
    df = df.copy()
    changes_made = 0
    
    # Ensure Date column is datetime
    if 'Date (MM/DD/YYYY)' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date (MM/DD/YYYY)']):
            df['Date (MM/DD/YYYY)'] = pd.to_datetime(df['Date (MM/DD/YYYY)'], errors='coerce')
            print("✓ Converted 'Date (MM/DD/YYYY)' to datetime")
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
                print(f"✓ Converted '{col}' to category")
                changes_made += 1
    
    # Ensure time columns are string type
    time_cols = ['Scheduled Arrival Time', 'Actual Arrival Time', 'Wheels-on Time']
    for col in time_cols:
        if col in df.columns:
            if df[col].dtype != 'string':
                df[col] = df[col].astype('string')
                print(f"✓ Converted '{col}' to string")
                changes_made += 1
    
    # Ensure Tail Number is string
    if 'Tail Number' in df.columns and df['Tail Number'].dtype != 'string':
        df['Tail Number'] = df['Tail Number'].astype('string')
        print(f"✓ Converted 'Tail Number' to string")
        changes_made += 1
    
    if changes_made == 0:
        print("✓ All data types are correct")
    else:
        print(f"\n✓ Data type validation complete ({changes_made} changes made)")
    
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
    print("\n" + "=" * 60)
    print(f"HANDLING OUTLIERS (method={method})")
    print("=" * 60)
    
    if method == 'keep':
        print("✓ Keeping outliers as-is (Random Forest is robust to outliers)")
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
        print(f"\n✓ Removed {removed:,} rows with outliers ({removed/initial_len*100:.2f}%)")
    
    print(f"✓ Outlier handling complete")
    
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with duplicates removed
    """
    print("\n" + "=" * 60)
    print("REMOVING DUPLICATES")
    print("=" * 60)
    
    initial_len = len(df)
    df = df.drop_duplicates()
    removed = initial_len - len(df)
    
    print(f"✓ Removed {removed:,} duplicate rows ({removed/initial_len*100:.2f}%)")
    print(f"✓ Remaining: {len(df):,} rows")
    
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
    print("\n" + "=" * 60)
    print("FILTERING BY DATE RANGE")
    print("=" * 60)
    
    df = df.copy()
    initial_len = len(df)
    
    if 'Date (MM/DD/YYYY)' not in df.columns:
        print("⚠ Date column not found, skipping date filter")
        return df
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['Date (MM/DD/YYYY)'] >= start_date]
        print(f"✓ Filtered from: {start_date.date()}")
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['Date (MM/DD/YYYY)'] <= end_date]
        print(f"✓ Filtered to: {end_date.date()}")
    
    removed = initial_len - len(df)
    print(f"✓ Removed {removed:,} rows outside date range")
    print(f"✓ Remaining: {len(df):,} rows")
    
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
    print("\n" + "=" * 80)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
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
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA")
    print("=" * 60)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Final shape: {len(df):,} rows, {len(df.columns)} columns")
    
    # Summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# Example usage
if __name__ == "__main__":
    # Run preprocessing pipeline
    input_file = '../../data/processed/flight_delays_combined.csv'
    output_file = '../../data/processed/flight_delays_cleaned.csv'
    
    df_clean = preprocessing_pipeline(
        input_path=input_file,
        output_path=output_file,
        remove_invalid=True,
        handle_missing=True,
        remove_dupes=True,
        validate_dtypes=True,
        handle_outliers_method='keep',  # Keep outliers for Random Forest
        date_range=None  # No date filtering
    )
    
    print("\n✓ Preprocessing complete!")