import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Get project root directory - more robust approach
try:
    # When imported, __file__ gives the actual file location
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback for interactive environments (Jupyter)
    PROJECT_ROOT = Path(os.getcwd()).parent if Path(os.getcwd()).name == 'notebooks' else Path(os.getcwd())

def load_data(file_path='data/processed/flight_delays_combined.csv', nrows=None):
    """Load the combined flight delay dataset with optimized data types"""
    
    # Define data type mapping
    dtype_mapping = {
        # Categorical data
        'Carrier Code': 'category',
        'Tail Number': 'string',
        'Origin Airport': 'category',
        
        # Time columns as strings (can convert later if needed)
        'Scheduled Arrival Time': 'string',
        'Actual Arrival Time': 'string',
        'Wheels-on Time': 'string',
        
        # Numeric data (integers) - Use nullable integers for proper NaN handling
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
    
    print(f"Loading data from {file_path}...")
    
    # Load with specified dtypes
    df = pd.read_csv(
        file_path,
        dtype=dtype_mapping,
        nrows=nrows,
        low_memory=False
    )
    
    # Normalize date columns and create a single canonical datetime column
    # New canonical column name: 'Date (YYYY-MM-DD)'
    # The source CSVs use 'Date (MM/DD/YYYY)'.
    if 'Date (MM/DD/YYYY)' in df.columns:
        src = 'Date (MM/DD/YYYY)'
        parsed = pd.to_datetime(df[src], format='%m/%d/%Y', errors='coerce')
    # elif 'Date (YYYY-MM-DD)' in df.columns:
    #     src = 'Date (YYYY-MM-DD)'
    #     parsed = pd.to_datetime(df[src], errors='coerce')
    else:
        parsed = pd.Series([pd.NaT] * len(df), index=df.index, dtype='datetime64[ns]')


    df['Date (YYYY-MM-DD)'] = parsed
    if 'Date (MM/DD/YYYY)' in df.columns:
        try:
            df = df.drop(columns=['Date (MM/DD/YYYY)'])
            print("Dropped column 'Date (MM/DD/YYYY)', using 'Date (YYYY-MM-DD)'")
        except Exception:
            pass

    # Move canonical date column to the second position for convenience
    # (index 1, keeping original first column ordering otherwise).
    date_col = 'Date (YYYY-MM-DD)'
    if date_col in df.columns:
        cols = list(df.columns)
        try:
            cols.remove(date_col)
            # Insert at position 1 (second column)
            cols.insert(1, date_col)
            df = df.loc[:, cols]
            print(f"Moved '{date_col}' to column position 2 (in-memory)")
        except Exception:
            # If something goes wrong, leave original ordering
            pass
    
    # Memory usage info
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display dtype info
    print(f"\n Data types optimized:")
    print(f"  - Categorical: {len(df.select_dtypes(include='category').columns)}")
    print(f"  - Datetime: {len(df.select_dtypes(include='datetime').columns)}")
    print(f"  - Numeric (int): {len(df.select_dtypes(include=['int8', 'int16', 'int32', 'Int16', 'Int32']).columns)}")
    print(f"  - String/Object: {len(df.select_dtypes(include=['object', 'string']).columns)}")
    
    return df

def basic_info(df):
    """Display basic dataset information"""
    
    print("DATASET BASIC INFORMATION")
    
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    
    print("COLUMN INFORMATION")
    
    print(f"\n{'Column Name':<45} {'Type':<20} {'Non-Null':<15} {'Null %'}")
    print("-" * 90)
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        print(f"{col:<45} {dtype:<20} {non_null:<15,} {null_pct:>6.2f}%")
    
    # Summary by data type
    
    print("DATA TYPE SUMMARY")
    
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"{str(dtype):<20} {count:>3} columns")
    
    return df.info()

def missing_value_analysis(df):
    """Analyze missing values in the dataset"""
    
    print("MISSING VALUE ANALYSIS")
    
    
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_df) == 0:
        print("\n No missing values found!")
    else:
        print(f"\nColumns with missing values: {len(missing_df)}/{len(df.columns)}")
        print("\n" + missing_df.to_string(index=False))
    
    return missing_df

def visualize_missing_values(df, figsize=(12, 6)):
    """Visualize missing value patterns"""
    # Use absolute path from project root
    output_dir = PROJECT_ROOT / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("No missing values to visualize")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    missing.plot(kind='bar', ax=ax1, color='coral')
    ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Number of Missing Values')
    ax1.tick_params(axis='x', rotation=45)
    
    # Percentage plot
    missing_pct = (missing / len(df)) * 100
    missing_pct.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Missing Values Percentage', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / 'missing_values_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()

def analyze_missing_patterns(df):
    """Analyze patterns of missing values across rows and columns"""
    
    print("MISSING VALUE PATTERN ANALYSIS")
   
    
    # Count missing values per row
    missing_per_row = df.isna().sum(axis=1)
    print(f"\nRows with missing values: {(missing_per_row > 0).sum():,} ({(missing_per_row > 0).sum() / len(df) * 100:.2f}%)")
    print(f"Rows with ALL values: {(missing_per_row == 0).sum():,} ({(missing_per_row == 0).sum() / len(df) * 100:.2f}%)")
    
    # Distribution of missing values per row
    
    print("MISSING VALUES PER ROW DISTRIBUTION")
    
    missing_counts = missing_per_row.value_counts().sort_index()
    for n_missing, count in missing_counts.items():
        if n_missing > 0:
            print(f"{n_missing:>2} missing columns: {count:>6,} rows ({count/len(df)*100:>6.2f}%)")
    
    # Identify rows with multiple missing values
    print("ROWS WITH MULTIPLE MISSING VALUES")
   
    
    # Find rows with > 1 missing value
    rows_multiple_missing = df[missing_per_row > 1]
    
    if len(rows_multiple_missing) > 0:
        print(f"\nFound {len(rows_multiple_missing):,} rows with multiple missing values")
        print("\nSample of rows with missing data:")
        
        # Show sample rows
        for idx in rows_multiple_missing.index[:5]:
            missing_cols = df.loc[idx].isna()
            missing_col_names = missing_cols[missing_cols].index.tolist()
            print(f"\n  Row {idx}: {len(missing_col_names)} columns missing")
            print(f"    Missing: {', '.join(missing_col_names[:5])}")
            if len(missing_col_names) > 5:
                print(f"    ... and {len(missing_col_names) - 5} more")
    else:
        print("\nNo rows with multiple missing values")
    
    # Co-occurrence of missing values

    print("MISSING VALUE CO-OCCURRENCE")

    
    # Find columns that are always missing together
    missing_mask = df.isna()
    cols_with_missing = missing_mask.sum()[missing_mask.sum() > 0].index.tolist()
    
    if len(cols_with_missing) > 1:
        print(f"\nChecking {len(cols_with_missing)} columns with missing values...")
        
        # Check if certain columns always have missing values together
        for i, col1 in enumerate(cols_with_missing[:5]):  # Limit to first 5 for readability
            for col2 in cols_with_missing[i+1:6]:
                both_missing = (missing_mask[col1] & missing_mask[col2]).sum()
                if both_missing > 0:
                    print(f"  '{col1}' & '{col2}': {both_missing} rows both missing")
    
    return missing_per_row

def detect_outliers(df, columns, method='iqr'):
    """Detect outliers using IQR method"""

    print("OUTLIER DETECTION (IQR Method)")

    
    outlier_summary = []
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_pct = (len(outliers) / len(df)) * 100
            
            outlier_summary.append({
                'Column': col,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound,
                'Outlier_Count': len(outliers),
                'Outlier_Percentage': outlier_pct
            })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print("\n" + outlier_df.to_string(index=False))
    return outlier_df

def plot_distributions(df, numeric_cols, figsize=(15, 10)):
    """Plot distributions of numeric columns"""
    # Use absolute path from project root
    output_dir = PROJECT_ROOT / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        df[col].hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{col} Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()

def correlation_analysis(df, numeric_cols, figsize=(12, 10)):
    """Generate correlation matrix and heatmap"""
    # Use absolute path from project root
    output_dir = PROJECT_ROOT / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("CORRELATION ANALYSIS")

    
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = output_dir / 'correlation_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()
    
    return corr_matrix

def data_quality_report(df):
    """Generate comprehensive data quality report"""
    print("DATA QUALITY REPORT")

    
    quality_issues = []
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        quality_issues.append(f"Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.2f}%)")
    else:
        quality_issues.append("No duplicate rows found")

    # Check for missing values
    missing_cols = df.isna().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if len(missing_cols) > 0:
        quality_issues.append(f"{len(missing_cols)} columns have missing values")
    else:
        quality_issues.append("No missing values")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        quality_issues.append(f"Constant columns (no variance): {', '.join(constant_cols)}")
    else:
        quality_issues.append("No constant columns")
    
    # Check data types
    quality_issues.append(f"\nData type distribution:")
    quality_issues.append(f"  - Numeric: {len(df.select_dtypes(include=[np.number]).columns)}")
    quality_issues.append(f"  - Object/String: {len(df.select_dtypes(include=['object', 'string']).columns)}")
    quality_issues.append(f"  - DateTime: {len(df.select_dtypes(include=['datetime']).columns)}")
    
    print("\n" + "\n".join(quality_issues))
    
    return quality_issues