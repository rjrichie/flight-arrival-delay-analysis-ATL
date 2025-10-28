import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path

def extract_and_process_flight_data(zip_path='Dataset_BTS.zip', 
                                     raw_folder='data/raw', 
                                     processed_folder='data/processed'):
    """
    Extract BTS flight data, organize CSVs, clean headers, and combine into single file.
    
    Parameters:
    - zip_path: Path to the zip file
    - raw_folder: Folder for raw CSV files
    - processed_folder: Folder for processed combined CSV
    """
    
    print("=" * 60)
    print("FLIGHT DELAY DATA EXTRACTION & PROCESSING")
    print("=" * 60)
    
    # Step 1: Create necessary folders
    raw_path = Path(raw_folder)
    processed_path = Path(processed_folder)
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Extract zip file
    temp_extract_folder = 'temp_extracted'
    print(f"\n[1/5] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_folder)
    print("!!! Extraction complete")

    # # Step 2.5: DEBUG - Explore the extracted folder structure
    # print(f"\nExploring extracted folder structure...")
    # temp_path = Path(temp_extract_folder)
    
    # print(f"\nContents of {temp_extract_folder}/:")
    # for item in temp_path.iterdir():
    #     print(f"  - {item.name} ({'folder' if item.is_dir() else 'file'})")
    
    # # Find all CSV files recursively
    # all_csv_files = list(temp_path.rglob('*.csv'))
    # print(f"\nFound {len(all_csv_files)} CSV files total")
    
    # if all_csv_files:
    #     print("\nCSV file locations:")
    #     for csv in all_csv_files:  # Show all
    #         relative_path = csv.relative_to(temp_path)
    #         print(f"  - {relative_path}")
    
    # Step 3: Find and copy CSV files to data/raw
    print(f"\n[2/5] Copying CSV files to {raw_folder}/...")
    csv_source = Path(temp_extract_folder) / 'Dataset_BTS'
    csv_files = list(csv_source.glob('*.csv'))
    
    for csv_file in csv_files:
        destination = raw_path / csv_file.name
        shutil.copy2(csv_file, destination)
        print(f"  !!! Copied: {csv_file.name}")
    
    print(f"\nTotal CSV files copied: {len(csv_files)}")
    
    # Step 4: Delete temporary extraction folder
    print(f"\n[3/5] Cleaning up temporary folder...")
    shutil.rmtree(temp_extract_folder)
    print("!!! Temporary folder deleted")
    
    # Step 5: Process and combine CSV files
    print(f"\n[4/5] Processing CSV files (removing headers)...")
    combined_data = []

    for csv_file in sorted(raw_path.glob('*.csv')):
        print(f"  Processing: {csv_file.name}")
        
        # Read file and skip header lines
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find where actual CSV data starts (after the metadata lines)
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('Carrier Code,Date'):
                data_start_idx = i
                break
        
        # Read the actual CSV data
        df = pd.read_csv(csv_file, skiprows=data_start_idx)
        combined_data.append(df)
        print(f"    !!! {len(df)} records loaded")
    
    # Step 6: Combine and save
    print(f"\n[5/5] Combining all data into single CSV...")
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    output_file = processed_path / 'flight_delays_combined.csv'
    combined_df.to_csv(output_file, index=False)
    
    print("=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\n!!! Combined CSV saved to: {output_file}")
    print(f"!!! Total records: {len(combined_df):,}")
    print(f"!!! Columns: {len(combined_df.columns)}")
    print(f"\nColumn names:")
    for col in combined_df.columns:
        print(f"  - {col}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Run the extraction and processing
    extract_and_process_flight_data()