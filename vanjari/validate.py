from pathlib import Path
import json
import os

import pandas as pd

DATA_DIR = Path(__file__).parent/"data"
VALID_NAMES_JSON = DATA_DIR / "MSL39v4_valid_names_per_taxa.json"

def validate_taxonomic_names(input_file:Path, output_file:Path|None=None):
    # Load the JSON file with valid names
    with open(VALID_NAMES_JSON, "r") as f:
        valid_names = json.load(f)
    
    # Submissions have used a mix of tsv and csv, so accounting for this here
    file_extension = os.path.splitext(input_file)[1].lower()
    sep = "\t" if file_extension == ".tsv" else ","
    df = pd.read_csv(input_file, sep=sep)
    
    # Rename columns: remove rank information (e.g., (-viria))
    df.columns = [col.split(" (")[0] for col in df.columns]
    
    # Identify relevant columns (excluding '_score' columns and 'sequenceid')
    columns_to_validate = [col for col in df.columns if not col.endswith("_score") and col != "sequenceid"]
    
    # Store validation failures
    failed_rows = []
    
    for index, row in df.iterrows():
        for col in columns_to_validate:
            if pd.notna(row[col]) and col in valid_names:  # Ignore empty cells
                if row[col] not in valid_names[col]:
                    failed_rows.append({"Row": index + 1, "Column": col, "Invalid Value": row[col]})
    
    # Save failing entries into failure report
    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        print("Validation completed - errors found")
        print(failed_df)
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            failed_df.to_csv(output_file, index=False)
            print(f"Errors saved to {output_file}")
    else:
        print(f"Validation completed - no errors found for {input_file}")