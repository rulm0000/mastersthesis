import pandas as pd
import os

def convert_sas_to_csv():
    """
    Reads a SAS dataset (.sas7bdat) and converts it to a CSV file.

    Input:
    - combinedbrfss_18_23v9.sas7bdat: Expected in the same directory as this script.

    Output:
    - combinedbrfss_18_23v9.csv: Will be saved in the same directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sas_file_name = 'combinedbrfss_18_23v9.sas7bdat'
    csv_file_name = 'combinedbrfss_18_23v9.csv'

    sas_file_path = os.path.join(script_dir, sas_file_name)
    csv_file_path = os.path.join(script_dir, csv_file_name)

    print(f"Attempting to read SAS file: {sas_file_path}...")
    try:
        # It's good practice to specify encoding if known, otherwise pandas tries to infer.
        # For BRFSS, 'iso-8859-1' or 'latin1' is common, but sometimes it might be 'utf-8'.
        # If encoding issues arise, this might need to be adjusted.
        df = pd.read_sas(sas_file_path, format='sas7bdat', encoding='iso-8859-1')
        print(f"Successfully read {sas_file_name}.")
    except FileNotFoundError:
        print(f"Error: SAS file not found at {sas_file_path}")
        print(f"Please ensure '{sas_file_name}' is in the same directory as this script.")
        return
    except Exception as e:
        print(f"An error occurred while reading the SAS file: {e}")
        return

    print(f"Attempting to write CSV file: {csv_file_path}...")
    try:
        df.to_csv(csv_file_path, index=False)
        print(f"Successfully converted and saved data to {csv_file_path}")
        print(f"The CSV file contains {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"An error occurred while writing the CSV file: {e}")
        return

if __name__ == '__main__':
    convert_sas_to_csv()
