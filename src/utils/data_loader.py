# src/utils/data_loader.py - Focuses ONLY on file path resolution and DataFrame creation.

import pandas as pd
from pathlib import Path
import os
import sys
import openpyxl

# --- 1. Define the Robust Path ---

# Get the absolute path to the current script (data_loader.py)
SCRIPT_PATH = Path(__file__).resolve()

# Navigate up two levels to reach the 'src' directory (assuming structure: src/utils/data_loader.py)
SRC_DIR = SCRIPT_PATH.parent.parent

# Construct the full, absolute path to the data file. This guaranteed path
# resolves correctly even when deployed.
DATA_FILE_PATH = SRC_DIR / "data" / "flow resistance reference values.xlsx"


def load_data_from_excel():
    """
    Loads the Excel data file into a pandas DataFrame using a guaranteed absolute path.
    
    This function contains ZERO Streamlit dependencies.
    
    Returns:
        pandas.DataFrame: The loaded data.
        
    Raises:
        FileNotFoundError: If the data file does not exist.
        Exception: For general reading errors (e.g., incorrect format).
    """
    # 1. Check if the file exists
    if not DATA_FILE_PATH.exists():
        # Raise an exception so the calling code (in 01_dashboard.py) can handle the UI error.
        raise FileNotFoundError(f"Data file not found at: {DATA_FILE_PATH}")

    # 2. Load the data
    try:
        # Use the absolute path
        data = pd.read_excel(DATA_FILE_PATH,sheet_name="reference", engine="openpyxl").iloc[10:,:]
        return data,DATA_FILE_PATH
        
    except Exception as e:
        # Raise a general exception for issues like parsing errors
        raise Exception(f"Failed to read Excel file at {DATA_FILE_PATH}. Error: {e}")
    
def tags_map():
    TAG_SOURCE_FOLDER_PATH = SRC_DIR / "data" / "tags"

    TAG_FILE_MAP = {
    "TFF": TAG_SOURCE_FOLDER_PATH / "3170_pi_tags.xlsx",
    "affinity_chrom": TAG_SOURCE_FOLDER_PATH/"affinity_chrom_pi_tags.xlsx",
    "AEX": TAG_SOURCE_FOLDER_PATH/"aex_pi_tags.xlsx",
    "CEX": TAG_SOURCE_FOLDER_PATH/"cex_pi_tags.xlsx",
    "Zeta-Freezer": TAG_SOURCE_FOLDER_PATH/"zeta_freezer_pi_tags.xlsx",
    }
    
    return TAG_FILE_MAP

def load_BI_PEM():
    PEM_SOURCE_FILE_PATH = SRC_DIR / "utils" / "Boehringer_CAs_Bundle.pem"

    return PEM_SOURCE_FILE_PATH

def load_equipment_csv():
    
    EQUIPMENT_FILE_PATH = SRC_DIR / "data" / "B3_equipments.csv"
    # 1. Check if the file exists
    if not EQUIPMENT_FILE_PATH.exists():
        # Raise an exception so the calling code (in 01_dashboard.py) can handle the UI error.
        raise FileNotFoundError(f"Data file not found at: {EQUIPMENT_FILE_PATH}")

    # 2. Load the data
    try:
        # Use the absolute path
        data = pd.read_csv(EQUIPMENT_FILE_PATH)
        return data, EQUIPMENT_FILE_PATH
        
    except Exception as e:
        # Raise a general exception for issues like parsing errors
        raise Exception(f"Failed to read Excel file at {EQUIPMENT_FILE_PATH}. Error: {e}")
    


