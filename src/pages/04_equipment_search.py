
import streamlit as st
import pandas as pd
from utils.data_loader import load_equipment_csv

# --- Data Loading and Initialization ---
st.cache_data
def load_data():
    df, EQUIPMENT_FILE_PATH = load_equipment_csv() # This line is correct
    # This function should load your data and return the DataFrame
    return df, EQUIPMENT_FILE_PATH # This function is returning (df, path)

df, CSV_FILE_PATH = load_data() 

if "full_df" not in st.session_state:
    try:
        # Load the full DataFrame using the cached function
        st.session_state.full_df = df
        
        # Get the file path separately for saving later
        st.session_state.csv_file_path = CSV_FILE_PATH
        
        # Initialize session state for filters only if data loads successfully
        st.session_state.filters = {col: [] for col in st.session_state.full_df.columns}
        
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Function to save changes to the CSV file
def save_full_dataframe(edited_df, file_path):
    """Saves the entire edited DataFrame back to the CSV file."""
    # Write the entire DataFrame back to the path
    edited_df.to_csv(file_path, index=False)
    # Update the session state copy
    st.session_state.full_df = edited_df.copy()

    # Reset filters to clear any invalid selections from the previous data state (due to removal of rows)
    for col in st.session_state.filters:
        st.session_state.filters[col] = []
    # ---------------------------------------
    st.success("Data successfully saved to CSV! ✅")
    st.rerun() 

# --- Streamlit UI and Logic ---

st.title("Equipment Data Management")

# Sidebar Mode Selector
st.sidebar.header("Application Mode")
app_mode = st.sidebar.radio(
    "Select Mode", 
    ("View & Filter Data", "Edit Equipment Data"), 
    index=0 # Default to view mode
)

# Use the full data from session state
df = st.session_state.full_df.copy()

# Define preferred column order (Your existing logic)
preferred_order = ['Train', 'Process Stream', 'system', 'Equipment ID', 'function', 'parameter', 'Value']
remaining_columns = sorted([col for col in df.columns if col not in preferred_order])
ordered_columns = preferred_order + remaining_columns
df = df[[col for col in ordered_columns if col in df.columns]]


# --- EDIT RAW DATA MODE ---
if app_mode == "Edit Equipment Data":
    st.subheader("Edit Original CSV Data")
    st.warning("⚠️ Edits in this table will directly modify the original CSV file upon saving.")
    
    # Display the full DataFrame in an editable editor
    edited_df_full = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic", # Allow users to add new rows
        key="full_data_editor"
    )

    # Check if the edited data is different from the original data
    if not edited_df_full.equals(df):
        st.info("You have unsaved changes in the full dataset.")
        
        # Button to save the changes
        if st.button("💾 Save All Changes to CSV", type="primary"):
            try:
                # Call the dedicated save function
                save_full_dataframe(edited_df_full,CSV_FILE_PATH)
                st.success("Data successfully saved to CSV! ✅")
                # Re-run the app to refresh the data and state
                st.rerun() 
            except Exception as e:
                st.error(f"An error occurred while saving: {e}")
                st.exception(e)
    else:
        st.info("No changes detected. The original file is safe.")


# --- VIEW & FILTER DATA MODE (Your Existing Logic) ---
elif app_mode == "View & Filter Data":
    st.sidebar.header("Filter Options")

    # Re-run your existing filter creation and application logic
    selected_filters = {}
    for col in ordered_columns:
        temp_df=df.copy()
        # Apply other column filters to temp_df to determine options for current col
        for other_col in ordered_columns:
            if other_col != col and st.session_state.filters[other_col]:
                temp_df = temp_df[temp_df[other_col].isin(st.session_state.filters[other_col])]
        options = sorted(temp_df[col].dropna().unique())

        selected = st.sidebar.multiselect(f"Filter by {col}", options, default=st.session_state.filters[col])
        st.session_state.filters[col] = selected

        if selected:
            selected_filters[col] = selected

    # Apply all selected filters to the DataFrame
    filtered_df = df.copy()
    for col, values in selected_filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(values)]

    # Display the filtered data (read-only)
    st.subheader("Filtered Data (Read-Only)")
    st.dataframe(filtered_df, use_container_width=True)


# how dataframe being filtered down every time a filter is applied
# 🧪 Scenario: No Filters Selected
# Let’s say:

# ordered_columns = ['Train', 'Process Stream', 'system']
# st.session_state.filters = {'Train': [], 'Process Stream': [], 'system': []}
# Say you are going ot use the filter for "Train" column:
# Now the inner loop runs:
# other_col = 'Train'
# Skipped because other_col == col

# other_col = 'Process Stream'
# st.session_state.filters['Process Stream'] is empty → condition fails → skipped

# other_col = 'system'
# Also empty → skipped

# as soon as the use selects "A" from "Train" filter, you get st.session_state.filters['Train'] = ['A']
# then app reruns, streamlit reruns the script from top to bottom, and loop begins:
# Now you're rendering filters again, starting with col = 'Train'.
# Since no other filters are selected yet, temp_df remains the full df. options = temp_df['Train'].dropna().unique() gives ['A',"B"]
# The next is col = process stream, it doesnt equal to other_col = 'Train' and other_col has filter selected. process stream also does not equal to equipment ID, or systems... and they dont have any selections.
# So temp_df is now only includes rows where Train = "A".
