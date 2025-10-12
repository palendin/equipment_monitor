import streamlit as st
import pandas as pd
import os
from utils.pi_vision_web_api import PIWebAPI # Assuming this library is installed
import numpy as np
import os
import plotly.graph_objects as go
from utils.data_loader import tags_map

# --- Configuration and Constants ---

# INITIALIZE SESSION STATE HERE for df_pivoted. This ensures 'df_pivoted' always exists, even on the very first script run.
if 'df_pivoted' not in st.session_state:
    # Initialize with an empty DataFrame to avoid the AttributeError
    st.session_state.df_pivoted = pd.DataFrame() 

# Define the structure for the tag file lookup. assumes tag files are in a 'tags/' directory and have a column 'tag_name' and 'tag_parameter_name'.
TAG_FILE_MAP = tags_map()
# Available unit operations for the dropdown
UNIT_OPERATIONS = list(TAG_FILE_MAP.keys())

@st.cache_data(show_spinner=True) # Cache the data retrieval for efficiency
def fetch_and_process_data(unit_op, time_ranges,acquisition_time_interval):
    """
    Fetches PI data for the selected unit operation and time ranges,
    then processes and pivots the data.
    """
    
    if unit_op not in TAG_FILE_MAP:
        st.error(f"Configuration error: Tag file not defined for {unit_op}.")
        return pd.DataFrame() # Return empty DataFrame on error

    tag_file_path = TAG_FILE_MAP[unit_op]

    if not os.path.exists(tag_file_path):
        st.error(f"Tag file not found: {tag_file_path}. Please check your 'tags' directory.")
        return pd.DataFrame()

    # 1. Load Tag File
    try:
        tag_file = pd.read_excel(tag_file_path, sheet_name="Sheet1", engine="openpyxl")
        tag_paths = tag_file["tag_name"].tolist()
        tag_name_no_prefix = tag_file["tag_name"].str.split("\\").str[-1]
        tag_to_param = dict(zip(tag_name_no_prefix, tag_file["tag_parameter_name"]))
    except Exception as e:
        st.error(f"Error loading tag file {tag_file_path}: {e}")
        return pd.DataFrame()
    
    # Check for valid time ranges
    # st.info(time_ranges)
    valid_time_ranges = [tr for tr in time_ranges if tr[0] and tr[1]]
    if not valid_time_ranges:
        return pd.DataFrame() # No data to fetch

    # 2. Initialize PI API
    try:
        # NOTE: The original script doesn't show API setup (e.g., server URL, authentication). 
        # Assuming PIWebAPI() initializes correctly based on environment/defaults.
        pi_api = PIWebAPI() 
    except Exception as e:
        st.error(f"Failed to initialize PIWebAPI: {e}")
        return pd.DataFrame()

    # 3. Collect Data
    all_data = []
    
    for start_time, end_time in valid_time_ranges:
        for path in tag_paths:
            try:
                # Get point info
                point_info = pi_api.points_getbypath(path)
                webid = point_info["WebId"]
                tag_name = point_info["Name"]
                
                # Get recorded data (sampled at 60s)
                recorded_data = pi_api.get_recorded_data(webid, start_time, end_time, acquisition_time_interval) # for example "60s"
                
                for item in recorded_data.get("Items", []):
                    raw_value = item.get("Value")
                    final_value = raw_value
                    
                    # Attempt to convert and round numeric values
                    try:
                        final_value = round(float(raw_value), 2)
                    except (ValueError, TypeError):
                        # Keep original string/value if conversion fails
                        pass

                    all_data.append({
                        "tag": tag_name,
                        "timestamp": item.get("Timestamp"),
                        "value": final_value
                    })
            except Exception as e:
                st.warning(f"Failed to retrieve data for tag **{path}** from {start_time} to {end_time}: {e}")

    if not all_data:
        st.info("No data was retrieved for the selected parameters and time ranges.")
        return pd.DataFrame()

    # 4. Process Data
    df = pd.DataFrame(all_data)

    # Map tag name to parameter name
    df['parameter_name'] = df['tag'].map(tag_to_param)

    # Convert timestamp to datetime (UTC) and then localize to Los Angeles
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_local"] = df["timestamp"].dt.tz_convert("America/Los_Angeles").dt.strftime("%Y-%m-%d %H:%M:%S")

    # Pivot the data
    df_pivoted = df.pivot_table(
        index=["timestamp_local", "timestamp"],
        columns="parameter_name",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Convert input of timestamp_local to datetime (originally in string format)
    dt_ranges = []
    for start_str, end_str in time_ranges:
        dt_ranges.append([
            pd.to_datetime(start_str),
            pd.to_datetime(end_str)
        ])

    # create a new column for assigning unique id in case if batch id does not exist
    df_pivoted["unique_id"] = ""
    df_pivoted["timestamp_local"] = pd.to_datetime(df_pivoted["timestamp_local"])
    for i, time in enumerate(dt_ranges):
        df_pivoted.loc[(df_pivoted['timestamp_local'] >= time[0]) & (df_pivoted['timestamp_local'] <= time[1]), "unique_id"] = f'T{i+1}'

    df_pivoted["time_mins"] = df_pivoted.groupby("unique_id")["timestamp_local"].transform(lambda x: ((x - x.min()).dt.total_seconds() / 60).round(1))
    
    return df_pivoted


# plotting. user will define batch id, axis names
def plot(df, batch_ids, xaxis_name,yaxis_name, unique_ids,template):

    # create list of yaxis string for multiple yaxis plots 
    yaxis_num = []
    for count in range(len(yaxis_name)):
        count += 1
        string = 'y'+ str(count)
        yaxis_num.append(string)

    # Ensure selected axes are valid
    if not xaxis_name and yaxis_name:
        # In a real app, you might raise an error or display a specific message
        return go.Figure().add_annotation(text="Invalid axis selection.", showarrow=False)
    
    # replace all NaN with None
    data = df.where(pd.notnull(df), None)

    # Create plot
    fig = go.Figure()

    # decide to filter with batch or unique id
    if unique_ids and batch_ids:
        st.error("Please select either batch id or unique id")

    if batch_ids:
        for i, batch_id in enumerate(batch_ids):

            # iterate over each y-variable
            for j, yaxis in enumerate(yaxis_name):

                # filtered x and y based on batch_id, xaxis_name, yaxis_name
                filtered_data = data[data['batch id'] == batch_id]
                x = filtered_data[xaxis_name]
                y = filtered_data[yaxis]

                fig.add_trace(go.Scatter(name = f'{batch_id} {yaxis}', x=x, y=y, yaxis=yaxis_num[j]))
    
    if unique_ids:
        for i, unique_id in enumerate(unique_ids):

            # iterate over each y-variable
            for j, yaxis in enumerate(yaxis_name):

                # filtered x and y based on batch_id, xaxis_name, yaxis_name
                filtered_data = data[data['unique_id'] == unique_id]
                x = filtered_data[xaxis_name]
                y = filtered_data[yaxis]

                fig.add_trace(go.Scatter(name = f'{unique_id} {yaxis}', x=x, y=y, yaxis=yaxis_num[j]))

   
    # create dictionary for multi axis plot
    args_for_update_layout = dict()
    for i, yaxis in enumerate(yaxis_name):
        key_name = 'yaxis' if i ==0 else f'yaxis{i+1}'
        if i == 0:
            yaxis_args = dict(title=yaxis_name[0])
        else:
            yaxis_args = dict(title=yaxis, anchor = 'free', overlaying =  'y', side = 'left', autoshift = True)
        
        # populate the dictionary
        args_for_update_layout[key_name] = yaxis_args
        #print(args_for_update_layout)


    # update layout using yaxis dictionary.
    fig.update_layout(**args_for_update_layout)
    fig.update_layout(template=template)

    return fig

# --- Streamlit Application Layout (Main Screen) ---
st.set_page_config(layout="wide", page_title="Multi-Batch Process Plotter")




# --- Sidebar for User Input ---
st.sidebar.header("Input Parameters")

# 1. Unit Operation Selection
user_select_unit_operation = st.sidebar.selectbox(
    "1. Select Unit Operation:",
    UNIT_OPERATIONS
)

# 2. Time Range Inputs
st.sidebar.subheader("2. Define Time Ranges (Start and End)")
st.sidebar.markdown("_Enter datetime strings (e.g., `9/27/2025 22:46:00`)_")

with st.sidebar:
    # Define number of time range entries
    num_entries = st.number_input(
        "Number of entries to add:",
        min_value=1,
        max_value=10,
        value=1,
        key = "num_entries1",
    )

    time_ranges_input = []
    for i in range(int(num_entries)):
        col1, col2 = st.columns(2)

        with col1:
            start_time = st.text_input(f"Start Time {i+1}", key=f"start_{i}",value="6/6/2025 0:17:00" if i == 0 else "")
        with col2:
            end_time = st.text_input(f"End Time {i+1}", key=f"end_{i}",value="6/6/2025 1:17:00" if i == 0 else "")
        
        if start_time or end_time: # Only append if the user has started filling it out
            time_ranges_input.append([start_time, end_time])
    # Filter out empty or incomplete ranges
    valid_time_ranges = [
        [start, end] for start, end in time_ranges_input 
        if start and end
    ]

    acquisition_time_interval = st.selectbox("Select data acquisition time interval",
            options=['3s','5s','10s','15s','30s','60s','300s','3600s']
    )

# button trigger, which controls when and how data fetching process starts
if st.sidebar.button("Fetch and Process Data"):
    if not valid_time_ranges:
        st.warning("Please define at least one complete **Start Time** and **End Time** range.")
    else:
        st.info(f"Fetching data for **{user_select_unit_operation}** across **{len(valid_time_ranges)}** time range(s)...")
        
        # Call the data retrieval function and store as a session
        df_result = fetch_and_process_data(user_select_unit_operation, valid_time_ranges, acquisition_time_interval)
        st.session_state.df_pivoted = df_result

        if not df_result.empty:
            st.success("Data retrieval and processing complete! 🎉")
        else:
            st.error("Data processing failed or returned an empty dataset. Check the log for warnings/errors.")

# main body (plotting section). Note the order matters because this depends on df_pivoted
st.title("Interactive Multi-Batch Data Plotter")
st.markdown("Select one or more batches and the process variables you wish to compare.")

# 🔑 NECESSARY RETRIEVAL: Pull the persisted data from session state for use in the main body
df_plot = st.session_state.df_pivoted
if df_plot.empty:
    st.warning("Please configure your parameters in the sidebar and click **Fetch and Process Data** to load data first.")
else:
    ## Processed Data
    # st.header("Processed and Pivoted Data")
    # st.write(df_plot)
    
    # Download Button
    csv = df_plot.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f'pi_data_{user_select_unit_operation}.csv',
        mime='text/csv',
    )
    
    ## Data Head (Preview)
    st.subheader("Data Preview (First 5 Rows)")
    st.dataframe(df_plot.head(), use_container_width=True)

    all_unique_ids = df_plot['unique_id'].unique()
    try:
        all_batch_ids = sorted(df_plot['batch id'].astype(str).unique())
    except:
        all_batch_ids = None
    all_columns = df_plot.columns.unique()

    # --- User Inputs (on the main screen) ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Allow selection of multiple batch IDs
        selected_batch_ids = st.multiselect(
            "Select Batch IDs for comparison (Min 1)",
            options=all_batch_ids,
            #default=[all_batch_ids[0]] if all_batch_ids else []
        )

    with col2:
        # Select the X-axis variable
        selected_xaxis = st.selectbox(
            "Select X-Axis Variable",
            options=all_columns,
        )

    with col3:
        # Select the Y-axis variable
        selected_yaxis = st.multiselect(
            "Select Y-Axis Variable",
            options=all_columns,
        )
    
    with col4:
        # Select the Y-axis variable
        unique_ids = st.multiselect(
            "Select Unique ID if Batch ID is not used",
            options=all_unique_ids,
        )
    
    # select a template
    template = st.selectbox("Select a plot background template",options=['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon', 'none'], index=10)

    st.markdown("---")

    # --- Plot Generation and Display ---
    if not (selected_xaxis and selected_yaxis and (selected_batch_ids or unique_ids)): # This is the crucial change
        st.warning("Please select axis, and least one Batch ID to generate the plot.")
    else:
        # Call the modified plotting function
        fig = plot(df_plot, selected_batch_ids, selected_xaxis, selected_yaxis, unique_ids,template)
        
        # Display the plot using st.plotly_chart
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Showing comparison for {selected_yaxis} against {selected_xaxis}.")

# Optional: Display the raw data for transparency
with st.expander("View Raw Data"):
    st.dataframe(df_plot)


# add legend
# add multi y axis
# add this to the tff valve app in another page