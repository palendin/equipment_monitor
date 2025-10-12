import streamlit as st
import os
from pathlib import Path

# --- 1. CONFIGURATION ---

st.set_page_config(
    page_title="MSAT APPLICATION Platform",
    page_icon="🚀",
    layout="wide"
)

# --- 2. CUSTOM CSS (Layout & Styling) ---

st.markdown("""
    <style>
    /* 1. Hides the Streamlit default page navigation in the sidebar */
    section[data-testid="stSidebar"] div.stRadio {
        display: none;
    }

    /* 2. Hide the default sidebar, but keep the space for a better layout */
    section[data-testid="stSidebar"] {
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
    }
    
    /* 3. Style the horizontal radio buttons (Our Custom Nav) */
    div[data-testid="stRadio"] > label {
        padding: 8px 16px;
        border-radius: 8px; /* Rounded corners */
        margin: 0 5px; /* Spacing between links */
        transition: background-color 0.2s, color 0.2s;
    }

    /* 4. Style the container to look like a clean header */
    .st-emotion-cache-1pxn4x0 { /* Target the container holding the nav */
        padding-top: 0;
        padding-bottom: 0;
    }

    /* 5. Style the selected (active) link */
    div[data-testid="stRadio"] > label[data-baseweb="radio"] input[type="radio"]:checked + span {
        background-color: #0E7673; /* A strong accent color for active state */
        color: white;
        font-weight: 600;
        border-bottom: 3px solid #FF4B4B; /* Optional: bottom highlight */
    }

    /* 6. Style unselected links (hover effect) */
    div[data-testid="stRadio"] > label[data-baseweb="radio"]:not([data-baseweb="radio"] input[type="radio"]:checked + span):hover {
        background-color: rgba(14, 118, 115, 0.1); /* Light hover effect */
        color: #0E7673;
        cursor: pointer;
    }
    
    /* 7. Ensure content has good top padding after the nav bar */
    .block-container {
        padding-top: 3rem;
    }
    
    </style>
    """, unsafe_allow_html=True)


# --- 3. CUSTOM TOP NAVIGATION BAR LOGIC ---

# Define the links based on your pages folder structure
# CRITICAL FIX: Use the base name 'main' for the root script for reliable navigation.
PAGES_MAP = {
    "Home": "main", # Fixed: Use 'main' (the base name) for reliable switch_page to the root.
    "PI data export and visualization": "pages/01_pi_data_export.py", 
    "TFF valve status check": "pages/02_tff_valve_status.py",
    "Bioprocess Scheduling Tool": "pages/03_schedule_ganatt.py",
    "Equpiment Search":"pages/04_equipment_search.py"
}

# Get the current Streamlit app page identifier (file stem)
def get_current_page_stem():
    """Gets the base file name (stem) of the currently active page file (e.g., 'main' or '01_pi_data_export')."""
    try:
        # Use st.source_script_path().stem to get 'main' or '01_pi_data_export'
        return Path(st.source_script_path()).stem
    except:
        # Fallback for the main page if st.source_script_path() is unavailable
        return Path(__file__).stem 

# --- Draw the Navigation Bar ---

# Use a container for the navigation bar to give it a clean, full-width look
with st.container():
    # Use st.columns for minor layout control (e.g., centering the radio)
    col1, col_nav, col2 = st.columns([1, 5, 1]) 

    with col_nav:
        
        # 1. Determine the currently active page name for initial selection
        current_page_stem = get_current_page_stem()
        
        # Find the friendly name corresponding to the current file stem
        page_names = list(PAGES_MAP.keys())
        active_name = "Home" # Default to Home
        
        # Find which friendly name corresponds to the current page stem
        for name, identifier in PAGES_MAP.items():
            if Path(identifier).stem == current_page_stem:
                 active_name = name
                 break
        
        # Determine the default index based on the found active name
        try:
            active_index = page_names.index(active_name)
        except ValueError:
            active_index = 0

        # 2. Use st.radio for the horizontal top bar navigation
        selected_page_name = st.radio(
            "Navigation",
            page_names,
            index=active_index,
            key="top_nav_radio",
            horizontal=True,
            label_visibility="hidden"
        )
        
        # 3. --- Execute Navigation ---
        
        # Get the target identifier (e.g., 'main' or 'pages/01_pi_data_export.py')
        target_identifier = PAGES_MAP[selected_page_name]
        
        # Determine the target page stem for comparison (e.g., 'main' or '01_pi_data_export')
        target_page_stem = Path(target_identifier).stem
        
        # Check if the selection is different from the current page
        if target_page_stem != current_page_stem:
            try:
                # Use Streamlit's built-in navigation
                # This call now correctly handles both full paths (for sub-pages)
                # and the base name 'main' (for the root page).
                st.switch_page(target_identifier) 
            except Exception as e:
                # Log error if switch fails (useful for debugging in Streamlit environments)
                st.error(f"Failed to switch page: {e}")
                
# Add a visual separator below the navigation
st.markdown("---")

# --- 4. MAIN PAGE CONTENT ---
st.header("🚀 MSAT App Platform")
st.markdown("Welcome to MSAT central hub for Bioprocess Tools. Use the navigation bar above to quickly access critical tools.")

st.markdown("""
<div style="padding: 15px; border-radius: 8px; background-color: #f0f8ff; border-left: 5px solid #0E7673;">
    <h4 style="margin-top: 0; color: #0E7673;">Navigation Instructions</h4>
    <p>The top bar replaces the standard Streamlit sidebar navigation. Simply click on a link to instantly switch between applications:</p>
    <ul>
        <li><strong>Home:</strong> Returns you to this dashboard overview.</li>
        <li><strong>PI data export and visualization:</strong> Access tools for historical data analysis.</li>
        <li><strong>TFF valve status check:</strong> Real-time process checks.</li>
        <li><strong>Bioprocess Scheduling Tool:</strong> Gantt chart scheduling tool.</li>
        <li><strong>Equpiment Search:</strong> Search and find equipment specifications.</li>
    </ul>
</div>
""", unsafe_allow_html=True)