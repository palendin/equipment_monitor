import pandas as pd
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression



def linear_regression(x, y):

    # Reshape x for sklearn
    x_reshaped = x.values.reshape(-1, 1)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)

    # Calculate residuals and standard error
    residuals = y - y_pred
    n = len(x)
    se = np.sqrt(np.sum(residuals**2) / (n - 2))

    ci = 0.95
    # Calculate t-value for 95% confidence (calculate for 2 tail)
    t_value = stats.t.ppf((1+ci)/2, df=n - 2)

    # Calculate confidence interval
    x_mean = np.mean(x)
    
    # This is a single number representing the spread of the training data.
    sum_of_squares_x = np.sum((x - x_mean)**2) 
    
    se_pred = se * np.sqrt(1+1/n + (x - x_mean)**2 / sum_of_squares_x) # prediction interval has a 1+1/n+ term, CI its just 1/n + term. this is good when you want to know where a single new observation is likely to fall into
    ci = t_value * se_pred
    upper_ci = y_pred + ci
    lower_ci = y_pred - ci

    return model, y_pred, upper_ci, lower_ci, t_value, x_mean, se_pred, se, n, sum_of_squares_x


def boxplot_stats(x,y):
    mean = np.mean(y)
    std = np.std(y)
    min = np.min(y)
    max = np.max(y)

    # 99.7% confidence interval
    # ci = 2.97 * std / np.sqrt(len(y))
    # lower_ci = mean - ci
    # upper_ci = mean + ci
    model = None
    return model, mean, min, max


def check_point_against_box_plots(model, x_new, y_new, mean, min, max):
        
    if min <= y_new <= max:
        status = "good"  # Point falls within the Confidence Band
    else:
        status = "bad"   # Point falls outside the Confidence Band
    
    return mean, status


# given x_new, the original model will predict a value, along with upper and lower CI band. then they will be compared the user's y_new
def check_point_against_ci_band(model, x_new, y_new, se, n, t_value, x_mean, sum_of_squares_x):
    
    
    # 1. Calculate the Predicted Mean (Center of the CI Band at x_new)
    x_new_reshaped = np.array([x_new]).reshape(-1, 1) 
    y_new_predicted_mean = model.predict(x_new_reshaped)[0]

    # 2. Calculate the Standard Error of the Mean Prediction (SE_CI) at x_new location (recall this is prediction interval)
    se_ci_new = se * np.sqrt(1 + 1/n + (x_new - x_mean)**2 / sum_of_squares_x)

    
    
    # 3. Calculate CI Bounds (Lower and Upper) at x_new location
    ci_margin = t_value * se_ci_new
    lower_ci_new = y_new_predicted_mean - ci_margin
    upper_ci_new = y_new_predicted_mean + ci_margin
    
    
    # 4. Check if the user's *OBSERVED* y_new falls within the calculated CI bounds
    if lower_ci_new <= y_new <= upper_ci_new:
        status = "good"  # Point falls within the Confidence Band
    else:
        status = "bad"   # Point falls outside the Confidence Band
    
    print(f"Checking point {x_new}, {y_new}")
    print(f"prediction, lower ci, upper ci, status = {y_new_predicted_mean}, {lower_ci_new}, {upper_ci_new},{status}")
    # Also return the bounds for plotting
    return y_new_predicted_mean, lower_ci_new, upper_ci_new, status


def plotting_confidence_interval(x, y,xaxis_name,yaxis_name, product, step):

    # Create plot
    fig = go.Figure()

    if len(x.unique()) == 1:
        fig.add_trace(go.Box(
        x=x, # 
        y=y,
        marker_color="#12a25c",
        boxpoints=False,  # Show all points (can be 'outliers', False, etc.)
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{product} {step} {yaxis_name} vs {xaxis_name}',
            xaxis_title=xaxis_name,
            yaxis_title=yaxis_name,
            template="plotly_white",

            # --- SIZE ADJUSTMENT ---
            width=1000,  # Set the width of the plot in pixels
            height=500  # Set the height of the plot in pixels
        )
         
    else:
        model, y_pred, upper_ci, lower_ci, t_value, x_mean, se_pred, se, n, sum_of_squares_x = linear_regression(x, y)

        # Sort values for in order for confidence interval band to work, since reading from pandas directly does not "sort" the % valves values
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        upper_ci_sorted = upper_ci[sorted_indices]
        lower_ci_sorted = lower_ci[sorted_indices]

        # Regression line
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_pred_sorted,
            mode='lines',
            name='Regression Line',
            line=dict(color='green')
        ))

        # Confidence interval shaded area
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_sorted, x_sorted[::-1]]),
            y=np.concatenate([upper_ci_sorted, lower_ci_sorted[::-1]]),
            fill='toself',
            # fill with a yellow color with some opacity
            fillcolor='rgba(255, 255, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Interval'
        ))

        # Original data points
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Data Points',
            marker=dict(color='white')
        ))

        # Update layout
        fig.update_layout(
            title=f'{product} {step} {yaxis_name} vs {xaxis_name}',
            xaxis_title=xaxis_name,
            yaxis_title=yaxis_name,
            template="plotly_white",

            # --- SIZE ADJUSTMENT ---
            width=1000,  # Set the width of the plot in pixels
            height=500  # Set the height of the plot in pixels
        )

    return fig


# User input

# Load the Excel file
# Get the directory of the current script (main.py)
current_dir = os.path.dirname(__file__)

# Construct the full path to the Excel file
file_path = os.path.join(current_dir, 'flow resistance reference values.xlsx')
df = pd.read_excel(file_path, sheet_name="reference", engine="openpyxl").iloc[10:,:]

# melt the dataframe to pivot so it can be filtered with user input
identifier_columns = df.columns[0:4].tolist()
melt_columns = df.columns[4:].tolist()
df = df.melt(
    id_vars=identifier_columns, 
    value_vars= melt_columns, # Pass the list of columns to melt
    var_name="parameter",        # <-- This MUST be a single string
    value_name="value"
)

# user input for selecting the product and step to generate the base plot
# generate the dataframe base on user selection
try:
    # 1. ATTEMPT FILTERING: The operation that might cause a hard error (e.g., if a column doesn't exist)
    product = st.selectbox("Select a product", df["Product"].unique(), key="product_select")
    step = st.selectbox("Select a step", df["step"].unique(), key="step_select")
    df_filtered = df[(df["Product"] == product) & (df["step"] == step)]

except KeyError as e:
    # 2. CATCH ERROR: If a column is missing (e.g., 'Product' or 'step'), display a clear error message.
    st.error(f"Configuration Error: The required data column '{e}' was not found in the file.")
    st.stop() # Stops execution gracefully after showing the message
    
except NameError:
    # Handle the case where the variable 'df' itself hasn't been defined (e.g., file loading failed).
    st.error("Error: The main data file could not be loaded or found.")
    st.stop()

# 3. CHECK FOR EMPTY DATA: If the filtering succeeded but yielded no results.
if df_filtered.empty:
    # Display a single, user-friendly warning message.
    st.warning(f"No data available for Product: {product} and Step: {step}. Please check your selections.")
    st.stop() # Stops execution gracefully

# Extract x and y values
retentate_valve_open = "Retentate valve % open"
retentate_resistance = "retentate Resistance (psi/LPM)"
filtrate_valve_open = "filtrate valve % open"
filtrate_resistance = "filtrate Resistance (psi/LPM)"

x = df_filtered[df_filtered['parameter'] == retentate_valve_open]['value'].reset_index(drop=True)
y = df_filtered[df_filtered['parameter'] == retentate_resistance]['value'].reset_index(drop=True)
x1 = df_filtered[df_filtered['parameter'] == filtrate_valve_open]['value'].reset_index(drop=True)
y1 = df_filtered[df_filtered['parameter'] ==  filtrate_resistance]['value'].reset_index(drop=True)

# print(x.values.reshape(-1,1))

# base plot
fig = plotting_confidence_interval(x, y, retentate_valve_open,retentate_resistance, product, step)
fig1 = plotting_confidence_interval(x1, y1,filtrate_valve_open,filtrate_resistance, product, step)

# Make Model Setup Conditional on Data using caching
# Streamlit follows a strict rule: It will only execute the code inside this function if one of the function's input parameters has changed since the last run.
@st.cache_resource(ttl=3600) # Use st.cache_resource for models/heavy objects
def setup_models(x_ret, y_ret, x_fil, y_fil):
    # This function will only rerun if any of its five input arguments change
    
    if len(x_ret.unique()) == 1:
        model, mean, min, max = boxplot_stats(x_ret, y_ret)
        ret_setup = {'model':model, 'mean':mean, 'min':min, 'max':max}
    if len(x_fil.unique()) == 1:
        model1, mean1, min1, max1 = boxplot_stats(x_fil, y_fil)
        fil_setup = {'model':model1,'mean':mean1, 'min':min1, 'max':max1}
    if len(x_ret.unique()) > 1:
        # 1. Train Retentate Model
        model, y_pred, upper_ci, lower_ci, t_value, x_mean, se_pred, se, n, sum_of_squares_x = linear_regression(x_ret, y_ret)
        # 1. Train Retentate Model
        model, y_pred, upper_ci, lower_ci, t_value, x_mean, se_pred, se, n, sum_of_squares_x = linear_regression(x_ret, y_ret)
        ret_setup = {'model':model, 'y_pred':y_pred, 'upper_ci':upper_ci, 'lower_ci':lower_ci, 't_value':t_value, 'x_mean':x_mean, 'se_pred':se_pred, 'se':se, 'n':n, 'sum_of_squares_x':sum_of_squares_x}
    if len(x_fil.unique()) > 1:
        # 2. Train Filtrate Model
        model1, y_pred1, upper_ci1, lower_ci1, t_value1, x_mean1, se_pred1, se1, n1, sum_of_squares_x1 = linear_regression(x_fil, y_fil)
        fil_setup = {'model':model1, 'y_pred':y_pred1, 'upper_ci':upper_ci1, 'lower_ci':lower_ci1, 't_value':t_value1, 'x_mean':x_mean1, 'se_pred':se_pred1, 'se':se1, 'n':n1, 'sum_of_squares_x':sum_of_squares_x1}

    return ret_setup, fil_setup


# train once and store constants in session state so no recalculation is needed for every input
with st.spinner('Setting up model and CI band...'):
    retentate_setup, filtrate_setup = setup_models(x, y, x1, y1)

# save the results immediately for easy access
st.session_state.model_setup = retentate_setup
st.session_state.model_setup1 = filtrate_setup

st.success('Model and CI band setup complete!')

# retrieve output from model's session state
retentate_constants = st.session_state.model_setup
filtrate_constants = st.session_state.model_setup1
    
# Now that the session state has been updated, you can retrieve the constants safely in your sidebar sections:
with st.sidebar:
    st.header("Add New Data Points for Retentate Valve")

    # Define number of entries
    num_entries = st.number_input(
        "Number of entries to add:",
        min_value=1,
        max_value=10,
        value=1,
        key = "num_entries1",
    )


    # results = [] # empty array for storing output of check_point_against_ci_band()

    # Add user point based on number of entries, using columns for compression
    for i in range(int(num_entries)):
        st.subheader(f"Retentate Data Point {i+1}")
        
        # Use st.columns to put the two inputs side-by-side
        col1, col2 = st.columns(2) 
        
        with col1:
            new_valve_opening = st.number_input(
                f"% Valve Opening", 
                key=f"valve_{i}", # Unique key is required in loops
                min_value=0.000, 
                max_value=100.0,
                value=50.0, # Add a default value
                step = 0.1,
            )
        
        with col2:
            new_resistance = st.number_input(
                f"Resistance (psi/LPM)", 
                key=f"resistance_{i}", # Unique key is required in loops
                min_value=0.000,
                value=0.020, # Add a default value
                step = 0.001,
                format="%.4f" 
            )

        
        # call the check point function with these inputs: model, valve open, resistance, se, n, t_value, x_mean, sum_of_squares_x
        if retentate_constants['model'] is not None:
            print('using linear model')
            y_new_predicted_mean, lower_ci_new, upper_ci_new, status = check_point_against_ci_band(retentate_constants['model'], new_valve_opening, new_resistance, retentate_constants['se'], retentate_constants['n'], retentate_constants['t_value'], retentate_constants['x_mean'], retentate_constants['sum_of_squares_x'])
        else:
            print('using box plot')
            y_new_predicted_mean, status = check_point_against_box_plots(retentate_constants['model'], new_valve_opening, new_resistance, retentate_constants['mean'], retentate_constants['min'], retentate_constants['max'])

        color = "green" if status == "good" else "red"
        # Display the status of the check point
        st.markdown(
            # Combine everything into one single f-string
            f"**Prediction Status:** :{color}[{status.upper()}]  \n" +
            f"Predicted resistance: {y_new_predicted_mean:.4f}"
        )
        # Store data for later plotting if needed
        # results.append({'x': x_new, 'y': y_new, 'status': status, 'lower_ci': lower_ci, 'upper_ci': upper_ci})


        # Check for valid input and add trace (this logic remains the same)
        if new_resistance > 0 and new_valve_opening > 0:
            fig.add_trace(go.Scatter(
                x=[new_valve_opening],
                y=[new_resistance],
                mode='markers',
                name=f'User Input Point {i+1}', # Give each point a unique name
                marker=dict(color='red', size=10, symbol='x')
            ))

with st.sidebar:
    st.header("Add New Data Points for Filtrate Valve")

    # Define number of entries
    num_entries = st.number_input(
        "Number of entries to add:",
        min_value=1,
        max_value=10,
        value=1,
        key = "num_entries2"
    )

    # Add user point based on number of entries, using columns for compression
    for i in range(int(num_entries)):
        st.subheader(f"Filtrate data point {i+1}")
        
        # Use st.columns to put the two inputs side-by-side
        col1, col2 = st.columns(2) 
        
        with col1:
            new_valve_opening = st.number_input(
                f"% Valve Opening", 
                key=f"filtrate_valve_{i}", # Unique key is required in loops
                min_value=0.000, 
                max_value=100.0,
                value=50.0, # Add a default value
                step = 0.1
            )
        
        with col2:
            new_resistance = st.number_input(
                f"Resistance (psi/LPM)", 
                key=f"filtrate_resistance_{i}", # Unique key is required in loops
                min_value=0.000,
                value=1.0, # Add a default value
                step = 0.001,
                format="%.4f" 
            )

         # call the check point function with these inputs: model, valve open, resistance, se, n, t_value, x_mean, sum_of_squares_x
        if filtrate_constants['model'] is not None:
            print('filtrate using linear model')
            y_new_predicted_mean1, lower_ci_new1, upper_ci_new1, status1 = check_point_against_ci_band(filtrate_constants['model'], new_valve_opening, new_resistance, filtrate_constants['se'], filtrate_constants['n'], filtrate_constants['t_value'], filtrate_constants['x_mean'], filtrate_constants['sum_of_squares_x'])
        else:
            print('filtrate using boxplot model')
            y_new_predicted_mean1, status1 = check_point_against_box_plots(filtrate_constants['model'], new_valve_opening, new_resistance, filtrate_constants['mean'], filtrate_constants['min'], filtrate_constants['max'])

        # Display the status of the check point
        color = "green" if status1 == "good" else "red"
        st.markdown(
            # Combine everything into one single f-string
            f"**Prediction Status:** :{color}[{status1.upper()}]  \n" +
            f"Predicted resistance: {y_new_predicted_mean1:.4f}"
        )
        # Store data for later plotting if needed
        # results.append({'x': x_new, 'y': y_new, 'status': status, 'lower_ci': lower_ci, 'upper_ci': upper_ci})

        # Check for valid input and add trace (this logic remains the same)
        if new_resistance > 0 and new_valve_opening > 0:
            fig1.add_trace(go.Scatter(
                x=[new_valve_opening],
                y=[new_resistance],
                mode='markers',
                name=f'User Input Point for filtrate_valve {i+1}', # Give each point a unique name
                marker=dict(color='red', size=10, symbol='x')
            ))


# show plot
st.plotly_chart(fig)
st.plotly_chart(fig1)

# cache executionf flow:
# 1. User selects product and step -> df_filtered is created
# 2. setup_models() is called, which trains two linear regression models and stores constants in session state
# 3. User inputs new data points in the sidebar -> check_point_against_ci_band() is called for each point using stored model and constants
# 4. if there is no change in product/step selection, the models are not retrained, speeding up the process
# 5. if there is change in product/step, setup_models() reruns to update models and constants, and new results are stored in the cache


# def linear_regression_with_ci(x, y):
#     # Fit linear regression
#     x_mean = np.mean(x)
#     y_mean = np.mean(y)

#     # least square estimate for slope for linear regression
#     m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
#     b = y_mean - m * x_mean

#     # Predicted values
#     y_pred = m * x + b

#     # Residuals and standard error
#     residuals = y - y_pred
#     n = len(x)
#     se = np.sqrt(np.sum(residuals ** 2) / (n - 2))

#     # t-value for 95% CI
#     t = stats.t.ppf(0.975, df=n - 2)

#     # Confidence interval
#     ci = t * se * np.sqrt(1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2))

#     return y_pred, ci