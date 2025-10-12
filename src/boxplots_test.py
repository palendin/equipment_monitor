import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import dash
import plotly.graph_objects as go

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

# manual test
product = "MK3"
step = "UF1"
df_filtered = df[(df["Product"] == product) & (df["step"] == step)]
retentate_valve_open = "Retentate valve % open"
retentate_resistance = "retentate Resistance (psi/LPM)"
filtrate_valve_open = "filtrate valve % open"
filtrate_resistance = "filtrate Resistance (psi/LPM)"

# x = df_filtered[df_filtered['parameter'] == retentate_valve_open]['value'].reset_index(drop=True)
# y = df_filtered[df_filtered['parameter'] == retentate_resistance]['value'].reset_index(drop=True)
x1 = df_filtered[df_filtered['parameter'] == filtrate_valve_open]['value'].reset_index(drop=True)
y1 = df_filtered[df_filtered['parameter'] ==  filtrate_resistance]['value'].reset_index(drop=True)


def boxplot_stats(x,y):
    mean = np.mean(y)
    std = np.std(y)
    min = np.min(y)
    max = np.max(y)

    # 99.7% confidence interval
    # ci = 2.97 * std / np.sqrt(len(y))
    # lower_ci = mean - ci
    # upper_ci = mean + ci

    return mean, min, max


fig = go.Figure()

if len(x1.unique()) == 1:

    fig.add_trace(go.Box(
    y=y1,
    name='Group A (Low)',
    marker_color='#ff7f0e',
    boxpoints=False,  # Show all points (can be 'outliers', False, etc.)
    ))
        
     #Update Layout (optional but recommended)
    fig.update_layout(
        title='Box Plot Comparison using go.Figure',
        yaxis_title='Value Distribution',
        showlegend=True,
        plot_bgcolor='white'
    )

fig.show()

# add boxplot to the base figure function
# initialize the model base on the unique x datapoints, whether its box or linear regression
# see if i can integrate into the plot ci band function
