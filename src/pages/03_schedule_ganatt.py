import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go

def run_simulation(upstream, downstream, ds_tank_cleaning_time, up_tank_cleaning_time):
    upstream_steps = list(upstream.keys())
    downstream_steps = list(downstream.keys())
    step_order = upstream_steps + downstream_steps

    bioreactor_days = []
    for i in upstream.values():
        total = i['setup'] + i['downtime1'] + i['operation'] + i['downtime2'] + i['cleaning'] + i['downtime3']
        bioreactor_days.append(total)

    bioreactor_days = min(bioreactor_days) / 24
    num_of_bioreactors = len(upstream.keys())
    cadence = int(bioreactor_days / num_of_bioreactors)
    st.write(f"Bioreactor Cadence (gap time between runs): {round(bioreactor_days/num_of_bioreactors,1)}")

    num_cycles = len(upstream_steps)
    st.write(f"Number of cycles (based on max utilization): {num_cycles}")

    schedule = []
    last_clean_end = {s: 0.0 for s in step_order}
    last_downstream_end = 0

    for i, bioreactor in enumerate(upstream_steps):
        upstream_info = upstream[bioreactor]
        ideal_br_start = i * cadence * 24

        br_setup_start = ideal_br_start
        br_setup_end = br_setup_start + upstream_info['setup']
        br_op_start = br_setup_end + upstream_info['downtime1']
        br_op_end = br_op_start + upstream_info['operation']
        br_clean_start = br_op_end + upstream_info['downtime2']
        br_clean_end = br_clean_start + upstream_info['cleaning']
        br_end = br_clean_end + upstream_info['downtime3']

        schedule.append({'task': f'{bioreactor} Setup (Cycle {i+1})', 'start': br_setup_start, 'end': br_setup_end, 'row': bioreactor})
        schedule.append({'task': f'{bioreactor} Operation (Cycle {i+1})', 'start': br_op_start, 'end': br_op_end, 'row': bioreactor})
        schedule.append({'task': f'{bioreactor} Cleaning (Cycle {i+1})', 'start': br_clean_start, 'end': br_clean_end, 'row': bioreactor})

        for tank in upstream_info['tanks']:
            t_start = br_setup_start
            t_end = t_start + up_tank_cleaning_time[tank]
            schedule.append({'task': f'{tank} Cleaning (Cycle {i+1})', 'start': t_start, 'end': t_end, 'row': tank})

        last_clean_end[bioreactor] = br_end

        current_downstream_start = max(br_op_end, last_downstream_end)
        for step in downstream_steps:
            info = downstream[step]
            ds_setup_start = current_downstream_start - info['setup']
            ds_setup_end = ds_setup_start + info['setup']
            ds_op_start = ds_setup_end + info['downtime1']
            ds_op_end = ds_op_start + info['operation']
            ds_clean_start = ds_op_end + info['downtime2']
            ds_clean_end = ds_clean_start + info['cleaning']
            ds_end = ds_clean_end + info['downtime3']

            schedule.append({'task': f'{step} Setup (Cycle {i+1})', 'start': ds_setup_start, 'end': ds_setup_end, 'row': step})
            schedule.append({'task': f'{step} Operation (Cycle {i+1})', 'start': ds_op_start, 'end': ds_op_end, 'row': step})
            schedule.append({'task': f'{step} Cleaning (Cycle {i+1})', 'start': ds_clean_start, 'end': ds_clean_end, 'row': step})

            for tank in info['tanks']:
                t_start = ds_setup_start
                t_end = t_start + ds_tank_cleaning_time[tank]
                schedule.append({'task': f'{tank} Cleaning (Cycle {i+1})', 'start': t_start, 'end': t_end, 'row': tank})

            last_clean_end[step] = ds_end
            current_downstream_start = ds_op_end
            last_downstream_end = current_downstream_start

        
    cycle_durations = []
    for i in range(len(upstream_steps)):
        # Find all tasks for this cycle
        cycle_tasks = [item for item in schedule if f"(Cycle {i+1})" in item['task']]
        if cycle_tasks:
            cycle_start = min(task['start'] for task in cycle_tasks)
            cycle_end = max(task['end'] for task in cycle_tasks)
            cycle_durations.append(cycle_end - cycle_start)

    if cycle_durations:
        longest_cycle_duration = max(cycle_durations)
        max_runs = 365 / (longest_cycle_duration / 24)
    else:
        max_runs = 0

    st.write('Max number of runs per year', round(max_runs))

    rows = sorted(set(item['row'] for item in schedule), key=lambda r: (r not in step_order, r))
    fig = go.Figure()
    added_to_legend = set()

    for item in schedule:
        if 'Tank' in item['task']:
            color = 'gray'
            task_name = 'Other Setup'
        else:
            color_type = item['task'].split()[1]
            if color_type == "Setup":
                color = 'lightblue'
                task_name = 'Setup'
            elif color_type == "Operation":
                color = 'lightgreen'
                task_name = 'Operation'
            else:
                color = 'salmon'
                task_name = 'Cleaning'

        show_legend = task_name not in added_to_legend
        if show_legend:
            added_to_legend.add(task_name)

        fig.add_trace(go.Bar(
            y=[item['row']],
            x=[item['end'] - item['start']],
            base=[item['start']],
            orientation='h',
            marker=dict(color=color),
            name=task_name,
            showlegend=show_legend,
            hovertext=[f"{item['task']}<br>Start: {item['start']}<br>End: {item['end']}"],
            width=0.8
        ))

    fig.update_layout(
        title="Pipeline Schedule",
        xaxis_title="Time (hours)",
        yaxis_title="Steps",
        height=max(400, 30 * len(rows))
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title("Manufacturing Scheduling App")

upstream_step_order = st.text_input("Enter order of upstream production bioreactor (comma separated)", "bioreactor1,bioreactor2,bioreactor3").split(",")
downstream_step_order = st.text_input("Enter order of downstream unit operation (comma separated)", "affinity,AEX,CEX").split(",")

upstream_steps = {}
for i, step in enumerate(upstream_step_order):
    with st.expander(f"⚙️ Define Upstream Step: {step}"):
        setup = st.number_input(f"Setup time for {step}", value=5.0, key=f"{step}_setup")
        downtime1 = st.number_input(f"Downtime after setup for {step}", value=1.0, key=f"{step}_downtime1")
        operation = st.number_input(f"Operation time for {step}", value=200.0, key=f"{step}_operation")
        downtime2 = st.number_input(f"Downtime after operation for {step}", value=1.0, key=f"{step}_downtime2")
        cleaning = st.number_input(f"Cleaning time for {step}", value=2.0, key=f"{step}_cleaning")
        downtime3 = st.number_input(f"Downtime after cleaning for {step}", value=1.0, key=f"{step}_downtime3")
        tanks = st.text_input(f"Peripheral equipment used by {step} (comma separated)", f"Feed_Tank{i}", key=f"{step}_tanks")
        tanks = [t.strip() for t in tanks.split(",") if t.strip()]
        upstream_steps[step] = {
            "setup": setup, "downtime1": downtime1, "operation": operation,
            "downtime2": downtime2, "cleaning": cleaning, "downtime3": downtime3, "tanks": tanks
        }

with st.expander("🥄 Define Upstream Peripheral Equipment Cleaning Times"):
    up_tank_cleaning_time = {}
    all_tanks = sorted({tank for s in upstream_steps.values() for tank in s["tanks"]})
    for tank in all_tanks:
        ttime = st.number_input(f"Cleaning time for {tank}", value=2.0, key=f"upstream_{tank}_time")
        up_tank_cleaning_time[tank] = ttime

downstream_steps = {}
for i, step in enumerate(downstream_step_order):
    with st.expander(f"⚙️ Define Downstream Step: {step}"):
        setup = st.number_input(f"Setup time for {step}", value=5.0, key=f"{step}_setup")
        downtime1 = st.number_input(f"Downtime after setup for {step}", value=1.0, key=f"{step}_downtime1")
        operation = st.number_input(f"Operation time for {step}", value=10.0, key=f"{step}_operation")
        downtime2 = st.number_input(f"Downtime after operation for {step}", value=1.0, key=f"{step}_downtime2")
        cleaning = st.number_input(f"Cleaning time for {step}", value=2.0, key=f"{step}_cleaning")
        downtime3 = st.number_input(f"Downtime after cleaning for {step}", value=1.0, key=f"{step}_downtime3")
        tanks = st.text_input(f"Peripheral equipment used by {step} (comma separated)", f"Hold_Tank{i+1}", key=f"{step}_tanks")
        tanks = [t.strip() for t in tanks.split(",") if t.strip()]
        downstream_steps[step] = {
            "setup": setup, "downtime1": downtime1, "operation": operation,
            "downtime2": downtime2, "cleaning": cleaning, "downtime3": downtime3, "tanks": tanks
        }

with st.expander("🥄 Define Downstream Peripheral Equipment Cleaning Times"):
    ds_tank_cleaning_time = {}
    all_tanks = sorted({tank for s in downstream_steps.values() for tank in s["tanks"]})
    for tank in all_tanks:
        ttime = st.number_input(f"Cleaning time for {tank}", value=2.0, key=f"downstream_{tank}_time")
        ds_tank_cleaning_time[tank] = ttime

if st.sidebar.button("Generate"):
    run_simulation(upstream_steps, downstream_steps, ds_tank_cleaning_time, up_tank_cleaning_time)
