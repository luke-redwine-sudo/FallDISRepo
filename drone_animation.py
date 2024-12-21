from bokeh.io import show, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, Slider, LinearColorMapper, ColorBar, LinearAxis
from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, ESRI_IMAGERY
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256  # Make sure to import the palette
from bokeh.models import HoverTool
import pandas as pd

import ml
import NN
import ts

import numpy as np
import time
import GNSSDataMonitor

import tkinter as tk
from tkinter import filedialog

print("Loading Models...")
print("Random Forest")
print("Decision Tree")
random, decision_tree = ml.main()
print("Neural Network")
neural_network = NN.main()
print("CNN")
cnn = ts.main()


selected_model = {"name": "CNN"}

root = tk.Tk()
root.withdraw()  # Hide the main window

# Process flight data
file_path = filedialog.askopenfilename(filetypes=[("Flight Data", ".csv")])

df = pd.read_csv(file_path)

# Function to convert latitude and longitude to Web Mercator format
def latlon_to_mercator(lat, lon):
    import math
    k = 6378137
    x = lon * (k * math.pi / 180.0)
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) * k
    return x, y

# Create a dictionary to group points by timestamp
data_by_time = {}
for index, row_i in df.iterrows():
    timestamp = row_i['DateTime']  # Replace with the actual timestamp column name
    lat = row_i['Lat']
    lon = row_i['Lng']
    reflected_signal_strength = row_i["Reflectivity"]  # Get the signal strength
    if timestamp not in data_by_time:
        data_by_time[timestamp] = []
    data_by_time[timestamp].append((lat, lon, reflected_signal_strength))

# Prepare the path for animation
timestamps = list(data_by_time.keys())
mercator_paths = {ts: [latlon_to_mercator(lat, lon) + (float(strength),) for lat, lon, strength in points] for ts, points in data_by_time.items()}
POINTS_SO_FAR = []

# Initial map setup
tile_provider = get_provider(ESRI_IMAGERY)
plot = figure(x_axis_type="mercator", y_axis_type="mercator", title="Drone Flight Path", width=800, height=600)

plot.add_tile(tile_provider)

# Hide the axis lines, grid lines, and axis labels
plot.axis.visible = False
plot.grid.visible = False

# Data source for the drone's position and color
source = ColumnDataSource(data=dict(x=[], y=[], strength=[]))

# Create a color mapper
color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)  # Adjust palette as needed
from bokeh.transform import transform

# Plot the drone's position with color mapping
circle_renderer = plot.circle(x="x", y="y", size=10, fill_color=transform('strength', color_mapper), fill_alpha=0.8, source=source)

# Add a HoverTool for displaying strength values
hover_tool = HoverTool(
    renderers=[circle_renderer],
    tooltips=[
        ("Strength", "@strength")
    ]
)

plot.add_tools(hover_tool)

# Add a color bar
color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
plot.add_layout(color_bar, 'right')

# Add a scale (optional) to the right side of the plot
scale = LinearAxis()
plot.add_layout(scale, 'right')

# Animation state variables
index = [0]  # Current timestamp index
reset_interval = 5000  # Time (ms) before resetting the animation
last_reset_time = [None]  # Track the last time the animation was reset
paused = [False]  # Animation pause state

# Pause/Unpause button
pause_button = Button(label="Pause", width=100)

def toggle_pause():
    paused[0] = not paused[0]
    pause_button.label = "Resume" if paused[0] else "Pause"

pause_button.on_click(toggle_pause)

# Slider for scrubbing through timestamps
slider = Slider(start=0, end=len(timestamps) - 1, value=0, step=1, title="Time", width=1200)

def update_slider(attr, old, new):
    index[0] = new

    # Collect all points up to the current timestamp

    #mercator_paths[timestamps[(index[0] + 1)]] = [[list(item)[0], list(item)[1], float(neural_network.predict(np.array(list(item)[2]).reshape(-1,1))[0])] for item in mercator_paths[timestamps[(index[0] + 1)]]]

    print(selected_model["name"])

    global POINTS_SO_FAR

    #POINTS_SO_FAR.append([[list(item)[0], list(item)[1],list(item)[2]] for item in mercator_paths[timestamps[(index[0] + 1)]]])
    #[print([list(item)[0], list(item)[1],list(item)[2]]) for item in mercator_paths[timestamps[(index[0] + 1)]]]

    if selected_model["name"] == "Random Forest":
        mercator_paths[timestamps[(index[0] + 1)]] = [[list(item)[0], list(item)[1], float(random.predict(np.array(list(item)[2]).reshape(-1,1))[0])] for item in mercator_paths[timestamps[(index[0] + 1)]]]
    elif selected_model["name"] == "Decision Tree":
        mercator_paths[timestamps[(index[0] + 1)]] = [[list(item)[0], list(item)[1], float(decision_tree.predict(np.array(list(item)[2]).reshape(-1,1))[0])] for item in mercator_paths[timestamps[(index[0] + 1)]]]
    elif selected_model["name"] == "CNN":
        if len(POINTS_SO_FAR) < 30:
            mercator_paths[timestamps[(index[0] + 1)]] = [[list(item)[0], list(item)[1], 0] for item in mercator_paths[timestamps[(index[0] + 1)]]]
        else:
            # print(np.array(POINTS_SO_FAR[-149:]).shape)
            # [print(np.array(POINTS_SO_FAR[-149:] + [item[2]]).reshape(-1,1).shape) for item in mercator_paths[timestamps[(index[0] + 1)]]]
            mercator_paths[timestamps[(index[0] + 1)]] = [[list(item)[0], list(item)[1], round(float(cnn.predict(np.array(POINTS_SO_FAR[-29:] + [item[2]]).reshape(1,-1))[0]))] for item in mercator_paths[timestamps[(index[0] + 1)]]]
    else:
        #[print(neural_network.predict(np.array(list(item)[2]).reshape(-1,1))[0][0]) for item in mercator_paths[timestamps[(index[0])]]]
        mercator_paths[timestamps[(index[0] + 1)]] = [[list(item)[0], list(item)[1], float(round(neural_network.predict(np.array(list(item)[2]).reshape(-1,1))[0][0]))] for item in mercator_paths[timestamps[(index[0] + 1)]]]

    print(mercator_paths[timestamps[(index[0] + 1)]])
    print("-------------------------------------------------------------------")
    points_to_display = []
    for i in range(index[0] + 1):  # Include all previous points
        timestamp = timestamps[i]
        points_to_display.extend(mercator_paths[timestamp])


    # Update the data source with all points up to the current index
    new_data = dict(x=[point[0] for point in points_to_display],
                    y=[point[1] for point in points_to_display],
                    strength=[point[2] for point in points_to_display])  # Include strength for color

    source.data = new_data

slider.on_change('value', update_slider)

def update():
    current_time = time.time() * 1000  # Current time in milliseconds
    if paused[0]:
        return  # Do not update if the animation is paused

    global POINTS_SO_FAR

    CURRENT_POINTS = []
    for item in mercator_paths[timestamps[(index[0]) + 2]]:
        POINTS_SO_FAR.append(list(item)[2])

    if index[0] < len(timestamps):
        timestamp = timestamps[index[0]]
        if selected_model["name"] == "Random Forest":
            mercator_paths[timestamps[(index[0])]] = [[list(item)[0], list(item)[1], float(random.predict(np.array(list(item)[2]).reshape(-1,1))[0])] for item in mercator_paths[timestamps[(index[0])]]]
        elif selected_model["name"] == "Decision Tree":
            mercator_paths[timestamps[(index[0])]] = [[list(item)[0], list(item)[1], float(decision_tree.predict(np.array(list(item)[2]).reshape(-1,1))[0])] for item in mercator_paths[timestamps[(index[0])]]]
        elif selected_model["name"] == "CNN":
            if len(POINTS_SO_FAR) < 30:
                mercator_paths[timestamps[(index[0] + 1)]] = [[list(item)[0], list(item)[1], 0] for item in mercator_paths[timestamps[(index[0] + 1)]]]
            else:
                # print(np.array(POINTS_SO_FAR[-149:]).shape)
                # [print(np.array(POINTS_SO_FAR[-149:] + [item[2]]).reshape(-1,1).shape) for item in mercator_paths[timestamps[(index[0] + 1)]]]
                mercator_paths[timestamps[(index[0] + 1)]] = [[list(item)[0], list(item)[1], round(float(cnn.predict(np.array(POINTS_SO_FAR[-29:] + [item[2]]).reshape(1,-1))[0]))] for item in mercator_paths[timestamps[(index[0] + 1)]]]
        else:
            [print(neural_network.predict(np.array(list(item)[2]).reshape(-1,1))[0][0]) for item in mercator_paths[timestamps[(index[0])]]]
            mercator_paths[timestamps[(index[0])]] = [[list(item)[0], list(item)[1], float(round(neural_network.predict(np.array(list(item)[2]).reshape(-1,1))[0][0]))] for item in mercator_paths[timestamps[(index[0])]]]

        points = mercator_paths[timestamp]

        # Update the data source with all points for the current timestamp
        new_data = dict(x=[point[0] for point in points],
                        y=[point[1] for point in points],
                        strength=[point[2] for point in points])  # Include strength for color
        source.stream(new_data)

        last_reset_time[0] = current_time  # Update last reset time

        # Update the data source to include all previous points
        points_to_display = []
        for i in range(index[0] + 1):  # Include all previous points
            timestamp = timestamps[i]
            points_to_display.extend(mercator_paths[timestamp])

        source.data = dict(x=[point[0] for point in points_to_display],
                           y=[point[1] for point in points_to_display],
                           strength=[point[2] for point in points_to_display])  # Include strength for color

        index[0] += 1  # Move to the next timestamp
        slider.value = index[0]  # Update slider position
    else:
        # Check if enough time has passed to reset the animation
        if last_reset_time[0] and (current_time - last_reset_time[0] >= reset_interval):
            index[0] = 0
            source.data = dict(x=[], y=[], strength=[])  # Clear the data for reset
            slider.value = index[0]  # Reset slider position
            last_reset_time[0] = None  # Reset the timer


# Callback function for button clicks
def button_callback(event_label):
    print(f"Button clicked: {event_label}")
    selected_model["name"] = event_label

# Create three buttons
button1 = Button(label="Random Forest", width=150)
button2 = Button(label="Decision Tree", width=150)
button3 = Button(label="Neural Network", width=150)
button4 = Button(label="CNN", width=150)

# Attach the same function to all buttons
button1.on_click(lambda: button_callback("Random Forest"))
button2.on_click(lambda: button_callback("Decision Tree"))
button3.on_click(lambda: button_callback("Neural Network"))
button4.on_click(lambda: button_callback("CNN"))

# Arrange the buttons in a row
button_row = row(button1, button2, button3, button4)

# Add periodic callback to update the plot
curdoc().add_periodic_callback(update, 1000)  # Update every 1000 ms

# Arrange layout and add to document
curdoc().add_root(column(column(pause_button, slider, plot), button_row))

# Display the plot in a browser
show(plot)