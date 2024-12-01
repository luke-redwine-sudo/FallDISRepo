from bokeh.io import show, curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Button, Slider, LinearColorMapper, ColorBar, LinearAxis
from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, ESRI_IMAGERY
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256  # Make sure to import the palette
import pandas as pd

from ml import main

import numpy as np
import time
import GNSSDataMonitor

import tkinter as tk
from tkinter import filedialog

print("Loading Model")
random = main()

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
for index, row in df.iterrows():
    timestamp = row['DateTime']  # Replace with the actual timestamp column name
    lat = row['Lat']
    lon = row['Lng']
    reflected_signal_strength = row["Reflectivity"]  # Get the signal strength
    if timestamp not in data_by_time:
        data_by_time[timestamp] = []
    data_by_time[timestamp].append((lat, lon, reflected_signal_strength))

# Prepare the path for animation
timestamps = list(data_by_time.keys())
mercator_paths = {ts: [latlon_to_mercator(lat, lon) + (float(strength),) for lat, lon, strength in points] for ts, points in data_by_time.items()}

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
plot.circle(x="x", y="y", size=10, fill_color=transform('strength', color_mapper), fill_alpha=0.8, source=source)

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


    points_to_display = []
    for i in range(index[0] + 1):  # Include all previous points
        timestamp = timestamps[i]
        points_to_display.extend(mercator_paths[timestamp])

    # Update the data source with all points up to the current index
    new_data = dict(x=[point[0] for point in points_to_display],
                    y=[point[1] for point in points_to_display],
                    strength=[float(random.predict(np.array(point[2]).reshape(-1,1))[0]) for point in points_to_display])  # Include strength for color
    source.data = new_data

slider.on_change('value', update_slider)

def update():
    current_time = time.time() * 1000  # Current time in milliseconds
    if paused[0]:
        return  # Do not update if the animation is paused

    if index[0] < len(timestamps):
        timestamp = timestamps[index[0]]
        points = mercator_paths[timestamp]

        # Update the data source with all points for the current timestamp
        new_data = dict(x=[point[0] for point in points],
                        y=[point[1] for point in points],
                        strength=[float(random.predict(np.array(point[2]).reshape(-1,1))[0]) for point in points])  # Include strength for color
        print(new_data["strength"])
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

# Add periodic callback to update the plot
curdoc().add_periodic_callback(update, 1000)  # Update every 1000 ms

# Arrange layout and add to document
curdoc().add_root(column(pause_button, slider, plot))

# Display the plot in a browser
show(plot)