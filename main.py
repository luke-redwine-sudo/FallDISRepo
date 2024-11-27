import GNSSDataMonitor

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide the main window

data_monitor = GNSSDataMonitor.GNSSDataMonitor()

#file_path = r"C:\Users\redwi\Downloads\ground_0911_GEOP255O.24o"
file_path = filedialog.askopenfilename(filetypes=[("Ground GNSS File",".22o .23o .24o")])
data_monitor.process_direct_gnss_data(file_path, "2024-06-10 17:25:01")

#file_path = r"C:\Users\redwi\Downloads\drone_0911_GEOP255O.24o"
file_path = filedialog.askopenfilename(filetypes=[("Drone GNSS File",".22o .23o .24o")])
data_monitor.process_reflected_gnss_data(file_path, "2024-06-10 17:31:30")

data_monitor.merge_gnss_data()

SATELLITE_DICT = {"G05":[86.9117082895294, 14.8894595579914],
                  "G10":[-67.1243596467499, 18.9862683137934],
                  "G15":[86.9117082895294, 14.8894595579914],
                  "G20":[-67.1243596467499, 18.9862683137934],
                  "G25":[86.9117082895294, 14.8894595579914],
                  "G30":[-67.1243596467499, 18.9862683137934]
                  }

data_monitor.process_satellite_location_dict(SATELLITE_DICT)

#file_path = r"C:\Users\redwi\Downloads\0911_Flight.csv"
file_path = filedialog.askopenfilename(filetypes=[("Flight Data", ".csv")])
data_monitor.process_flight_data(file_path)
data_monitor.filter_satellites()
# spt 16 data_monitor.merge_flight_data(1694853630000000 - 73000000)
# sep 24 data_monitor.merge_flight_data(1695545555000000 - 73000000)
# oct 07 data_monitor.merge_flight_data(1696667198000000 - 73000000)
# oct 20 data_monitor.merge_flight_data(1697788338000000 - 73000000)
# oct 28 data_monitor.merge_flight_data(1698491517000000 - 73000000)
# nov 06 data_monitor.merge_flight_data(1699257694000000 - 73000000)

# mar 03 data_monitor.merge_flight_data(1709451464000000 - 73000000)
# mar 17 data_monitor.merge_flight_data(1710663299000000 - 73000000)
# apr 14 data_monitor.merge_flight_data(1713079782000000 - 73000000)
# apr 30 data_monitor.merge_flight_data(1714462251000000 - 73000000)
# jun 01 data_monitor.merge_flight_data(1717240030000000 - 73000000)
# jun 10
data_monitor.merge_flight_data(1718040777000000 - 73000000)

print(data_monitor.reflectivity_dataframe)
print(data_monitor.flight_dataframe)
print(data_monitor.combined_dataframe)

data_monitor.write_combined_dataframe(str(file_path.rsplit('/', 1)[:-1][0]))
