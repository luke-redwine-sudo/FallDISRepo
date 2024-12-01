import pandas as pd
import math

class GNSSDataMonitor:

    def __init__(self):
        self.direct_dataframe = None
        self.reflected_dataframe = None
        self.reflectivity_dataframe = None
        self.combined_dataframe = None
        self.flight_dataframe = None
        self.satellite_dict = None

    def process_direct_gnss_data(self, file_path, date_string):
        self.direct_dataframe = self.process_raw_gnss_data(file_path, date_string)

    def process_reflected_gnss_data(self, file_path, date_string):
        self.reflected_dataframe = self.process_raw_gnss_data(file_path, date_string)

    def process_flight_data(self, file_path):
        self.flight_dataframe = pd.read_csv(file_path)

    def process_satellite_location_dict(self, satellite_dict):
        self.satellite_dict = satellite_dict

    def add_direct_gnss_data(self, file_path):
        self.direct_dataframe = pd.append(self.direct_dataframe, self.process_raw_gnss_data(file_path))

    def add_reflected_gnss_data(self, file_path):
        self.reflected_dataframe = pd.append(self.reflected_dataframe, self.process_raw_gnss_data(file_path))

    def merge_gnss_data(self):
        # Merge DataFrames
        grouping_tolerance = '500 milliseconds'
        tol = pd.Timedelta(grouping_tolerance)

        # Perform asof merge
        self.reflectivity_dataframe = pd.merge_asof(
            self.direct_dataframe.reset_index(),
            self.reflected_dataframe.reset_index(),
            on='DateTime',
            by='Satellite',
            direction='nearest',
            tolerance=tol
        )

        self.reflectivity_dataframe = self.reflectivity_dataframe.rename(columns={"Signal_Strength_x": "Reflected_Signal_Strength", "Signal_Strength_y": "Direct_Signal_Strength"})
        self.reflectivity_dataframe["Reflectivity"] = self.reflectivity_dataframe["Direct_Signal_Strength"] - self.reflectivity_dataframe["Reflected_Signal_Strength"]
        self.reflectivity_dataframe = self.reflectivity_dataframe.dropna()

        self.reflectivity_dataframe = self.reflectivity_dataframe.drop(columns=['index_x', 'index_y'])

    def write_combined_dataframe(self, file_path):
        self.combined_dataframe.to_csv(file_path + "/combined_dataframe.csv")

    def get_satellites(self):
        return self.reflectivity_dataframe["Satellite"].unique()

    def count_satellite_appearances(self):
        return self.reflectivity_dataframe["Satellite"].value_counts()

    def filter_satellites(self):
        sat_list = list(self.satellite_dict.keys())
        self.reflectivity_dataframe = self.reflectivity_dataframe[self.reflectivity_dataframe.Satellite.isin(sat_list) == True]

    def process_raw_gnss_data(self, file_path, date_string):

        with open(file_path, 'r') as file:
            raw_gnss_data = file.read()

        datetime_segmented_gnss_data = [">" + entries for entries in raw_gnss_data.split(">")][1:]

        gnss_dataframe = pd.DataFrame(columns=['DateTime', 'Satellite', 'Signal_Strength'])
        datetime = ""
        previous_datetime = ""
        tracker_datetime = ""

        for line in datetime_segmented_gnss_data:
            for entry in line.split("\n"):
                if (">" in entry and datetime != ""):
                    date_split = entry.split()
                    tracker_datetime = date_split[2] + "-" + date_split[3] + "-" + date_split[1] + " " + date_split[4] + ":" + date_split[5] + ":" + date_split[6]
                    datetime = str(pd.to_datetime(datetime) + pd.to_timedelta((pd.to_datetime(tracker_datetime) - pd.to_datetime(previous_datetime)).total_seconds(), unit='s'))
                    previous_datetime = tracker_datetime
                elif (">" in entry and datetime == ""):
                    datetime = date_string
                    date_split = entry.split()
                    previous_datetime = date_split[2] + "-" + date_split[3] + "-" + date_split[1] + " " + date_split[4] + ":" + date_split[5] + ":" + date_split[6]
                elif ("G" in entry):
                    gnss_data_split = entry.replace("-", " ").split()
                    if (len(gnss_data_split) == 3):
                        gnss_data_entry = {'DateTime': datetime, 'Satellite': gnss_data_split[0], 'Signal_Strength': gnss_data_split[2]}
                        gnss_dataframe.loc[gnss_dataframe.index.size] = gnss_data_entry
                    elif (len(gnss_data_split) == 4):
                        gnss_data_entry = {'DateTime': datetime, 'Satellite': gnss_data_split[0], 'Signal_Strength': gnss_data_split[3]}
                        gnss_dataframe.loc[gnss_dataframe.index.size] = gnss_data_entry
                    elif (len(gnss_data_split) == 5):
                        gnss_data_entry = {'DateTime': datetime, 'Satellite': gnss_data_split[0], 'Signal_Strength': gnss_data_split[4]}
                        gnss_dataframe.loc[gnss_dataframe.index.size] = gnss_data_entry
                    elif (len(gnss_data_split) == 6):
                        gnss_data_entry = {'DateTime': datetime, 'Satellite': gnss_data_split[0], 'Signal_Strength': gnss_data_split[5]}
                        gnss_dataframe.loc[gnss_dataframe.index.size] = gnss_data_entry

        gnss_dataframe["DateTime"] = pd.to_datetime(gnss_dataframe["DateTime"], format="mixed")
        gnss_dataframe["Signal_Strength"] = pd.to_numeric(gnss_dataframe["Signal_Strength"])

        return gnss_dataframe

    def merge_flight_data(self, time_delta):
        # Merge DataFrames
        grouping_tolerance = '499 milliseconds'
        tol = pd.Timedelta(grouping_tolerance)

        # Convert microseconds to seconds
        self.flight_dataframe['TimeUS'] = self.flight_dataframe['TimeUS'] + time_delta#1726066470914000

        # Convert seconds to datetime and set timezone to UTC
        self.flight_dataframe['DateTime'] = pd.to_datetime(self.flight_dataframe['TimeUS'], unit='us')

        self.combined_dataframe = pd.merge_asof(
            self.reflectivity_dataframe.reset_index(),
            self.flight_dataframe.reset_index(),
            on='DateTime',
            direction='nearest',
            tolerance=tol
        )

        self.combined_dataframe = self.combined_dataframe.dropna()
        self.combined_dataframe = self.combined_dataframe.drop(columns=['index_x', 'index_y'])

        self.combined_dataframe["Alt"] = self.combined_dataframe["Alt"].astype("float32")
        self.combined_dataframe["Adjust_LAT_M"] = 0
        self.combined_dataframe["Adjust_LON_M"] = 0

        for row_index, current_row in self.combined_dataframe.iterrows():
            self.combined_dataframe.at[row_index, "Adjust_LAT_M"] = float(float(current_row["Alt"]) * math.tan(math.radians(float(current_row["Pitch"]))))
            self.combined_dataframe.at[row_index, "Adjust_LON_M"] = float(float(current_row["Alt"]) * math.tan(math.radians(float(current_row["Roll"]))))
            self.combined_dataframe.at[row_index, "Lat"] = self.combined_dataframe.at[row_index, "Lat"] + (self.combined_dataframe.at[row_index, "Adjust_LAT_M"] / 6378000) * (180 / math.pi) / math.cos(self.combined_dataframe.at[row_index, "Lat"] * math.pi / 180)
            self.combined_dataframe.at[row_index, "Lng"] = self.combined_dataframe.at[row_index, "Lng"] + (self.combined_dataframe.at[row_index, "Adjust_LON_M"] / 6378000) * (180 / math.pi)

        self.combined_dataframe["Adjust_LAT_M"] = 0.0
        self.combined_dataframe["Adjust_LON_M"] = 0.0

        for row_index, row in self.combined_dataframe.iterrows():
            SAT_AZIMUTH_ANGLE = float(self.satellite_dict[row["Satellite"]][0])
            SAT_ELEVATION_ANGLE = float(self.satellite_dict[row["Satellite"]][1])

            if (SAT_AZIMUTH_ANGLE < 0):
                SAT_AZIMUTH_ANGLE = float(360 + SAT_AZIMUTH_ANGLE)

            distance_c = float(float(row["Alt"]) * math.tan(math.radians(float(SAT_ELEVATION_ANGLE))))

            if (SAT_AZIMUTH_ANGLE == 0):
                self.combined_dataframe.at[row_index, "Adjust_LAT_M"] = distance_c
            elif (SAT_AZIMUTH_ANGLE == 90):
                self.combined_dataframe.at[row_index, "Adjust_LON_M"] = float(-1.0 * distance_c)
            elif (SAT_AZIMUTH_ANGLE == 180):
                self.combined_dataframe.at[row_index, "Adjust_LAT_M"] = float(-1.0 * distance_c)
            elif (SAT_AZIMUTH_ANGLE == 270):
                self.combined_dataframe.at[row_index, "Adjust_LON_M"] = distance_c
            elif (SAT_AZIMUTH_ANGLE > 0 and SAT_AZIMUTH_ANGLE < 90):
                theta_h = float(90 - SAT_AZIMUTH_ANGLE)
                self.combined_dataframe.at[row_index, "Adjust_LON_M"] = float(-1.0 * float(math.cos(math.radians(theta_h)) * distance_c))
                self.combined_dataframe.at[row_index, "Adjust_LAT_M"] = float(math.sin(math.radians(theta_h)) * distance_c)
            elif (SAT_AZIMUTH_ANGLE > 90 and SAT_AZIMUTH_ANGLE < 180):
                theta_h = float(SAT_AZIMUTH_ANGLE - 90)
                self.combined_dataframe.at[row_index, "Adjust_LON_M"] = float(-1.0 * float(math.cos(math.radians(theta_h)) * distance_c))
                self.combined_dataframe.at[row_index, "Adjust_LAT_M"] = float(-1.0 * float(math.sin(math.radians(theta_h)) * distance_c))
            elif (SAT_AZIMUTH_ANGLE > 180 and SAT_AZIMUTH_ANGLE < 270):
                theta_h = float(270 - SAT_AZIMUTH_ANGLE)
                self.combined_dataframe.at[row_index, "Adjust_LON_M"] = float(math.cos(math.radians(theta_h)) * distance_c)
                self.combined_dataframe.at[row_index, "Adjust_LAT_M"] = float(-1.0 * float(math.sin(math.radians(theta_h)) * distance_c))
            else:
                theta_h = float(SAT_AZIMUTH_ANGLE - 270)
                self.combined_dataframe.at[row_index, "Adjust_LON_M"] = float(math.cos(math.radians(theta_h)) * distance_c)
                self.combined_dataframe.at[row_index, "Adjust_LAT_M"] = float(math.sin(math.radians(theta_h)) * distance_c)

            self.combined_dataframe.at[row_index, "Lat"] = float(float(self.combined_dataframe.at[row_index, "Lat"]) + (float(self.combined_dataframe.at[row_index, "Adjust_LAT_M"]) / 6378000.0) * (180.0 / math.pi) / math.cos(float(self.combined_dataframe.at[row_index, "Lat"]) * math.pi / 180.0))
            self.combined_dataframe.at[row_index, "Lng"] = float(float(self.combined_dataframe.at[row_index, "Lng"]) - (float(self.combined_dataframe.at[row_index, "Adjust_LON_M"]) / 6378000.0) * (180.0 / math.pi))

        self.combined_dataframe = self.combined_dataframe.drop(columns=['Unnamed: 0'])