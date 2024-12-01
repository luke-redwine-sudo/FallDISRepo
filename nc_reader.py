import xarray as xr
import pandas as pd # only needed if you want to do pandas stuff
dataset = xr.open_dataset('C:\\Users\\redwi\Downloads\\NSIDC-0795_L1B_D_20241012_v1.0.nc', engine="netcdf4")
print(dataset)

# Convert xarray.Dataset to pandas.DataFrame
df = dataset.to_dataframe().reset_index()

# Save to CSV
df.to_csv("output.csv", index=False)

print("Dataset saved to CSV.")
