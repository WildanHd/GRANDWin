import pandas as pd
import numpy as np
import datetime as dt
import h5py
import math as ma
from astropy.io import fits
from astropy.time import Time
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from pyuvdata import UVData


df = pd.read_csv("/KUMA/kuma6/whidayat/python_script/general_eor1_update_241010.csv", header=0, engine='python')
df['date'] = df.starttime_utc.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z").date())
df['date_time'] = df.starttime_utc.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z"))
df['partition'] = pd.factorize(df['date'])[0] + 1
df = df[df['partition'] == 1].reset_index(drop=True)
df = df[df['gridpoint_number'] == 0].reset_index(drop=True)

POL_ORDER = ['XX', 'XY', 'YX', 'YY']
pol_to_plot = 'XX'  # ← customize this

obs_id = df['obs_id'].tolist()

all_data = []

for i in range(len(obs_id)):
    # No flags data
    with_no_flags = fits.open("/KUMA/kuma6/MWA/data/eor1_2014/%s/%s_w_no_flags059-078.uvfits" %(obs_id[i], obs_id[i]))
    date_unique, unique_id = np.unique(with_no_flags[0].data['DATE'], return_inverse=True)

    ## Reshape data
    reshaped_no_flags = with_no_flags[0].data.data.reshape(
        len(date_unique),
        len(np.unique(with_no_flags[0].data['BASELINE'])),
        with_no_flags[0].data.data.shape[3],
        with_no_flags[0].data.data.shape[4],
        with_no_flags[0].data.data.shape[5])

    ## Extract weights and flags
    weights_no_flags = reshaped_no_flags[..., 2]    # shape: (56, 8128, 640, 4)
    flags_no_flags = weights_no_flags < 0   # boolean array of shape (56, 8128, 640, 4)

    ## Reorder axes: (time, frequency, baseline, polarization)
    flags_no_flags = np.transpose(flags_no_flags, (0, 2, 1, 3)) # shape: (56, 640, 8128, 4)

    ## Select polarization
    pol_idx = POL_ORDER.index(pol_to_plot)
    flags_pol_no_flags = flags_no_flags[..., pol_idx]   # shape: (56, 640, 8128)

    ## Sum over baselines
    flag_counts_no_flags = np.sum(flags_pol_no_flags, axis=2)   # shape: (56, 640)

    ## Normalize by total number of baselines
    total_possible_no_flags = flags_pol_no_flags.shape[2]
    occupancy_no_flags = flag_counts_no_flags / total_possible_no_flags # shape: (56, 640)

    # With flags
    with_flags = fits.open("/KUMA/kuma6/MWA/data/eor1_2014/%s/%s_w_flags_corrections_time_edges059-078_wins_flags_allbl_5.uvfits" %(obs_id[i], obs_id[i]))

    reshaped_with_flags = with_flags[0].data.data.reshape(
        len(date_unique),
        len(np.unique(with_flags[0].data['BASELINE'])),
        with_flags[0].data.data.shape[3],
        with_flags[0].data.data.shape[4],
        with_flags[0].data.data.shape[5])

    ## Extract weights and flags
    weights_with_flags = reshaped_with_flags[..., 2] 
    print("weights shape: ", weights_with_flags.shape)
    flags_with_flags = weights_with_flags < 0

    ## Reorder axes: (time, frequency, baseline, polarization)
    flags_with_flags = np.transpose(flags_with_flags, (0, 2, 1, 3))
    print("flagss shape: ", flags_with_flags.shape)

    ## Select polarization
    pol_idx = POL_ORDER.index(pol_to_plot)
    flags_pol_with_flags = flags_with_flags[..., pol_idx]
    print("flags_pol shape: ", flags_pol_with_flags.shape)

    ## Sum over baselines
    flag_counts_with_flags = np.sum(flags_pol_with_flags, axis=2)
    print("flags_counts shape: ", flag_counts_with_flags.shape)

    ## Normalize by total number of baselines
    total_possible_with_flags = flags_pol_with_flags.shape[2]
    occupancy_with_flags = flag_counts_with_flags / total_possible_with_flags
    occupancy_total = occupancy_with_flags - occupancy_no_flags

    # Combine data
    all_data.append(occupancy_total)

all_data_combined = np.concatenate(all_data, axis=0)

# Create plot

# No flags data
with_no_flags = fits.open("/KUMA/kuma6/MWA/data/eor1_2014/%s/%s_w_no_flags059-078.uvfits" %(obs_id[i], obs_id[i]))

# Extract metadata
total_channels = with_no_flags[0].header["NAXIS4"]
center_channel = int(with_no_flags[0].header["CRPIX4"])
center_freq_mhz = with_no_flags[0].header["CRVAL4"]*10**-6
channel_spacing_mhz = with_no_flags[0].header["CDELT4"]*10**-6

# Calculate first and last frequencies
start_freq = center_freq_mhz - (center_channel * channel_spacing_mhz)
end_freq = center_freq_mhz + ((total_channels - center_channel - 1) * channel_spacing_mhz)

# Create a custom colormap that maps NaN to black
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='black')

# Plot
plt.figure(figsize=(14, 6))
plt.imshow(
    all_data_combined,
    aspect='auto',
    interpolation='none',
    cmap=cmap,
    extent=[start_freq, end_freq, 56*len(obs_id), 0]
)
plt.colorbar(label="Flag occupancy (fraction)")
plt.xlabel("Frequency channel (MHz)")
plt.ylabel("Time index")
plt.title("Flag Occupancy")
plt.tight_layout()
plt.savefig("occupancy_multiple_obs_id.png")
plt.show()