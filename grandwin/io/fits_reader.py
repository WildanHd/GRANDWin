# grandwin/io/fits_reader.py
# This script designed to read the fits file of the gain calibration solutions data and it depends on the data type

import numpy as np
from astropy.io import fits

def import_data(obs_list, data_directory, data_type):
    all_data = []

    for obs_id in obs_list:
        f = fits.open(f"{data_directory}hyperdrive_solutions_{obs_id}_noise.fits")

        if data_type == 'real':
            data = f['SOLUTIONS'].data[:, :, :, ::2]
        elif data_type == 'imaginary':
            data = f['SOLUTIONS'].data[:, :, :, 1::2]
        elif data_type == 'amplitude':
            data = np.abs(f['SOLUTIONS'].data[:, :, :, ::2] + f["SOLUTIONS"].data[:, :, :, 1::2] * 1j)
        elif data_type == 'phase':
            data = np.rad2deg(np.angle(f['SOLUTIONS'].data[:, :, :, ::2] + f["SOLUTIONS"].data[:, :, :, 1::2] * 1j))
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        all_data.append(data)

    combined_array = np.concatenate(all_data, axis=0)
    return combined_array
