import os
import glob
import sys
import pandas as pd
import datetime as dt
import numpy as np
import math as ma
from matplotlib import pyplot as plt
from astropy import constants
from scipy import stats
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.signal import savgol_filter, detrend
from scipy.stats.mstats import winsorize
import argparse
import multiprocessing as mp
import os


def observation_id_preparation(observation_file):

    # Import observation data

    df = pd.read_csv(observation_file, header=0, engine='python')
    df = df[['obs_id', 'groupid', 'starttime_utc', 'local_sidereal_time_deg', 'duration',
            'int_time', 'freq_res', 'dataqualityname', 'bad_tiles', 'calibration',
            'calibration_delays', 'center_frequency_mhz', 'channel_center_frequencies_mhz_csv',
            'ra', 'ra_pointing', 'ra_phase_center', 'dec', 'dec_pointing', 'dec_phase_center',
            'deleted_flag', 'good_tiles', 'mode', 'sky_temp', 'stoptime_utc', 'total_tiles', 'gridpoint_name', 'gridpoint_number']]
    df['date'] = df.starttime_utc.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z").date())
    df['date_time'] = df.starttime_utc.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z"))
    df['partition'] = pd.factorize(df['date'])[0] + 1
    df = df[df['partition'] == 1].reset_index(drop=True)

    # Group by (number_days, number_gridpoint)
    grouped = df.groupby(["partition", "gridpoint_number"])["obs_id"].apply(list).reset_index()
    
    # Prepare data for multiprocessing
    task_list = [(row["partition"], row["gridpoint_number"], row["obs_id"]) for _, row in grouped.iterrows()]

    print("Task list has been generated!")

    return task_list


def import_data(obs_list, data_directory, data_type):

    print("In import data, this information will be use: ", obs_list, data_directory, data_type)

    combined_array = None

    if data_type == 'real':
        all_data = []

        for i in range(len(obs_list)):
            f = fits.open(data_directory + "hyperdrive_solutions_%s_noise.fits" %(obs_list[i]))

            data = f['SOLUTIONS'].data[:, :, :, ::2]

            all_data.append(data)

        combined_array = np.concatenate(all_data, axis=0)

    elif data_type == 'imaginary':
        all_data = []

        for i in range(len(obs_list)):
            f = fits.open(data_directory + "hyperdrive_solutions_%s_noise.fits" %(obs_list[i]))

            data = f['SOLUTIONS'].data[:, :, :, 1::2]

            all_data.append(data)

        combined_array = np.concatenate(all_data, axis=0)

    elif data_type == 'amplitude':
        all_data = []

        for i in range(len(obs_list)):
            f = fits.open(data_directory + "hyperdrive_solutions_%s_noise.fits" %(obs_list[i]))

            data = np.abs(f['SOLUTIONS'].data[:, :, :, ::2] + f["SOLUTIONS"].data[:, :, :, 1::2] * 1j)

            all_data.append(data)

        combined_array = np.concatenate(all_data, axis=0)

    print("All data has been collected!", combined_array)

    return combined_array


def exponential_weighting_outlier_detection_3d(obs_day, grid, obs_list, data_directory, results_directory, number_of_timeblocks, integration_time, data_type, sigma, max_iters=100, tol=1e-6, epsilon=1e-8):

    """
    Implements the exponential weighting algorithm for robust variance estimation and outlier detection
    for 3D data (time, antenna, frequency).

    Parameters:
    - data: 3D numpy array, input data of shape (time, antenna, frequency).
    - max_iters: int, maximum number of iterations for convergence.
    - tol: float, tolerance for convergence.
    - epsilon: float, minimum variance to avoid division by zero.
    - k: float, threshold for Z-score to classify outliers (default: 3-sigma rule).

    Returns:
    - flag_array: 3D numpy array of the same shape as `data`, where 1 indicates an outlier and 0 otherwise.
    - stats: Dictionary containing robust mean and variance for each (antenna, frequency) combination.
    """

    # Generate data
    data = import_data(obs_list, data_directory, data_type)

    k = sigma
    time, antennas, frequencies, polarizations = data.shape
    flag_array = np.zeros_like(data, dtype=int)  # Initialize flag array
    stats = {}  # To store mean and variance for each (antenna, frequency)
    all_z_scores = {}

    print("Exponential weighting process ...")

    # Iterate over antennas and frequencies
    for polarization in range(polarizations):
        for antenna in range(antennas):
            for frequency in range(frequencies):

                # Extract the time series for the current antenna and frequency
                time_series = data[:, antenna, frequency, polarization]
                time_series = np.nan_to_num(time_series, nan=np.nanmedian(time_series))

                # Step 1: Initialization
                mu_r = np.median(time_series)  # Robust initial guess for the mean
                sigma_r_squared = np.var(time_series) if np.var(time_series) > epsilon else epsilon  # Initial variance

                for iteration in range(max_iters):
                    # Step 2: Compute weights based on current mean and variance
                    q = (time_series - mu_r)**2 / sigma_r_squared
                    weights = np.exp(-q / 4)

                    # Step 3: Update mean
                    new_mu_r = np.sum(time_series * weights) / np.sum(weights)

                    # Step 4: Update variance
                    new_sigma_r_squared = (3/2) * (np.sum((time_series - new_mu_r)**2 * weights) / np.sum(weights))
                    new_sigma_r_squared = max(new_sigma_r_squared, epsilon)  # Ensure variance is positive

                    # Step 5: Check for convergence
                    if abs(new_mu_r - mu_r) < tol and abs(new_sigma_r_squared - sigma_r_squared) < tol:
                        break

                    # Update for the next iteration
                    mu_r, sigma_r_squared = new_mu_r, new_sigma_r_squared

                # Store robust statistics
                stats[(antenna, frequency, polarization)] = {'mean': mu_r, 'variance': sigma_r_squared}

                # Step 6: Compute Z-scores and identify outliers
                sigma_r = np.sqrt(sigma_r_squared)
                z_scores = (time_series - mu_r) / sigma_r
                outliers = np.abs(z_scores) > k
                
                # Store all data and z_score for further analysis
                all_z_scores[(antenna, frequency, polarization)] = {'data': time_series, 'z_score':z_scores, 'obs_id':np.repeat(obs_list, time/len(obs_list), axis=0)}

                # Update flag array
                flag_array[:, antenna, frequency, polarization] = outliers.astype(int)

    # Export results to the csv file
    outlier_indices = np.argwhere(flag_array == 1)

    # Data and z score
    print("Saving the z score data ...")

    df_z_scores = pd.DataFrame(all_z_scores).T.reset_index()
    df_z_scores = df_z_scores.rename(columns={'level_0': 'tile', 'level_1': 'frequency', 'level_2': 'polarization'})
    df_z_scores.to_parquet(results_directory+"all_z_scores_data_day_%s_grid_%s_integration_%s_%s.parquet" %(obs_day, grid, integration_time, data_type), engine="pyarrow", compression="snappy")
    
    # Outliers statistics
    print("Saving the outlier statistics data ...")

    df_outlier_statistics = pd.DataFrame(stats).T.reset_index()
    df_outlier_statistics = df_outlier_statistics.rename(columns={'level_0': 'tile', 'level_1': 'frequency', 'level_2': 'polarization'})
    df_outlier_statistics.to_parquet(results_directory+"outlier_statistics_day_%s_grid_%s_integration_%s_%s.parquet" %(obs_day, grid, integration_time, data_type), engine="pyarrow", compression="snappy")
    
    # Outliers location if the threshold fixed
    print("Saving the outliers location data ...")

    df_obs_id = pd.DataFrame(np.repeat(obs_list, number_of_timeblocks, axis=0)).reset_index()
    df_obs_id = df_obs_id.rename(columns={0: 'obs_id'})

    df_outlier_indices = pd.DataFrame(outlier_indices)
    df_outlier_indices = df_outlier_indices.rename(columns={0: 'index', 1:'tile', 2:'frequency', 3:'polarization'})
    df_outlier_indices['obs_id'] = df_outlier_indices['index'].map(df_obs_id.set_index('index')['obs_id'].to_dict())
    df_outlier_indices.to_parquet(results_directory+"outlier_locations_day_%s_grid_%s_integration_%s_%s.parquet" %(obs_day, grid, integration_time, data_type), engine="pyarrow", compression="snappy")
    
    return print("All results files has been generated!")


def running(data_directory, results_directory, observation_file, number_of_timeblocks, integration_time, data_type, sigma):
    task_list = observation_id_preparation(observation_file)

    # Run multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(exponential_weighting_outlier_detection_3d, [(obs_days, grid, obs_list, data_directory, results_directory, number_of_timeblocks, integration_time, data_type, sigma) for obs_days, grid, obs_list in task_list])
    
    # Print results
    #for res in results:
    #    print(res)


def main():
    """
    Several things to take a note here are:
    1. The program designed for all polarization MWA data [XX, XY, YX, YY]
    2. The data taken from CSV file that downloaded previously with the column defined bellow
    """

    parser = argparse.ArgumentParser(description="Process observation data in parallel.")
    parser.add_argument("-d", "--data_directory", required=True, help="Directory containing hyperdrive calibration solutions")
    parser.add_argument("-r", "--results_directory", required=True, help="Directory to store the calculation results")
    parser.add_argument("-o", "--obs_file", required=True, help="Directory containing the observation id file")
    parser.add_argument("-nt", "--number_of_timeblocks", type=int, required=True, help="Number of timeblocks")
    parser.add_argument("-it", "--integration_time", type=int, required=True, help="Integration time per each timeblock second ex. 2, 8")
    parser.add_argument("-s", "--sigma", type=int, required=True, help="Minimum standard deviation to calculate the outliers")
    parser.add_argument("-t", "--type", required=True, help="The data type that consist of amplitude, real and imaginary part")

    args = parser.parse_args()

    data_directory = args.data_directory #"/Users/eormacstudio/Documents/20240821_multiple_simulation_higher_thermal_noise_sn1/uvfits/calibrated_fits/"
    results_directory = args.results_directory
    observation_file = args.obs_file #"/Users/eormacstudio/Documents/winsorized_statistics/python_script/general_eor1_update_241010.csv"

    number_of_timeblocks = args.number_of_timeblocks
    integration_time = args.integration_time
    sigma = args.sigma
    data_type = args.type

    running(data_directory, results_directory, observation_file, number_of_timeblocks, integration_time, data_type, sigma)


if __name__ == "__main__":
    main()