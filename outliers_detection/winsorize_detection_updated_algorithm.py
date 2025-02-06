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


def winsorized_statistics_outlier_detection_3d(obs_day, grid, obs_list, data_directory, results_directory, number_of_timeblocks, integration_time, data_type, sigma, max_iters=100, tol=1e-6, epsilon=1e-8):

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
                print(all_z_scores)
                if len(df_data[~df_data['%s' %(polr[l])].isna()]) != 0:

                    # Select the best gamma based on the most occurence of outliers

                    tgamma = []
                    outliers_num = []

                    for m in range(iteration_number+1):
                        gamma = np.round(((m+1) * step), 3)
                        b = df_data[~df_data['%s' %(polr[l])].isna()].sort_values(by='%s' %(polr[l]), ascending = True).reset_index(drop=True)

                        b = stats.mstats.winsorize(b['%s' %(polr[l])], limits=[gamma/2, gamma/2])
                        b = pd.DataFrame(data=b.data, columns=['amplitude'])

                        # Calculate winsorized statistics and z score of the data

                        win_mean = np.mean(b['amplitude'])
                        win_std = np.std(b['amplitude'])
                        df_data['z_score_mean'] = (df_data['%s' %(polr[l])] - win_mean) / win_std

                        outliers_num.append(len(df_data[(df_data['z_score_mean'] <= -outlier_threshold) | (df_data['z_score_mean'] >= outlier_threshold)]))
                        tgamma.append(gamma)

                    dfg = pd.DataFrame({"gamma": tgamma, "outliers_number": outliers_num})
                    dfg['count'] = dfg['outliers_number'].map(dfg.groupby('outliers_number').size())
                    #dfg = dfg[dfg['outliers_number'] != 0].reset_index(drop=True)
                    dfg = dfg[dfg['count'] == np.max(dfg['count'])].reset_index(drop=True)

                    print("Gamma dataframe: ", dfg)

                    if len(dfg) != 0:
                        change_indices = [n for n in range(1, len(dfg['outliers_number'])) if dfg['outliers_number'][n] != dfg['outliers_number'][n-1]]

                        if len(change_indices) != 0:
                            gamma_final = dfg['gamma'][change_indices[0]-1]
                        else:
                            gamma_final = dfg['gamma'].iloc[-1]
                    else:
                        gamma_final = 0.

                    print(gamma_final)

                    # Recalculate the number of outliers using final gamma and store the outliers data

                    b = df_data[~df_data['%s' %(polr[l])].isna()].sort_values(by='%s' %(polr[l]), ascending=True).reset_index(drop=True)
                    b = stats.mstats.winsorize(b['%s' %(polr[l])], limits=[gamma_final/2, gamma_final/2])
                    b = pd.DataFrame(data=b.data, columns=['amplitude'])

                    # Calculate winsorized statistics, z score of the data, and select the outliers from the data outside the outlier threshold

                    win_mean = np.mean(b['amplitude'])
                    win_std = np.std(b['amplitude'])
                    df_data['z_score_mean'] = (df_data['%s' %(polr[l])] - win_mean) / win_std
                    c = df_data[(df_data['z_score_mean'] <= -outlier_threshold) | (df_data['z_score_mean'] >= outlier_threshold)].reset_index(drop = True)
                    c['polr'] = polr[l]

                    print("Outliers: ", c)

                    c.to_csv(result_directory+file_outliers, mode='a', index=False, header=False)

                    e = str(i) + ', ' + str(j) + ', ' + str(polr[l]) + ', ' + str(stats.skew(df_data[~df_data['%s' %(polr[l])].isna()]['%s' %(polr[l])])) + ', ' + str(np.nanmax(df_data['%s' %(polr[l])])) + ', ' + str(np.nanmin(df_data['%s' %(polr[l])])) + ', ' + str(np.nanmean(df_data['%s' %(polr[l])])) + ', ' + str(win_mean) + ', ' + str(np.nanstd(df_data['%s' %(polr[l])]))  + ', ' + str(win_std) + ', ' + str(gamma_final) + ', ' + str(len(c)) + '\n'

                    print(e)

                    file_stats = open(result_directory+file_statistics, mode='a', newline='\n')
                    file_stats.write(e)
                    file_stats.close()
                else:
                    continue

    return print("All results files has been generated!")


def running(data_directory, results_directory, observation_file, number_of_timeblocks, integration_time, data_type, sigma):
    task_list = observation_id_preparation(observation_file)

    # Run multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(winsorized_statistics_outlier_detection_3d, [(obs_days, grid, obs_list, data_directory, results_directory, number_of_timeblocks, integration_time, data_type, sigma) for obs_days, grid, obs_list in task_list])


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