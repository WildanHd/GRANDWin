import os
import glob
import sys
import pandas as pd
import datetime as dt
import numpy as np
import math as ma
import argparse
import multiprocessing as mp
import os
import h5py
from astropy import constants
from scipy import stats
from astropy.io import fits


def observation_id_preparation(observation_file):

    # Import observation data

    df = pd.read_csv(observation_file, header=0, engine='python')
    df['date'] = df.starttime_utc.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z").date())
    df['date_time'] = df.starttime_utc.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z"))
    df['partition'] = pd.factorize(df['date'])[0] + 1
    df = df[df['partition'] == 1].reset_index(drop=True)

    print(df)

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

    elif data_type == 'phase':
        all_data = []

        for i in range(len(obs_list)):
            f = fits.open(data_directory + "hyperdrive_solutions_%s_noise.fits" %(obs_list[i]))

            data = np.rad2deg(np.angle(f['SOLUTIONS'].data[:, :, :, ::2] + f["SOLUTIONS"].data[:, :, :, 1::2] * 1j))

            all_data.append(data)

        combined_array = np.concatenate(all_data, axis=0)

    print("All data has been collected!", combined_array)

    return combined_array


def winsorizing_vectorizer(data, gamma, threshold):
    """
    Winsorizes data along the time axis based on either a scalar or per-feature gamma.

    Parameters:
    - data: ndarray of shape (time, antennas, frequencies, polarizations)
    - gamma: float or ndarray of shape (antennas, frequencies, polarizations)
    - threshold: float, z-score threshold for outlier detection

    Returns:
    - data_wins: winsorized data, same shape as input
    - outlier_masks: ndarray of shape (antennas, frequencies, polarizations) represent the outlier locations
    - outlier_counts: ndarray of shape (antennas, frequencies, polarizations) represent the number of outlier per location
    """

    time, antennas, frequencies, polarizations = data.shape
    T = time
    N = antennas * frequencies * polarizations

    # Winsorizing the original dataset
    ## Flatten data for easier processing
    data_reshape = data.reshape(T, -1)
    sorted_data = np.sort(data_reshape, axis=0)

    ## Handle gamma: broadcast if scalar, flatten if array
    if np.isscalar(gamma):
        gamma_flat = np.full((N,), gamma)
    else:
        if gamma.shape != (antennas, frequencies, polarizations):
            raise ValueError("gamma must be scalar or have shape (antennas, frequencies, polarizations)")
        gamma_flat = gamma.reshape(-1)

    ## Compute the number of data that should be winsorize per column depends on the gamma
    k_vals = np.floor(gamma_flat / 2 * T).astype(int)

    ## Calculate low and high winsorization values per column
    low_vals = np.array([
        sorted_data[k, i] if k < T else sorted_data[0, i]
        for i, k in enumerate(k_vals)
    ])
    high_vals = np.array([
        sorted_data[-k-1, i] if k < T else sorted_data[-1, i]
        for i, k in enumerate(k_vals)
    ])

    ## Winsorize the original reshaped data
    data_wins = np.clip(data_reshape, low_vals, high_vals)

    ## Calculate winsorized mean and std
    win_mean = data_wins.mean(axis=0)
    win_std = data_wins.std(axis=0)

    ## Compute z-scores and reshape back
    win_z_scores = (data_reshape - win_mean) / win_std
    win_z_scores = win_z_scores.reshape(time, antennas, frequencies, polarizations)

    # Count the number of outliers for each antennas, frequencies, and polarizations
    ## Identify outliers and count
    outlier_masks = (win_z_scores > threshold) | (win_z_scores < -threshold)
    outlier_counts = np.sum(outlier_masks, axis=0)

    return data_wins, win_z_scores, outlier_masks, outlier_counts


def export_data(obs_list, data, data_wins, outlier_masks, outlier_counts):
    """
    This function is used for exporting the data

    Parameters:
    - obs_list: the list of observtaion id that will we process
    - data: ndarray of shape (time, antennas, frequencies, polarizations)
    - data_wins: winsorized data, same shape as input
    - outlier_masks: the location of outliers on the data
    - outlier_counts: the number of outlier for each antenna, frequency, and polarization

    Returns:
    - df_stats: dataframe of statistical properties of the winsorized data
    - df_outlier_counts: dataframe of the number of outlier for each antenna, frequency, and polarization
    - df_outlier_locs: dataframe of the outlier locations
    """

    time, antennas, frequencies, polarizations = data.shape
    T = time
    N = antennas * frequencies * polarizations

    data_reshape = data.reshape(T, -1)

    # Create the data statistical properties
    ## Calculate original data statistical properties
    ori_mean = data.mean(axis=0)
    ori_std = data.std(axis=0)
    ori_max = data.max(axis=0)
    ori_min = data.min(axis=0)
    ori_skew = stats.skew(data, axis=0)

    ## Calculate winsorized mean and std
    data_wins = data_wins.reshape(time, antennas, frequencies, polarizations)
    win_mean = data_wins.mean(axis=0)
    win_std = data_wins.std(axis=0)
    win_skew = stats.skew(data_wins, axis=0)

    print(ori_mean.shape, ori_std.shape, ori_max.shape, ori_min.shape, ori_skew.shape, win_mean.shape, win_std.shape, win_skew.shape)

    ## Flatten all the data
    om_flat = ori_mean.flatten()
    os_flat = ori_std.flatten()
    omax_flat = ori_max.flatten()
    omin_flat = ori_min.flatten()
    osk_flat = ori_skew.flatten()
    wm_flat = win_mean.flatten()
    ws_flat = win_std.flatten()
    wsk_flat = win_skew.flatten()

    ## Create MultiIndex for antenna, frequency, polarization
    ant, freq, pol = ori_mean.shape
    index = pd.MultiIndex.from_product(
        [range(ant), range(freq), range(pol)],
        names=['antenna', 'frequency', 'polarization']
    )

    ## Build statistical properties DataFrame
    df_stats = pd.DataFrame({
        'ori_mean': om_flat,
        'ori_std': os_flat,
        'ori_max': omax_flat,
        'ori_min': omin_flat,
        'ori_skew': osk_flat,
        'winsorized_mean': wm_flat,
        'winsorized_std': ws_flat,
        'winsorized_skew': wsk_flat,
    }, index=index).reset_index()

    # Create outliers count dataframe
    index = pd.MultiIndex.from_product(
        [range(antennas), range(frequencies), range(polarizations)],
        names=["antenna", "frequency", "polarization"]
    )
    df_outlier_counts = pd.DataFrame({
        "outlier_count": outlier_counts.flatten()
    }, index=index).reset_index()

    # Create outliers location on the data
    ## Create observation id and timeblock conversion arrays
    time_to_obs = np.repeat(obs_list, T/len(obs_list))
    time_to_tb = np.tile(np.arange(0,T/len(obs_list), 1), len(obs_list))
    true_indices = np.argwhere(outlier_masks)

    ## Convert to a DataFrame
    df_outlier_locs = pd.DataFrame(
        true_indices,
        columns=['time', 'antenna', 'frequency', 'polarization']
    )
    df_outlier_locs['observation_id'] = df_outlier_locs['time'].map(lambda t: time_to_obs[t])
    df_outlier_locs['timeblock'] = df_outlier_locs['time'].map(lambda t: int(time_to_tb[t]))

    return df_stats, df_outlier_counts, df_outlier_locs


def winsorizing_outlier_detection_3d(obs_day, grid, obs_list, data_directory, results_directory, integration_time, data_type, iter, iter_threshold, final_threshold):

    print("Check all parameters: ", obs_day, grid, obs_list, data_directory, results_directory, integration_time, data_type, iter, iter_threshold, final_threshold)

    # Running winsorizing
    ## Import raw data
    data = import_data(obs_list, data_directory, data_type)

    ## Select the best gamma value
    ### Generate outlier counts based on set of gamma values
    gamma_test = np.linspace(0.0, 0.5, int(iter))

    time, antennas, frequencies, polarizations = data.shape

    combined_outlier_counts = np.empty((int(iter), antennas, frequencies, polarizations), dtype=int)

    for i in range(len(gamma_test)):
        _, _, _, outlier_counts = winsorizing_vectorizer(data, gamma_test[i], int(iter_threshold))
        combined_outlier_counts[i] = outlier_counts

    ### Final gamma selection
    combined_outlier_flat = combined_outlier_counts.reshape(int(iter), -1)
    n_series = combined_outlier_flat.shape[1]
    final_gamma = np.empty(n_series)

    for s in range(n_series):
        counts = combined_outlier_flat[:, s] 
        
        # Find the most frequently occurring count value
        vals, freqs = np.unique(counts, return_counts=True)
        most_common_val = vals[np.argmax(freqs)]
        
        # Get indices where this value occurs
        match_indices = np.where(counts == most_common_val)[0]
        
        # Pick the first occurrence (you can change this logic to pick k-th if needed)
        final_gamma[s] = gamma_test[match_indices[-1]]

    final_gamma = final_gamma.reshape(antennas, frequencies, polarizations)

    ## Generate index for saving the data
    index_time_blocks = np.tile(np.arange(time/len(obs_list)), len(obs_list))
    index_obs_id = np.repeat(obs_list, time/len(obs_list))

    ## Winsorizing with final gamma value
    data_wins, win_z_scores, outliers_mask, outlier_counts = winsorizing_vectorizer(data, final_gamma, int(final_threshold))

    ## Generate some data results
    df_stats, df_outlier_counts, _ = export_data(obs_list, data, data_wins, outliers_mask, outlier_counts)


    # Save data
    ## Final gamma
    index = pd.MultiIndex.from_product([range(antennas), range(frequencies), range(polarizations)], names=["antenna", "frequency", "polarization"])
    df_final_gamma = pd.DataFrame({"final_gamma": final_gamma.flatten()}, index=index).reset_index()
    df_final_gamma.to_parquet(results_directory+"final_gamma_day_%s_grid_%s_integration_%s_%s.parquet" %(obs_day, grid, integration_time, data_type), engine="pyarrow", compression="snappy")

    ## Outliers statistics
    print("Saving the outlier statistics data ...")
    df_stats.to_parquet(results_directory+"outlier_statistics_day_%s_grid_%s_integration_%s_%s.parquet" %(obs_day, grid, integration_time, data_type), engine="pyarrow", compression="snappy")
    
    ## Outliers count
    print("Saving the outliers counts data ...")
    df_outlier_counts.to_parquet(results_directory+"outlier_counts_day_%s_grid_%s_integration_%s_%s.parquet" %(obs_day, grid, integration_time, data_type), engine="pyarrow", compression="snappy")

    ## Outliers location
    print("Saving the outliers location data ...")
    with h5py.File(results_directory+"outliers_location_day_%s_grid_%s_integration_%s_%s.h5" %(obs_day, grid, integration_time, data_type), "w") as f:
        f.create_dataset("outliers_mask", data=outliers_mask)
        f.create_dataset("obs_id", data=index_obs_id)
        f.create_dataset("time_blocks", data=index_time_blocks)

    ## Winsorize z score
    print("Saving the winsorize z score data ...")
    with h5py.File(results_directory+"win_z_scores_data_day_%s_grid_%s_integration_%s_%s.h5" %(obs_day, grid, integration_time, data_type), "w") as f:
        f.create_dataset("wins_z_score", data=win_z_scores)
        f.create_dataset("obs_id", data=index_obs_id)
        f.create_dataset("time_blocks", data=index_time_blocks)
    
    return print("All results files has been generated!")


def running(data_directory, results_directory, observation_file, integration_time, data_type, iter, iter_threshold, final_threshold):
    task_list = observation_id_preparation(observation_file)

    # Run multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(winsorizing_outlier_detection_3d, [(obs_days, grid, obs_list, data_directory, results_directory, integration_time, data_type, iter, iter_threshold, final_threshold) for obs_days, grid, obs_list in task_list])
    
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
    parser.add_argument("-it", "--integration_time", type=int, required=True, help="Integration time per each timeblock second ex. 2, 8")
    parser.add_argument("-t", "--type", required=True, help="The data type that consist of amplitude, phase, real and imaginary part")
    parser.add_argument("-ir", "--iter", required=True, help="Number of iteration to find final gamma value ex. 10, 100")
    parser.add_argument("-ito", "--iterthresh", required=True, help="Iteration threshold", default=3)
    parser.add_argument("-fto", "--finalthresh", required=True, help="Final threshold used for outliers detection", default=5)

    args = parser.parse_args()

    data_directory = args.data_directory
    results_directory = args.results_directory
    observation_file = args.obs_file

    integration_time = args.integration_time
    data_type = args.type
    iter = args.iter
    iter_threshold = args.iterthresh 
    final_threshold = args.finalthresh

    running(data_directory, results_directory, observation_file, integration_time, data_type, iter, iter_threshold, final_threshold)


if __name__ == "__main__":
    main()