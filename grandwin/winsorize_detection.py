import numpy as np
import pandas as pd
import h5py
from astropy import constants
from scipy import stats
from astropy.io import fits
from grandwin.io.fits_reader import import_data
from grandwin.io.detected_outliers_file_output import export_data

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
    #data_wins = np.clip(data_reshape, low_vals, high_vals)
    data_wins = data_reshape.copy()
    mask_low = data_reshape < low_vals
    mask_high = data_reshape > high_vals

    # Reshape low/high_vals for broadcasting
    low_vals_reshaped = low_vals.reshape(1, -1)
    high_vals_reshaped = high_vals.reshape(1, -1)

    # Vectorized winsorization
    data_wins = data_reshape.copy()
    data_wins = np.where(data_reshape < low_vals_reshaped, low_vals_reshaped, data_wins)
    data_wins = np.where(data_reshape > high_vals_reshaped, high_vals_reshaped, data_wins)


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


def winsorizing_outlier_detection_3d(obs_day, grid, obs_list, data_directory, results_directory, integration_time, data_type, iter, iter_threshold, final_threshold):

    print("Check all parameters: ", obs_day, grid, obs_list, data_directory, results_directory, integration_time, data_type, iter, iter_threshold, final_threshold)

    # Running winsorizing
    ## Import raw data
    data = import_data(obs_list, data_directory, data_type)

    ## Select the best gamma value
    ### Generate outlier counts based on set of gamma values
    gamma_test = np.linspace(0.0, 0.202, int(iter))

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
        gamma_list = gamma_test

        # Step 1: Build gamma ↔ outlier count DataFrame
        df = pd.DataFrame({"gamma": gamma_list, "outlier_count": counts})

        # Step 2: Find the most common outlier count
        count_freq = df["outlier_count"].value_counts()
        most_common_count = count_freq.index[0]

        # Step 3: Filter all gammas with that common count
        df_filtered = df[df["outlier_count"] == most_common_count].reset_index(drop=True)

        # Step 4: Find the first change point
        change_indices = [i for i in range(1, len(df_filtered)) if df_filtered["outlier_count"][i] != df_filtered["outlier_count"][i - 1]]

        # Step 5: Pick final gamma
        if change_indices:
            selected_gamma = df_filtered["gamma"].iloc[change_indices[0] - 1]
        else:
            selected_gamma = df_filtered["gamma"].iloc[-1]

        # Step 6: Optional correction if gamma > 0.2 and plateau is wide
        if selected_gamma > 0.2:
            trace_back = df_filtered[df_filtered["gamma"] <= 0.2]
            if not trace_back.empty:
                selected_gamma = trace_back["gamma"].iloc[-1]
            else:
                selected_gamma = 0.1  # safe fallback

        # Store result
        final_gamma[s] = selected_gamma


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