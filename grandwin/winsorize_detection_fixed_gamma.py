import numpy as np
import pandas as pd
import h5py
from scipy import stats
from grandwin.io.fits_reader import import_data
from grandwin.io.detected_outliers_file_output import export_data


def winsorizing_vectorizer(data, gamma, threshold, state):
    """
    Winsorizes data along the time axis based on either a scalar or per-feature gamma.
    gamma represents the proportion of data to clip on EACH tail.
    Applies the rigorous statistical consistency correction factor for the Winsorized variance.
    """
    run_state = str(state).strip().lower()
    time, antennas, frequencies, polarizations = data.shape
    T = time
    N = antennas * frequencies * polarizations

    data_reshape = data.reshape(T, -1)
    nan_mask_flat = np.isnan(data_reshape)

    data_wins = np.full_like(data_reshape, np.nan)
    win_z_scores = np.full_like(data_reshape, np.nan)
    outlier_masks = np.zeros_like(data_reshape, dtype=bool)

    # --- Calculate the statistical correction factor ---
    if np.isscalar(gamma):
        gamma_flat = np.full((N,), gamma)
       
        if gamma > 0:
            c = stats.norm.ppf(1 - gamma)
            phi_c = stats.norm.pdf(c)
            var_ratio = (1 - 2 * gamma) - 2 * c * phi_c + 2 * (c**2) * gamma
            corr_factor = 1.0 / np.sqrt(var_ratio)
        else:
            corr_factor = 1.0
           
        correction_flat = np.full((N,), corr_factor)
    else:
        if gamma.shape != (antennas, frequencies, polarizations):
            raise ValueError("gamma must be scalar or have shape (antennas, frequencies, polarizations)")
        gamma_flat = gamma.reshape(-1)
        correction_flat = np.ones_like(gamma_flat, dtype=float)
       
        # Mask to avoid math errors if any gamma is 0
        valid_mask = gamma_flat > 0
        if np.any(valid_mask):
            valid_g = gamma_flat[valid_mask]
            c = stats.norm.ppf(1 - valid_g)
            phi_c = stats.norm.pdf(c)
            var_ratio = (1 - 2 * valid_g) - 2 * c * phi_c + 2 * (c**2) * valid_g
            correction_flat[valid_mask] = 1.0 / np.sqrt(var_ratio)

    print("Correction factor: ", correction_flat)

    # Main loop for winsorizing
    for i in range(N):
        col = data_reshape[:, i]
        mask = nan_mask_flat[:, i]
        valid_data = col[~mask]

        if valid_data.size == 0:
            continue

        sorted_col = np.sort(valid_data)
       
        # Calculate k using gamma directly (no dividing by 2!)
        k = int(np.floor(gamma_flat[i] * len(valid_data)))

        low = sorted_col[k] if k < len(valid_data) else sorted_col[0]
        high = sorted_col[-k - 1] if k < len(valid_data) else sorted_col[-1]

        # Winsorize valid data
        wins_data = np.clip(valid_data, low, high)
        mu = np.mean(wins_data)
       
        # Apply the mathematical correction to scale the standard deviation
        sigma = np.std(wins_data) * correction_flat[i]
       
        if sigma == 0:
            sigma = 1e-6

        full_col = col.copy()
        full_col[~mask] = wins_data
        data_wins[:, i] = full_col

        z_col = (valid_data - mu) / sigma
        win_z_scores[~mask, i] = z_col

        if run_state == 'both':
            outlier_masks[~mask, i] = (z_col > threshold) | (z_col < -threshold)
        if run_state == 'positive':
            outlier_masks[~mask, i] = (z_col > threshold)
        if run_state == 'negative':
            outlier_masks[~mask, i] = (z_col < -threshold)

    data_wins = data_wins.reshape(time, antennas, frequencies, polarizations)
    win_z_scores = win_z_scores.reshape(time, antennas, frequencies, polarizations)
    outlier_masks = outlier_masks.reshape(time, antennas, frequencies, polarizations)
    outlier_counts = np.sum(outlier_masks, axis=0)

    return data_wins, win_z_scores, outlier_masks, outlier_counts

def winsorizing_outlier_detection_3d(obs_day, grid, obs_list, data_directory, results_directory, integration_time, data_type, final_threshold, gamma, partition, grid_point, state):

    print("Check all parameters: ", obs_day, grid, obs_list, data_directory, results_directory, integration_time, data_type, final_threshold, gamma, flush=True)

    print("Import data ...", flush=True)
    data = import_data(obs_list, data_directory, data_type)
   
    time, antennas, frequencies, polarizations = data.shape

    ## Generate index for saving the data
    index_time_blocks = np.tile(np.arange(time/len(obs_list)), len(obs_list))
    index_obs_id = np.repeat(obs_list, time/len(obs_list))

    print(f"Winsorizing with fixed gamma ({gamma}) and corrected variance ...", flush=True)

    ## Winsorizing with fixed gamma value
    data_wins, win_z_scores, outliers_mask, outlier_counts = winsorizing_vectorizer(data, gamma, float(final_threshold), state)

    ## Generate some data results
    df_stats, df_outlier_counts, _ = export_data(obs_list, data, data_wins, outliers_mask, outlier_counts)

    print("Save output ...", flush=True)

    # Note: df_final_gamma saving block was deleted since gamma is now a single fixed constant

    ## Outliers statistics
    print("... Saving the outlier statistics data", flush=True)
    df_stats.to_parquet(results_directory+"outlier_statistics_day_%s_grid_%s_integration_%s_%s_part_%s_gp_%s.parquet" %(obs_day, grid, integration_time, data_type, partition, grid_point), engine="pyarrow", compression="snappy")
   
    ## Outliers count
    print("... Saving the outliers counts data", flush=True)
    df_outlier_counts.to_parquet(results_directory+"outlier_counts_day_%s_grid_%s_integration_%s_%s_part_%s_gp_%s.parquet" %(obs_day, grid, integration_time, data_type, partition, grid_point), engine="pyarrow", compression="snappy")

    ## Outliers location
    print("... Saving the outliers location data", flush=True)
    with h5py.File(results_directory+"outliers_location_day_%s_grid_%s_integration_%s_%s_part_%s_gp_%s.h5" %(obs_day, grid, integration_time, data_type, partition, grid_point), "w") as f:
        f.create_dataset("outliers_mask", data=outliers_mask)
        f.create_dataset("obs_id", data=index_obs_id)
        f.create_dataset("time_blocks", data=index_time_blocks)

    ## Winsorize z score
    print("... Saving the winsorize z score data", flush=True)
    with h5py.File(results_directory+"win_z_scores_data_day_%s_grid_%s_integration_%s_%s_part_%s_gp_%s.h5" %(obs_day, grid, integration_time, data_type, partition, grid_point), "w") as f:
        f.create_dataset("wins_z_score", data=win_z_scores)
        f.create_dataset("obs_id", data=index_obs_id)
        f.create_dataset("time_blocks", data=index_time_blocks)
   
    return print("All results files have been generated!", flush=True)