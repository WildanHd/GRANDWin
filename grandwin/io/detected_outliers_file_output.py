# grandwin/io/detected_outliers_file_output.py
# This script use to export the detected outliers location and statistical properties

import numpy as np
import pandas as pd
from scipy import stats

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
    ori_mean = np.nanmean(data, axis=0)
    ori_std = np.nanstd(data, axis=0)
    ori_max = np.nanmax(data, axis=0)
    ori_min = np.nanmin(data, axis=0)
    ori_skew = stats.skew(data, axis=0, nan_policy="omit")

    ## Calculate winsorized mean and std
    data_wins = data_wins.reshape(time, antennas, frequencies, polarizations)
    win_mean = np.nanmean(data_wins, axis=0)
    win_std = np.nanstd(data_wins, axis=0)
    win_skew = stats.skew(data_wins, axis=0, nan_policy="omit")

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