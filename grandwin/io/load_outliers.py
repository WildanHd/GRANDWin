# grandwin/flagging/load_outliers.py

import h5py
import numpy as np
import pandas as pd

def load_outliers_from_h5(file_path, polarizations):
    with h5py.File(file_path, "r") as f:
        outliers_mask = f["outliers_mask"][:]
        obs_id = f["obs_id"][:].astype(str)
        time_blocks = f["time_blocks"][:]

    print("Outliers: ", outliers_mask.shape)

    flat_data = outliers_mask.reshape(-1, outliers_mask.shape[-1])
    
    # Create DataFrame
    df_outliers = pd.DataFrame(flat_data, columns=["XX", "XY", "YX", "YY"])

    df_outliers["obs_id"] = np.tile(obs_id, outliers_mask.shape[1]*outliers_mask.shape[2])
    df_outliers["time_index"] = np.repeat(np.arange(outliers_mask.shape[0]), outliers_mask.shape[1] * outliers_mask.shape[2])
    df_outliers["frequency"] = np.tile(np.arange(outliers_mask.shape[2]), outliers_mask.shape[0] * outliers_mask.shape[1])
    df_outliers["antenna"] = np.tile(np.repeat(np.arange(outliers_mask.shape[1]), outliers_mask.shape[2]), outliers_mask.shape[0])
    df_outliers["obs_id"] = df_outliers['time_index'].map(lambda t: int(obs_id[t]))
    df_outliers["timeblock"] = df_outliers['time_index'].map(lambda t: int(time_blocks[t]))

    selected_polarizations = polarizations  

    # Filter rows where any of the selected columns is True
    df_outliers = df_outliers[df_outliers[selected_polarizations].any(axis=1)].reset_index(drop=True)

    return df_outliers