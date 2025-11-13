# grandwin/flagging/apply_flags.py

import numpy as np
from astropy.io import fits

def expand_timeblocks(timeblocks, win_step, uv_step):
    factor = int(win_step / uv_step)
    return np.array([tb * factor + i for tb in timeblocks for i in range(factor)])

def flag_uvfits_data(obs_id, uvfits_path, df_outliers, output_path, win_step, uv_step, results_dir):
    total_flags = []
    print("... Open the uvfits file", flush=True)

    uv = fits.open(uvfits_path)

    flagsb = np.count_nonzero(uv[0].data.data[:, :, :, :, :, 2] < 0)
    datatot = np.count_nonzero(uv[0].data.data[:, :, :, :, :, 2] < 0) + np.count_nonzero(uv[0].data.data[:, :, :, :, :, 2] >= 0)
    print("... Data shape before: ", uv[0].data.data.shape, flush=True)
    print("... Flags before: ", flagsb, flush=True)
    print("... Total data before: ", datatot)
    print("df_outliers: ", df_outliers)

    _, unique_id = np.unique(uv[0].data['DATE'], return_inverse=True)

    dtimeblocks = np.unique(df_outliers['timeblock'])

    for j in range(len(dtimeblocks)):
        dfreqs = np.unique(df_outliers[df_outliers['timeblock'] == dtimeblocks[j]]['frequency'])
        print("dfreqs: ", len(dfreqs), "\n", dfreqs)

        extimeblocks = expand_timeblocks(df_outliers[df_outliers['timeblock'] == dtimeblocks[j]]['timeblock'].unique(), win_step, uv_step)
        print("timeblocks: ", extimeblocks)

        blindices = np.where(np.isin(unique_id, extimeblocks))[0]
        print("baselines: ", len(blindices), "\n", blindices)

        uv[0].data.data[np.ix_(blindices, [0], [0], dfreqs, [0,1,2,3], [2])] = np.abs(uv[0].data.data[np.ix_(blindices, [0], [0], dfreqs, [0,1,2,3], [2])]) * -1

        print("... The data that being flagged ", len(dfreqs)*len(blindices)*4, flush=True)
        total_flags.append(len(dfreqs)*len(blindices)*4)

    flagsa = np.count_nonzero(uv[0].data.data[:, :, :, :, :, 2] < 0)
    
    print("... Data shape after: ", uv[0].data.data.shape, flush=True)
    print("... Flags after: ", flagsa, flush=True)
    print("... Total flags: ", np.sum(total_flags), flush=True)

    uv.writeto(output_path, overwrite=True)
    uv.close()

    g = str(obs_id) + ', ' + str(flagsb) + ', ' + str(flagsa) + ', ' + str(np.sum(total_flags)) + ', ' + str(datatot) + '\n'
    fflagged = open(results_dir+f"{obs_id}.log", mode='a', newline='\n')
    fflagged.write(g)
    fflagged.close()
    print("... Flagging finished!", flush=True)