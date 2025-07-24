import sys
import resource
import datetime
import numpy as np
import pandas as pd
import astropy as ap
import scipy.stats as stats
from astropy.io import fits
from pyuvdata import UVData
import math as ma
import matplotlib.pyplot as plt
import pyuvdata
import multiprocessing


def flagging_raw_fits(obs_list, raw_directory, finished_file, result_directory, polr, file_flagged, file_outliers):

	def create_2s_tb(val):
		start = val * 4
		return list(range(start, start + 4))

	# Flags the data for the specific observation id, tile, timeblocks, and frequency channels
	# Reduced flagged data

	print("Beginning data: ", obs_list)
	print("Raw data directory: ", raw_directory)
	print("Finished obs_id file: ", finished_file)
	print("Result directory: ", result_directory)
	print("Polarization: ", polr)

	df_finished = pd.read_csv(finished_file, sep='\t', header=None, engine='python')
	df_finished.columns = ['obs_id']
	finished_list = df_finished['obs_id'].to_list()

	print("Finished_list: ", finished_list)

	if len(finished_list) != 0:
		obs_data = np.array(obs_list)[~np.isin(obs_list, finished_list)]
	else:
		obs_data = obs_list

	print("Use obs data: ", len(obs_data))

	# Import outliers data
	df_outliers = pd.read_csv(result_directory+file_outliers, sep=',', header=None, engine='python')
	df_outliers.columns = ['XX', 'XY', 'YX', 'YY', 'obs_id', 'timeblocks', 'tile', 'freq_channels', 'z_score', 'polr']
	#df_outliers.columns = ['XX', 'obs_id', 'timeblocks', 'tile', 'freq_channels', 'z_score', 'polr']

	df_outliers['floor_z_score'] = df_outliers['z_score'].apply(lambda x: ma.floor(x))
	df_outliers = df_outliers[df_outliers['floor_z_score'] < -5].reset_index(drop=True)
	df_outliers = df_outliers[df_outliers['obs_id'].isin(obs_data)].reset_index(drop=True)
	df_outliers = df_outliers[df_outliers['polr'].isin(polr)].reset_index(drop=True)

	df_outliers['2s_timeblocks'] = df_outliers['timeblocks'].apply(create_2s_tb)
	df_outliers = df_outliers.explode('2s_timeblocks').reset_index(drop=True)

	print("Outliers data: ", df_outliers)

	for i in range(len(obs_data)):
		total_flags = []

		# Open the raw data

		uv = fits.open(raw_directory + '%s/raw_%s_w_flag059-078.uvfits' %(obs_data[i], obs_data[i]))
		flagsb = np.count_nonzero(uv[0].data.data[:, :, :, :, :, 2] < 0)
		datatot = np.count_nonzero(uv[0].data.data[:, :, :, :, :, 2] < 0) + np.count_nonzero(uv[0].data.data[:, :, :, :, :, 2] >= 0)

		date_data = uv[0].data['DATE'] #observation time of the data
		date_unique, unique_id = np.unique(date_data, return_inverse=True)

		print("Data shape before: ", uv[0].data.data.shape)
		print("Flags before: ", flagsb)
		print("Total data: ", datatot)

		a = df_outliers[df_outliers['obs_id'] == obs_data[i]].sort_values(by = ['tile', '2s_timeblocks', 'freq_channels']).reset_index(drop=True)
		dtimeblocks = np.unique(a['2s_timeblocks'])

		print("Dataframe specific obs id for flag: ", a)

		for j in range(len(dtimeblocks)):

			print("Outlier timeblocks: ", dtimeblocks[j])

			dfreqs = np.unique(a[(a['2s_timeblocks'] == dtimeblocks[j])]['freq_channels'])

			print("For ", dtimeblocks[j], ", we have freqs: ", len(dfreqs), ", which are: ", dfreqs)

			dindices = np.where(unique_id == dtimeblocks[j])[0]

			print("Indices that we will flag: ", len(dindices), ", which are: ", dindices)

			uv[0].data.data[np.ix_(dindices, [0], [0], dfreqs, [0,1,2,3], [2])] = np.abs(uv[0].data.data[np.ix_(dindices, [0], [0], dfreqs, [0,1,2,3], [2])]) * -1

			print("The data that being flagged ", len(dfreqs)*len(dindices)*4)
			total_flags.append(len(dfreqs)*len(dindices)*4)

		flagsa = np.count_nonzero(uv[0].data.data[:, :, :, :, :, 2] < 0)
		print("Data shape after: ", uv[0].data.data.shape)
		print("Flags after: ", flagsa)
		print(np.sum(total_flags))

		g = str(obs_data[i]) + ', ' + str(flagsb) + ', ' + str(flagsa) + ', ' + str(np.sum(total_flags)) + ', ' + str(datatot) + '\n'
		fflagged = open(result_directory+file_flagged, mode='a', newline='\n')
		fflagged.write(g)
		fflagged.close()

		uv.writeto(raw_directory + "%s/raw_%s_w_flag059-078_updated.uvfits" %(obs_data[i], obs_data[i]), overwrite=True)
		uv.close()

		print("Add observationid %s for flagged observation!" %(obs_data[i]))
		file = open(finished_file, mode='a', newline='')
		file.write("%s \n" %(obs_data[i]))
		file.close()
		print(datetime.datetime.now())