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


def outliers_selection(obs_list, data_directory, result_directory, polr, tiles, freqs, outlier_threshold, iteration_number, step, df, part):

	print("Processed observation id: ", obs_list)
	print("Data directory: ", data_directory)
	print("Result directory: ", result_directory)
	print("Polarization: ", polr)
	print("Number of tiles: ", tiles)
	print("Number of frequency: ", freqs)
	print("Outlier threshold: ", outlier_threshold)
	print("Number of iteration: ", iteration_number)
	print("Number of steps: ", step)

	gp_name = np.unique(df[df['obs_id'].isin(obs_list)]['gridpoint_number'])

	file_outliers = 'both_side_selected_outliers_%s_sigma_day_%s_grid_%s_iter_%s_real_part_8s.csv' %(outlier_threshold, part, gp_name, iteration_number)
	file_statistics = 'both_side_outliers_statistics_%s_sigma_day_%s_grid_%s_iter_%s_real_part_8s.csv' %(outlier_threshold, part, gp_name, iteration_number)

	print("File outliers: ", file_outliers)
	print("File statistics: ", file_statistics)

	for i in range(tiles):
		for j in range(freqs):

			print('Tile: ', i, ' and Freqs: ', j)

			df_data = pd.DataFrame()
			for k in range(len(obs_list)):
				f = fits.open(data_directory + "hyperdrive_solutions_%s_noise.fits" %(obs_list[k]))

				data = f['SOLUTIONS'].data[:, i, j, ::2]

				a = pd.DataFrame(data[:, [0,3]], columns=polr)
				a['obs_id'] = obs_list[k]
				a['timeblocks'] = range(0, len(a))
				a['tile'] = i
				a['freq_channels'] = j
				df_data = pd.concat([df_data, a]).reset_index(drop = True)

			print("All_data: ", df_data)
			print("Minimum timeblocks: ", np.min(df_data['timeblocks']), "maximum timeblocks: ", np.max(df_data['timeblocks']))

			for l in range(len(polr)):
				print("Winsorizing polarization: %s" %(polr[l]))

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

					#print("Gamma dataframe: ", dfg)

					if len(dfg) != 0:
						change_indices = [n for n in range(1, len(dfg['outliers_number'])) if dfg['outliers_number'][n] != dfg['outliers_number'][n-1]]

						if len(change_indices) != 0:
							gamma_final = dfg['gamma'][change_indices[0]-1]
						else:
							gamma_final = dfg['gamma'].iloc[-1]
					else:
						gamma_final = 0.

					#print(gamma_final)

					# Recalculate the number of outliers using final gamma and store the outliers data

					b = df_data[~df_data['%s' %(polr[l])].isna()].sort_values(by='%s' %(polr[l]), ascending=True).reset_index(drop=True)
					b = stats.mstats.winsorize(b['%s' %(polr[l])], limits=[gamma_final/2, gamma_final/2])
					b = pd.DataFrame(data=b.data, columns=['amplitude'])

					# Calculate winsorized statistics, z score of the data, and select the outliers from the data outside the outlier threshold

					win_mean = np.mean(b['amplitude'])
					win_std = np.std(b['amplitude'])
					df_data['z_score_mean'] = (df_data['%s' %(polr[l])] - win_mean) / win_std
					c = df_data #[(df_data['z_score_mean'] <= -outlier_threshold) | (df_data['z_score_mean'] >= outlier_threshold)].reset_index(drop = True)
					c['polr'] = polr[l]

					#print("Outliers: ", c)

					c.to_csv(result_directory+file_outliers, mode='a', index=False, header=False)

					e = str(i) + ', ' + str(j) + ', ' + str(polr[l]) + ', ' + str(stats.skew(df_data[~df_data['%s' %(polr[l])].isna()]['%s' %(polr[l])])) + ', ' + str(np.nanmax(df_data['%s' %(polr[l])])) + ', ' + str(np.nanmin(df_data['%s' %(polr[l])])) + ', ' + str(np.nanmean(df_data['%s' %(polr[l])])) + ', ' + str(win_mean) + ', ' + str(np.nanstd(df_data['%s' %(polr[l])]))  + ', ' + str(win_std) + ', ' + str(gamma_final) + ', ' + str(len(c)) + '\n'

					print(e)

					file_stats = open(result_directory+file_statistics, mode='a', newline='\n')
					file_stats.write(e)
					file_stats.close()
				else:
					continue

if __name__ == "__main__":

	# Define variables
	polr = ['XX', 'YY']
	tiles = 128
	freqs = 768
	part = 1

	outlier_threshold = 3
	iteration_number = 100
	step = 0.2 / iteration_number

	# Define data directory
	data_directory = "/Users/eormacstudio/Documents/GitHub/GRANDWin/data/raw/calibration_solutions/20240831_multiple_simulation_higher_thermal_noise_sn3_8s/uvfits/calibrated_fits/"
	observation_file = "/Users/eormacstudio/Documents/GitHub/GRANDWin/data/raw/observation_id/observation_id.csv"
	result_directory = "/Users/eormacstudio/Documents/GitHub/GRANDWin/test/results/slower/"

	# Import observation id
	try:
		df = pd.read_csv(observation_file, header=0, engine='python')
		df['date'] = df.starttime_utc.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z").date())
		df['partition'] = pd.factorize(df['date'])[0] + 1
		df = df[df['partition'] == part].reset_index(drop=True)

	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)

	gp = np.unique(df['gridpoint_number'])

	run_list = []
	for i in range(len(gp)):
		run_list.append(df[df['gridpoint_number'] == gp[i]]['obs_id'].to_list())

	args = [(lst, data_directory, result_directory, polr, tiles, freqs, outlier_threshold, iteration_number, step, df, part) for lst in run_list]
	
	pool = multiprocessing.Pool(processes=len(gp))

	results = pool.starmap(outliers_selection, args)

	pool.close()
	pool.join()