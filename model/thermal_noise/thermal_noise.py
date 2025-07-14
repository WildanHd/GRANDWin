import numpy as np
import pandas as pd
import argparse
import sys
import os
from scipy.constants import k
from astropy.io import fits
from astropy.time import Time
from astropy import constants

class ThermalNoiseAdder:
    def __init__(self, sky_temperature, telescope_temperature, frequency_0, power_law, beam_response_file, uvfits_file, finished_file):
        self.sky_temperature = sky_temperature
        self.telescope_temperature = telescope_temperature
        self.frequency_0 = frequency_0
        self.power_law = power_law
        self.beam_response_file = beam_response_file
        self.uvfits_file = uvfits_file
        self.finished_file = finished_file

    def see_inside(self):
        return print(self.sky_temperature, self.telescope_temperature, self.frequency_0, self.power_law, self.beam_response_file, self.uvfits_file)
    
    def import_beam_response(self):
        br = pd.read_csv(self.beam_response_file)
        br = br[['frequency', 'solid_angle']]
        br = br.sort_values(by='frequency')

        beam_response = np.round(br['solid_angle'].to_list(), 3)

        return beam_response

    def generate_noise_shape_frequency_and_time(self):
        uf = fits.open(self.uvfits_file)

        # Calculate noise size

        noise_shape = (uf[0].data.data.shape[0], uf[0].data.data.shape[1], uf[0].data.data.shape[2], uf[0].data.data.shape[3], uf[0].data.data.shape[4])
        
        # Calculate the frequency channel used in the data in (MHz) and delta frequency (Hz)

        first_freq = uf[1].header['FREQ']-(uf[0].header['CDELT4']*(uf[0].header['NAXIS4']/2))
        last_freq = uf[1].header['FREQ']+(uf[0].header['CDELT4']*(uf[0].header['NAXIS4']/2))
        num_freq = uf[0].header['NAXIS4']
        freqs = np.round(np.linspace(first_freq, last_freq, num_freq), 3)/10**6

        dfreqs = uf[0].header['CDELT4']

        # Calculate the time resolutions used in the data in s

        time1 = Time(np.unique(uf[0].data['DATE'], return_inverse=True)[0][0], format='jd')
        datetime_obj_1 = time1.to_datetime()

        time2 = Time(np.unique(uf[0].data['DATE'], return_inverse=True)[0][1], format='jd')
        datetime_obj_2 = time2.to_datetime()

        time_resolutions = (datetime_obj_2 - datetime_obj_1).seconds

        return noise_shape, freqs, dfreqs, time_resolutions
    
    def generate_gaussian_noise(self):
        noise_shape, _, _, _ = self.generate_noise_shape_frequency_and_time()

        gauss_std = 1 / np.sqrt(2)
        gaussian_noise = np.random.normal(0, gauss_std, size = noise_shape) + 1j * np.random.normal(0, gauss_std, size = noise_shape)

        print("...... Noise: ", gaussian_noise[1000,:,:,5,:])

        return gaussian_noise

    def temperature_model(self):
        _, freqs, _, _ = self.generate_noise_shape_frequency_and_time()

        T_all = [] # For the sky model temperature and receiver temperature
        T_sys = [] # For the receiver temperature only

        for i in range(len(freqs)):
            tempr = self.sky_temperature * (freqs[i] / self.frequency_0)**self.power_law # This is the power law function
            T_all.append(tempr + 200) # Assume MWA receiver temperature 200K

        for i in range(len(freqs)):
            T_sys.append(self.telescope_temperature)

        print("T_all: ", T_all)
        print("T_sys: ", T_sys)

        return T_all, T_sys

    def thermal_noise_conversion(self):
        T_all, T_sys = self.temperature_model()
        _, freqs, dfreqs, time_resolutions = self.generate_noise_shape_frequency_and_time()
        beam_response = self.import_beam_response()

        V_thermal_all = T_all / np.sqrt(time_resolutions * dfreqs)
        V_thermal_sys = T_sys / np.sqrt(time_resolutions * dfreqs)

        wavelengths = constants.c.value / (freqs * 1e6)

        V_thermal_all /= (1e-26 * wavelengths**2) / (2 * constants.k_B.value * beam_response) 
        V_thermal_sys /= (1e-26 * wavelengths**2) / (2 * constants.k_B.value * beam_response)

        V_thermal_xx_yy = np.stack((V_thermal_all, V_thermal_all), axis=-1).reshape(1,1,1,len(freqs),2)
        print("...... V_thermal_xx_yy: ", V_thermal_xx_yy[:,:,:,5,:])

        V_thermal_xy_yx = np.stack((V_thermal_sys, V_thermal_sys), axis=-1).reshape(1,1,1,len(freqs),2)
        print("...... V_thermal_xy_yx: ", V_thermal_xy_yx[:,:,:,5,:])

        return V_thermal_xx_yy, V_thermal_xy_yx

    def adding_thermal_noise(self):
        V_thermal_xx_yy, V_thermal_xy_yx = self.thermal_noise_conversion()
        gaussian_noise = self.generate_gaussian_noise()

        uf = fits.open(self.uvfits_file)

        print("... Real")

        V_thermal_xx_yy_real = (gaussian_noise.real[:,:,:,:,[0,1]] * V_thermal_xx_yy)
        V_thermal_xy_yx_real = (gaussian_noise.real[:,:,:,:,[2,3]] * V_thermal_xy_yx)

        print("...... V_thermal_xx_yy sample real: ", (V_thermal_xx_yy_real[1000,:,:,5,:]))
        print("...... V_thermal_xy_yx sample real: ", (V_thermal_xy_yx_real[1000,:,:,5,:]))

        V_thermal_real = np.concatenate((V_thermal_xx_yy_real, V_thermal_xy_yx_real), axis=-1)

        print("...... V_thermal_real: ", V_thermal_real[1000,:,:,5,:])

        print("... Add real part")

        uf[0].data["DATA"][:,:,:,:,:,0] = uf[0].data["DATA"][:,:,:,:,:,0] + V_thermal_real

        del V_thermal_xx_yy_real, V_thermal_xy_yx_real, V_thermal_real

        print("... Imaginary")

        V_thermal_xx_yy_imag = (gaussian_noise.imag[:,:,:,:,[0,1]] * V_thermal_xx_yy)
        V_thermal_xy_yx_imag = (gaussian_noise.imag[:,:,:,:,[2,3]] * V_thermal_xy_yx)

        print("...... V_thermal_xx_yy sample imag: ", (V_thermal_xx_yy_imag[1000,:,:,5,:]))
        print("...... V_thermal_xy_yx sample imag: ", (V_thermal_xy_yx_imag[1000,:,:,5,:]))

        V_thermal_imag = np.concatenate((V_thermal_xx_yy_imag, V_thermal_xy_yx_imag), axis=-1)

        print("...... V_thermal_imag: ", V_thermal_imag[1000,:,:,5,:])

        print("... Add imaginary part")

        uf[0].data["DATA"][:,:,:,:,:,1] = uf[0].data["DATA"][:,:,:,:,:,1] + V_thermal_imag

        del V_thermal_xx_yy_imag, V_thermal_xy_yx_imag, V_thermal_imag

        print("... Write the updated data to the file")

        # Save the uvfits with additional thermal noise and RFI

        uf.writeto(self.finished_file, overwrite=True)
        uf.close()

        del uf
        return "Finished adding thermal noise for %s!" %(self.finished_file)