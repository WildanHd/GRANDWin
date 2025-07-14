import numpy as np
import pandas as pd
import random
import argparse
import sys
import os
from scipy.constants import k
from astropy.io import fits
from astropy.time import Time
from astropy import constants

class RFIAdder:
    def __init__(self, rfi_intensity, max_number_rfi, polarization, uvfits_file, finished_file, rfi_output):
        self.rfi_intensity = rfi_intensity
        self.max_number_rfi = max_number_rfi
        self.polarization = polarization
        self.uvfits_file = uvfits_file
        self.finished_file = finished_file
        self.rfi_output = rfi_output

    def radectolmn(ra, ra0, dec, dec0):
        l = np.cos(dec) * np.sin(ra - ra0)
        m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
        n = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
        status = np.where(n<0,0,1)
        return status, l, m, n
    
    def adding_rfi(self):
        uf = fits.open(self.uvfits_file)
	
        # Point sources-like RFI

        print("Add random RFI-like noise!")
        print("... Generate necessary parameters")

        # Necessary parameters for this observation

        print("...... Observation information details!")

        freq = uf[1].header['FREQ']-(uf[0].header['CDELT4']*(uf[0].header['NAXIS4']/2)) # fiducial frequency
        dfreq = uf[0].header['CDELT4']
        uu = uf[0].data['UU'] # baseline u in units of s
        vv = uf[0].data['VV'] # baseline v in units of s
        ww = uf[0].data['WW'] # baseline w in units of s
        ra_c = uf[0].header['CRVAL5'] #phase centre RA in deg
        dec_c = uf[0].header['CRVAL6'] #phase centre Dec in deg

        _, unique_id = np.unique(uf[0].data['DATE'], return_inverse=True)
        
        print(self.max_number_rfi, self.rfi_intensity, self.polarization)

        tb_list_all = list(np.unique(unique_id))
        print(len(tb_list_all))
        number_tb = int(self.max_number_rfi * int(len(tb_list_all)))
        print(number_tb)
        tb_list = random.sample(tb_list_all, number_tb)

        isrc_rfi = self.rfi_intensity
        pol_rfi = [int(digit) for digit in str(self.polarization)]

        for j in range(number_tb):
            
            ra_rfi = np.random.randint(35, 65) 
            dec_rfi = np.random.randint(-30, 30) 

            ch_rfi_full = list(np.arange(0, uf[0].header['NAXIS4'], 1))
            print(ch_rfi_full)
            number_freq = int(self.max_number_rfi * uf[0].header['NAXIS4'])
            print(number_freq)
            ch_rfi_all = random.sample(ch_rfi_full, number_freq)
            print(ch_rfi_all)

            print("number_rfi: ", number_freq)
            print("channel_rfi: ", ch_rfi_all)

            print("...... Number of affected frequency channels: ", number_freq)

            for k in range(len(ch_rfi_all)):

                # Generate RFI details (flux intensity, direction and locations, time, number of polarization)

                print("...... RFI details!")
    
                ch_rfi = ch_rfi_all[k]
                time_rfi = tb_list[j]

                print("......... We have the RFI with strength: %s, channel: %s, time: %s, polarization: %s" %(isrc_rfi, ch_rfi, time_rfi, pol_rfi))

                # Making an array of RFI parameters

                deg2rad = np.pi/180.

                isrc = np.asarray([isrc_rfi])
                c_ra = np.asarray([ra_rfi * deg2rad])
                c_dec = np.asarray([dec_rfi * deg2rad])
                freq_rfi = freq + ch_rfi * dfreq

                # Transform RFI locations to telescope coordinate systems

                print(c_ra, c_dec, ra_c * deg2rad, dec_c * deg2rad)
                status,l,m,n = self.radectolmn(c_ra, c_dec, ra_c * deg2rad, dec_c * deg2rad)

                # Select the u,v,w where the RFI exist
                # This u,v,w taken from the visibility location in u,v,w coordinate systems

                id_uv = np.where(unique_id==time_rfi)[0]
                uu_rfi = uu[id_uv] * freq_rfi
                vv_rfi = vv[id_uv] * freq_rfi
                ww_rfi = ww[id_uv] * freq_rfi

                # Here we calculate the RFI infulence on the visibility

                print("... RFI influence on the visibility")

                v_cpu_real_array = np.zeros(len(id_uv))
                v_cpu_imag_array = np.zeros(len(id_uv))

                n_in = np.sqrt(1 - l*l - m*m)
                polar = 2 * np.pi * (np.outer(l, uu_rfi) + np.outer(m, vv_rfi) +  np.outer((n_in-1), ww_rfi))

                input_intensity = ((isrc + 0*1j)).T

                vis = np.sum((input_intensity *  (np.cos(polar) + np.sin(polar)*1j)),axis=0)

                # Addition to the raw uvfits

                print("...... Real")

                v_cpu_real_array = vis.real
                v_cpu_real_array = np.tile(v_cpu_real_array.reshape(int(len(unique_id)/len(np.unique(unique_id))), 1, 1, 1, 1, 1), uf[0].data['DATA'][np.ix_([0],[0],[0],[ch_rfi],pol_rfi,[0])].shape)

                print("...... Add real part")

                uf[0].data['DATA'][np.ix_(id_uv,[0],[0],[ch_rfi],pol_rfi,[0])] = uf[0].data['DATA'][np.ix_(id_uv,[0],[0],[ch_rfi],pol_rfi,[0])] + v_cpu_real_array

                del v_cpu_real_array

                print("...... Imaginary")

                v_cpu_imag_array = vis.imag
                v_cpu_imag_array = np.tile(v_cpu_imag_array.reshape(int(len(unique_id)/len(np.unique(unique_id))), 1, 1, 1, 1, 1), uf[0].data['DATA'][np.ix_([0],[0],[0],[ch_rfi],pol_rfi,[0])].shape)

                print("...... Add imaginary part")

                uf[0].data['DATA'][np.ix_(id_uv,[0],[0],[ch_rfi],pol_rfi,[1])] = uf[0].data['DATA'][np.ix_(id_uv,[0],[0],[ch_rfi],pol_rfi,[1])] + v_cpu_imag_array

                del v_cpu_imag_array, vis

                file_r = open(self.rfi_output, mode = "a", newline = "")
                file_r.write("%s, %s, %s, %s, %s, %s, %s, %s \n" %(isrc_rfi, ra_rfi, dec_rfi, ch_rfi, time_rfi, number_tb, number_freq, pol_rfi))
                file_r.close()

        print("... Write the updated data to the file")

        # Save the uvfits with additional thermal noise and RFI

        uf.writeto(self.finished_file, overwrite=True)
        uf.close()

        del uf
    

    

