import numpy as np
import argparse
import sys
from scipy.constants import k
from astropy.io import fits
from common.parameters import sky_temperature, frequency_0, telescope_temperature, power_law
from thermal_noise.thermal_noise import ThermalNoiseAdder
from radio_frequency_interference.rfi import RFIAdder

def main():
    parser = argparse.ArgumentParser(description="Select process and provide related parameters ...", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="process", help="Choose process to execute")

    parser_process1 = subparsers.add_parser("thermal_noise", help="Generate thermal noise ...")

    parser_process1.add_argument("--beamfile", help="Beam response file ...", default="fitted_beam_response.csv")
    parser_process1.add_argument("--uvfitsfile", help="The name of uvfits file ...", default="visibility_model_1095449728.uvfits")
    parser_process1.add_argument("--finishedfile", help="The name of finished uvfits file ...", default="visibility_model_1095449728_noise.uvfits")

    parser_process2 = subparsers.add_parser("rfi", help="Generate point-sourcce-like radio frequency interference ...")
    
    parser_process2.add_argument("--rfiintensity", type=float, help="Define the maximum intensity of RFI. Ex. 0.5, means 50%% of the average absolute amplitude of thermal noise level.", default=0.05)
    parser_process2.add_argument("--maxrfi", type=float, help="Define the maximum number of RFI addded depend on the number of frequency channel and timeblocks. Ex. 0.5, means 50%% of frequency channel and timeblock is affected by RFI.", default=0.05)
    parser_process2.add_argument("--polarization", help="Define the polarization to add rfi. Ex. 01 if you want to add the RFI to XX and YY polarization. MWA data used 0 = XX, 1 = YY, 2 =XY, 3 = YX.", default="01")
    parser_process2.add_argument("--uvfitsfile", help="The name of uvfits file ...", default="visibility_model_1095449728.uvfits")
    parser_process2.add_argument("--finishedfile", help="The name of finished uvfits file ...", default="visibility_model_1095449728_noise.uvfits")
    parser_process2.add_argument("--rfioutput", help="File name to save the rfi added to the uvfits file ...", default="added_rfi.txt")

    args = parser.parse_args()

    if args.process == "thermal_noise":
        gaussian_thermal_noise = ThermalNoiseAdder(sky_temperature, telescope_temperature, frequency_0, power_law, beam_response_file=args.beamfile, uvfits_file=args.uvfitsfile, finished_file=args.finishedfile)
        gaussian_thermal_noise.see_inside()
        #print(gaussian_thermal_noise.import_beam_response())
        #print(gaussian_thermal_noise.generate_noise_shape_frequency_and_time())
        #print(gaussian_thermal_noise.generate_gaussian_noise())
        #print(gaussian_thermal_noise.temperature_model())
        #print(gaussian_thermal_noise.thermal_noise_conversion())
        print(gaussian_thermal_noise.adding_thermal_noise())
    
    elif args.process == "rfi":
        random_radio_frequency_interference = RFIAdder(rfi_intensity=args.rfiintensity, max_number_rfi=args.maxrfi, polarization=args.polarization, uvfits_file=args.uvfitsfile, finished_file=args.finishedfile, rfi_output=args.rfioutput)
        random_radio_frequency_interference.adding_rfi()
        print("rfi")
    
    elif args.process == "winsorizing":
        print("winsorizing")

    elif args.process == "flagging":
        print("flagging")
    
    else:
        parser.error("Unknown operation!")

if __name__ == "__main__":
    main()