# scripts/flag_data.py

import os
import argparse
from grandwin.flagger import flagging_raw_fits
from grandwin.metadata_parser import observation_id_preparation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flag UVFITS files based on winsorizing outliers.")
    parser.add_argument("-d", "--data_dir", required=True)
    parser.add_argument("-r", "--results_dir", required=True)
    parser.add_argument("-oi", "--obs_input", required=True, help="Either a single observation ID or a path to a CSV file containing observation IDs")
    parser.add_argument("-of", "--outlier_file", required=True)
    parser.add_argument("-f", "--finished_file", required=True)
    parser.add_argument("-pol", "--polarization", nargs="+", required=True)
    parser.add_argument("-wi", "--winsor_integration", type=int, required=True)
    parser.add_argument("-ui", "--uvfits_integration", type=int, required=True)

    args = parser.parse_args()

    def get_observation_list(obs_input):
        if os.path.isfile(obs_input):
            # Treat as CSV file
            task_list = observation_id_preparation(obs_input)
            obs_lists = [obs_list for _, _, obs_list in task_list]
            return [item for sublist in obs_lists for item in sublist]  # flatten
        else:
            # Treat as a single observation ID
            return [obs_input]

    obs_list = get_observation_list(args.obs_input)

    # Call the flagging function
    flagging_raw_fits(
        obs_list,
        raw_directory=args.data_dir,
        finished_file=args.finished_file,
        result_directory=args.results_dir,
        polr=args.polarization,
        file_flagged=None,
        file_outliers=args.outlier_file,
        winsor_step=args.winsor_integration,
        uvfits_step=args.uvfits_integration
    )

