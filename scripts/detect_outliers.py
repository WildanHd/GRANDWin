# scripts/detect_outliers.py
# The main script to add several arguments to the python script

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":

    import argparse
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    from grandwin.metadata_parser import observation_id_preparation
    from grandwin.winsorize_detection import winsorizing_outlier_detection_3d

    parser = argparse.ArgumentParser(description="Process observation data in parallel.")
    parser.add_argument("-d", "--data_directory", required=True, help="Directory containing hyperdrive calibration solutions")
    parser.add_argument("-r", "--results_directory", required=True, help="Directory to store the calculation results")
    parser.add_argument("-oi", "--obs_file", required=True, help="Directory containing the observation id file")
    parser.add_argument("-it", "--integration_time", type=int, required=True, help="Integration time per each timeblock second ex. 2, 8")
    parser.add_argument("-t", "--type", required=True, help="The data type that consist of amplitude, phase, real and imaginary part")
    parser.add_argument("-ir", "--iter", required=True, help="Number of iteration to find final gamma value ex. 10, 100")
    parser.add_argument("-ito", "--iterthresh", required=True, help="Iteration threshold", default=3.0)
    parser.add_argument("-fto", "--finalthresh", required=True, help="Final threshold used for outliers detection", default=5.0)
    parser.add_argument("-g", "--gamma", type=float, default=0.02)
    parser.add_argument("-p", "--day_partition", type=int, default=1)
    parser.add_argument("-gp", "--grid_point", type=int, default=0)

    args = parser.parse_args()

    print("Generate arguments!")

    data_directory = args.data_directory
    results_directory = args.results_directory
    observation_file = args.obs_file

    integration_time = args.integration_time
    data_type = args.type
    iter = args.iter
    iter_threshold = args.iterthresh 
    final_threshold = args.finalthresh
    gamma = args.gamma
    partition = args.day_partition
    grid_point = args.grid_point

    print(f"... data directory: {data_directory}")
    print(f"... results directory: {results_directory}")

    print("Generate task list!")

    task_list = observation_id_preparation(observation_file, partition, grid_point)

    print("Task list: ", task_list)

    print("Run multiprocessing!")

    # Run multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(winsorizing_outlier_detection_3d, [(obs_days, grid, obs_list, data_directory, results_directory, integration_time, data_type, iter, iter_threshold, final_threshold, gamma, partition, grid_point) for obs_days, grid, obs_list in task_list])

    print("Detection completed!", flush=True)