# scripts/flag_data.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":

    import argparse
    from grandwin.flagger import flag_uvfits_data
    from grandwin.io.load_outliers import load_outliers_from_h5

    parser = argparse.ArgumentParser(description="Flag UVFITS files based on winsorizing outliers.")
    parser.add_argument("-d", "--data_dir", required=True)
    parser.add_argument("-r", "--results_dir", required=True)
    parser.add_argument("-of", "--outlier_file", required=True)
    parser.add_argument("-oi", "--obs_id", nargs='*', help="List of observation IDs to process (optional)")
    parser.add_argument("-pol", "--polarization", nargs='*', default=['XX', 'YY'])
    parser.add_argument("-wi", "--win_integration", type=int, required=True)
    parser.add_argument("-ui", "--uvfits_integration", type=int, required=True)
    parser.add_argument("-min", "--min_antenna", type=int, required=True)

    args = parser.parse_args()

    print("Set all arguments ...")
    print(args.results_dir, args.obs_id, args.polarization, args.win_integration, args.uvfits_integration)
    print("Import outliers file ...")

    df_outliers = load_outliers_from_h5(args.outlier_file, args.polarization)

    print("Generate observation id ...")

    if args.obs_id:
        unique_obs_ids = args.obs_id
    else:
        unique_obs_ids = df_outliers["obs_id"].unique().tolist()

    for obs_id in unique_obs_ids:
        uvfits_path = os.path.join(args.data_dir, f"{obs_id}_w_no_flags059-078.uvfits")
        output_path = os.path.join(args.results_dir, f"{obs_id}_w_no_flags059-078_flagged.uvfits")

        sel_df_outliers = df_outliers[df_outliers['obs_id'] == int(obs_id)]['frequency'].value_counts().to_frame().reset_index()
        sel_df_outliers = sel_df_outliers[sel_df_outliers['count'] > args.min_antenna]['frequency'].to_list()

        print(f"Run flagging for {obs_id} ...")

        flag_uvfits_data(
            obs_id,
            uvfits_path,
            df_outliers[(df_outliers["obs_id"] == int(obs_id)) & (df_outliers['frequency'].isin(sel_df_outliers))].reset_index(drop=True),
            output_path,
            args.win_integration,
            args.uvfits_integration,
            args.results_dir
        )
