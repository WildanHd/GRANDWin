# grandwin/metadata_parser.py
# This script designated to create a list of observation id which will be used for running the script depending on the observation day and pointing center

import pandas as pd
import datetime as dt

def observation_id_preparation(observation_file):
    df = pd.read_csv(observation_file, header=0, engine='python')
    df['date'] = df.starttime_utc.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z").date())
    df['date_time'] = df.starttime_utc.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z"))
    df['partition'] = pd.factorize(df['date'])[0] + 1
    df = df[df['partition'] == 1].reset_index(drop=True)

    grouped = df.groupby(["partition", "gridpoint_number"])["obs_id"].apply(list).reset_index()
    task_list = [(row["partition"], row["gridpoint_number"], row["obs_id"]) for _, row in grouped.iterrows()]

    return task_list
