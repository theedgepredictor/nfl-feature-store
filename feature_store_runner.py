import os

import pandas as pd
import pyarrow as pa

from src.feature_stores.event_regular_season_game import make_event_regular_season_feature_store
from src.utils import put_dataframe, get_seasons_to_update


event_meta = {
    "name":'event/regular_season_game',
    "start_season": 2002,
    "obj": make_event_regular_season_feature_store
    }
FEATURE_STORE_METAS = [
    #player_meta,
    event_meta
]



def main():

    root_path = './data/feature_store'
    for fs_meta_obj in FEATURE_STORE_METAS:
        feature_store_name = fs_meta_obj['name']
        start_season = fs_meta_obj['start_season']
        ## Determine pump mode
        update_seasons = get_seasons_to_update(root_path, feature_store_name)
        #update_seasons = [2002,2003]
        mode = 'refresh' if start_season in update_seasons else 'upsert'

        # Use the last 2 seasons for aggregate stats for upsert mode
        load_seasons = update_seasons if mode == 'refresh' else list(range(min(update_seasons) - 2, max(update_seasons)+1))

        print(f"Running Feature Store: {feature_store_name} from {min(update_seasons)}-{max(update_seasons)} (loads: {min(load_seasons)}-{max(load_seasons)})")

        fs_df = fs_meta_obj['obj'](load_seasons)
        print(f"Adds: {round(fs_df.memory_usage(deep=True).sum() / (1024 ** 2), 2)} MB to the Feature Store")
        for season in update_seasons:
            put_dataframe(fs_df[fs_df.season==season].copy(), f"{root_path}/{feature_store_name}/{season}.parquet")

if __name__ == '__main__':
    main()