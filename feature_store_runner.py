import os
import pyarrow as pa
from src.feature_stores.player_season import make_season_feature_store
from src.utils import put_dataframe, get_dataframe


def main():
    root_path = './data/feature_store'

    fs_df = make_season_feature_store()
    fs_type_path = 'player/season'
    fs_file_name = 'fs.parquet'
    path = f"{root_path}/{fs_type_path}/{fs_file_name}"

    put_dataframe(fs_df, path)

    fs_df = get_dataframe(path, columns=['player_id','display_name','position_group','total_last_year_position_rank','total_last_year_fantasy_points','total_last_year_games_played'])



if __name__ == '__main__':
    main()