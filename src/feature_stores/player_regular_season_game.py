import datetime

import numpy as np
import pandas as pd

###########################################################
## Loaders
###########################################################
from src.components.game import GameComponent
from src.components.player import GamePlayerComponent
from src.components.player_stat import PlayerStatComponent
from src.components.team import TeamComponent
from src.transforms.ranks import make_rank_cols
from src.utils import get_dataframe


def make_off_player_regular_season_feature_store(load_seasons):
    #game_player_component = GamePlayerComponent(load_seasons, season_type='REG')
    #game_player_df = game_player_component.run_pipeline()
    #del game_player_component

    player_stat_component = PlayerStatComponent(load_seasons, season_type='REG', group='off')
    off_player_df = player_stat_component.run_pipeline()
    del player_stat_component

    #df = off_player_df.merge(game_player_df, on=['player_id','position_group', 'season', 'week'], how='left')
    #df = df.dropna(subset=['player_id'])

    ### Handle Future
    #inference_df = game_player_df[game_player_df.player_id.isnull()].copy()
    #latest_off_player_df = off_player_df[off_player_df].sort_values(['player_id', 'season', 'week']).drop_duplicates(['player_id'], keep='last')

    return off_player_df

if __name__ == '__main__':
    df = make_off_player_regular_season_feature_store([2022,2023,2024])

