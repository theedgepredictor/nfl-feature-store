import datetime

import numpy as np
import pandas as pd

###########################################################
## Loaders
###########################################################
from nfl_data_loader.workflows.components.events.game import GameComponent
from nfl_data_loader.workflows.components.teams.team import TeamComponent
from nfl_data_loader.workflows.transforms.events.ranks import make_rank_cols


def make_event_regular_season_feature_store(load_seasons):
    g_component = GameComponent(load_seasons, season_type='REG')
    game_features_df = g_component.df.copy()
    del g_component
    
    t_component = TeamComponent(load_seasons, season_type='REG')
    team_features_df = t_component.df.copy()
    del t_component
    
    df = game_features_df.merge(
        team_features_df.rename(columns={'team': 'home_team'}),
        on=['home_team', 'season', 'week'],
        how='left'
    ).merge(
        team_features_df.rename(columns={'team': 'away_team'}),
        on=['away_team', 'season', 'week'],
        how='left',
        suffixes=('_home', '_away')
    )

    # Suffix to prefix
    df.columns = [
        'home_' + col.replace('_home', '') if '_home' in col and 'actual_' not in col else
        'away_' + col.replace('_away', '') if '_away' in col and 'actual_' not in col else
        col
        for col in df.columns
    ]

    # Remove where the spread or total line is missing and games havent happened yet
    df = df.dropna(subset=['actual_home_score', 'actual_away_score', 'spread_line', 'total_line'])

    # Make Inference set
    inference_df = game_features_df[((game_features_df.actual_home_score.isnull()) & (game_features_df.actual_away_score.isnull()) & (game_features_df.spread_line.notna()) & (game_features_df.total_line.notna()))].copy()
    latest_epa = team_features_df.groupby('team').nth(-1).drop(columns=['week', 'season'])

    inference_df = inference_df.merge(
        latest_epa.rename(columns={'team': 'home_team'}),
        on=['home_team'],
        how='left'
    ).merge(
        latest_epa.rename(columns={'team': 'away_team'}),
        on=['away_team'],
        how='left',
        suffixes=('_home', '_away')
    )

    # Suffix to prefix
    inference_df.columns = [
        'home_' + col.replace('_home', '') if '_home' in col and 'actual_' not in col else
        'away_' + col.replace('_away', '') if '_away' in col and 'actual_' not in col else
        col
        for col in inference_df.columns
    ]

    df = pd.concat([df, inference_df])

    # Remove the first week of the dataset since it is used as aggregate
    if df.season.min() <= 2002:
        df = df[~((df.season == df.season.min()) & (df.week == df.week.min()))].copy()

    ### Handle Rank Columns
    rank_df = make_rank_cols(df.copy())
    df = df.merge(rank_df, on=['season', 'week', 'away_team', 'home_team'], how='left')
    return df