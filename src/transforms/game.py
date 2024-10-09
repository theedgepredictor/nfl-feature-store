import datetime

import numpy as np
import pandas as pd

from src.transforms.averages import dynamic_window_rolling_average


def make_weekly_avg_group_features(off_weekly, def_weekly):
    """
    Calculate dynamic window avg for multiple attributes (like epa, rushing_yards, etc.) for both offense and defense.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data. (Filter data prior to calling this function)
        group_features: List of attributes to calculate dynamic window avg

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive avg values for each attribute.
    """

    features = pd.DataFrame()
    group_features_list = [
        'fantasy_points',
        'fantasy_points_half_ppr',
        'fantasy_points_ppr',
        'total_plays',
        'total_yards',
        'total_fumbles',
        'total_fumbles_lost',
        'total_turnovers',
        'total_touchdowns',
        'total_first_downs',
        'touchdown_per_play',
        'yards_per_play',
        'fantasy_point_per_play',

        'completions',
        'attempts',
        'passing_yards',
        'passing_tds',
        'interceptions',
        'sacks',
        'sack_yards',
        # 'sack_fumbles',
        'sack_fumbles_lost',
        'passing_air_yards',
        'passing_yards_after_catch',
        'passing_first_downs',
        'passing_epa',
        # 'passing_2pt_conversions',
        'pacr',
        'dakota',
        'avg_time_to_throw',
        'avg_completed_air_yards',
        'avg_intended_air_yards_passing',
        'avg_air_yards_differential',
        'aggressiveness',
        'max_completed_air_distance',
        'avg_air_yards_to_sticks',
        'passer_rating',
        'completion_percentage',
        'expected_completion_percentage',
        'completion_percentage_above_expectation',
        'avg_air_distance',
        'max_air_distance',
        #'qbr',
        'air_yards_per_pass_attempt',
        'pass_to_rush_ratio',
        'pass_to_rush_first_down_ratio',
        'yards_per_pass_attempt',
        'sack_rate',

        'carries',
        'rushing_yards',
        'rushing_tds',
        # 'rushing_fumbles',
        'rushing_fumbles_lost',
        'rushing_first_downs',
        'rushing_epa',
        # 'rushing_2pt_conversions',
        'efficiency',
        'percent_attempts_gte_eight_defenders',
        'avg_time_to_los',
        # 'avg_rush_yards',
        'expected_rush_yards',
        'rush_yards_over_expected',
        'rush_yards_over_expected_per_att',
        'rush_pct_over_expected',
        'yards_per_rush_attempt',

        # 'receptions',
        # 'targets',
        # 'receiving_yards',
        # 'receiving_tds',
        # 'receiving_fumbles',
        # 'receiving_fumbles_lost',
        # 'receiving_air_yards',
        # 'receiving_yards_after_catch',
        # 'receiving_first_downs',
        # 'receiving_epa',
        # 'receiving_2pt_conversions',
        # 'racr',
        # 'target_share',
        # 'air_yards_share',
        # 'wopr',
        'avg_cushion',
        'avg_separation',
        'avg_intended_air_yards_receiving',
        # 'percent_share_of_intended_air_yards',
        # 'catch_percentage',
        # 'avg_yac',
        # 'avg_expected_yac',
        'avg_yac_above_expectation',
        # 'special_teams_tds',
    ]
    for attr in group_features_list:
        # Separate attribute values into  offense and defense
        offense = off_weekly.rename(columns={'team': 'posteam'}).groupby(['posteam', 'season', 'week'], as_index=False).agg({attr: 'sum'})
        defense = def_weekly.rename(columns={'team': 'defteam'}).groupby(['defteam', 'season', 'week'], as_index=False).agg({attr: 'sum'})

        # Lag attribute one period back
        offense[f'{attr}_shifted'] = offense.groupby('posteam')[attr].shift()
        defense[f'{attr}_shifted'] = defense.groupby('defteam')[attr].shift()

        # Calculate dynamic window EWMA for the attribute
        offense[f"avg_{attr.replace('avg_', '')}"] = offense.groupby('posteam').apply(dynamic_window_rolling_average, attr).values
        defense[f"avg_{attr.replace('avg_', '')}"] = defense.groupby('defteam').apply(dynamic_window_rolling_average, attr).values

        # Merge offense and defense attributes
        avgs = offense.rename(columns={'posteam': 'team'}).merge(
            defense.rename(columns={'defteam': 'team'}),
            on=['team', 'season', 'week'],
            suffixes=('_offense', '_defense')
        )
        avgs = avgs[['team', 'season', 'week', f"avg_{attr.replace('avg_', '')}_offense", f"avg_{attr.replace('avg_', '')}_defense"]]
        # Collect features for this attribute
        if features.shape[0] == 0:
            features = avgs
        else:
            features = pd.merge(features, avgs, on=['team', 'season', 'week'])

    return features.drop_duplicates(subset=['team', 'season', 'week'])

