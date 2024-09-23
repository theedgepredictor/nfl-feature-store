import datetime

import numpy as np
import pandas as pd

from src.utils import df_rename_fold


def dynamic_window_ewma(x):
    """
    Calculate rolling exponentially weighted features with a dynamic window size
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x.epa_shifted[:i + 1]
        if row.week > 10:
            values[i] = epa.ewm(min_periods=1, span=row.week).mean().values[-1]
        else:
            values[i] = epa.ewm(min_periods=1, span=10).mean().values[-1]

    return pd.Series(values, index=x.index)


def dynamic_window_rolling_average(x, attr):
    """
    Calculate rolling features with a dynamic window size for the specified attribute.

    Parameters:
        x (DataFrame): DataFrame containing the play-by-play data grouped by team.
        attr (str): The attribute for which rolling average is calculated.

    Returns:
        pd.Series: Series with the dynamic rolling EWMA for the attribute.
    """
    values = np.zeros(len(x))
    attr_shifted = f'{attr}_shifted'

    for i, (_, row) in enumerate(x.iterrows()):
        attr_data = x[attr_shifted][:i + 1]
        if row['week'] != 1:
            values[i] = attr_data.rolling(min_periods=1, window=row['week'] - 1).mean().values[-1]
        else:
            # Handle edge case for the first week or season start
            values[i] = attr_data.rolling(min_periods=1, window=18 if row['season'] >= 2021 else 17).mean().values[-1]

    return pd.Series(values, index=x.index)


def make_rushing_epa(data):
    """
    Calculate the rushing EPA for both offense and defense with dynamic window EWMA.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive rushing EPA values.
    """
    # Separate EPA into rushing offense and defense
    rushing_offense_epa = data.loc[(data['play_type'].isin(['run', 'qb_kneel'])), :] \
        .groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()

    rushing_defense_epa = data.loc[data['rush_attempt'] == 1, :] \
        .groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()

    # Lag EPA one period back
    rushing_offense_epa['epa_shifted'] = rushing_offense_epa.groupby('posteam')['epa'].shift()
    rushing_defense_epa['epa_shifted'] = rushing_defense_epa.groupby('defteam')['epa'].shift()

    # Calculate dynamic window EWMA
    rushing_offense_epa['ewma_rushing'] = rushing_offense_epa.groupby('posteam') \
        .apply(dynamic_window_ewma).values

    rushing_defense_epa['ewma_rushing'] = rushing_defense_epa.groupby('defteam') \
        .apply(dynamic_window_ewma).values

    # Merge offense and defense EPA
    rushing_epa = rushing_offense_epa.rename(columns={'posteam': 'team'}).merge(
        rushing_defense_epa.rename(columns={'defteam': 'team'}),
        on=['team', 'season', 'week'],
        suffixes=('_offense', '_defense')
    )
    features = [column for column in rushing_epa.columns if 'ewma' in column] + ['team', 'season', 'week']

    return rushing_epa[features]


def make_passing_epa(data):
    """
    Calculate the passing EPA for both offense and defense with dynamic window EWMA.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive passing EPA values.
    """
    # Separate EPA into passing offense and defense
    passing_offense_epa = data.loc[(data['play_type'].isin(['pass', 'qb_spike'])), :] \
        .groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()

    passing_defense_epa = data.loc[(data['play_type'].isin(['pass', 'qb_spike'])), :] \
        .groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()

    # Lag EPA one period back
    passing_offense_epa['epa_shifted'] = passing_offense_epa.groupby('posteam')['epa'].shift()
    passing_defense_epa['epa_shifted'] = passing_defense_epa.groupby('defteam')['epa'].shift()

    passing_offense_epa['ewma_passing'] = passing_offense_epa.groupby('posteam') \
        .apply(dynamic_window_ewma).values

    passing_defense_epa['ewma_passing'] = passing_defense_epa.groupby('defteam') \
        .apply(dynamic_window_ewma).values

    # Merge offense and defense EPA
    passing_epa = passing_offense_epa.rename(columns={'posteam': 'team'}).merge(
        passing_defense_epa.rename(columns={'defteam': 'team'}),
        on=['team', 'season', 'week'],
        suffixes=('_offense', '_defense')
    )
    features = [column for column in passing_epa.columns if 'ewma' in column] + ['team', 'season', 'week']

    return passing_epa[features]

def make_score_feature(data):
    """
    Calculate the score for both offense and defense with dynamic window EWMA.

    Parameters:
        schedule (DataFrame): Dataframe containing the schedule and scores.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive score values with EWMA.
    """
    # Separate EPA into passing offense and defense
    passing_offense_epa = data[
        (~data['down'].isna()) &
        (data['play_type'].isin(['pass', 'qb_kneel', 'qb_spike', 'run']))
        ].copy() \
        .groupby(['posteam', 'season', 'week'], as_index=False)['posteam_score_post'].last()

    passing_defense_epa = data[
        (~data['down'].isna()) &
        (data['play_type'].isin(['pass', 'qb_kneel', 'qb_spike', 'run']))
        ].copy() \
        .groupby(['defteam', 'season', 'week'], as_index=False)['defteam_score_post'].last()

    # Lag EPA one period back
    passing_offense_epa['epa_shifted'] = passing_offense_epa.groupby('posteam')['posteam_score_post'].shift()
    passing_defense_epa['epa_shifted'] = passing_defense_epa.groupby('defteam')['defteam_score_post'].shift()

    passing_offense_epa['ewma_score'] = passing_offense_epa.groupby('posteam') \
        .apply(dynamic_window_ewma).values

    passing_defense_epa['ewma_score'] = passing_defense_epa.groupby('defteam') \
        .apply(dynamic_window_ewma).values

    # Merge offense and defense EPA
    passing_epa = passing_offense_epa.rename(columns={'posteam': 'team'}).merge(
        passing_defense_epa.rename(columns={'defteam': 'team'}),
        on=['team', 'season', 'week'],
        suffixes=('_offense', '_defense')
    )
    features = [column for column in passing_epa.columns if 'ewma' in column] + ['team', 'season', 'week']
    return passing_epa[features]


def make_cover_feature(schedule):
    """
    Calculate the cover feature for both the team (home or away) and whether the game went under.

    Parameters:
        schedule (DataFrame): DataFrame containing the schedule, scores, spread, and total line.

    Returns:
        DataFrame: DataFrame with added columns for rolling average of team covering and under cover.
    """
    # Calculate if the away team covered the spread and if the game went under
    schedule['away_team_covered'] = (schedule['away_score'] + schedule['spread_line'] >= schedule['home_score']).astype(int)
    schedule['home_team_covered'] = (schedule['home_score'] - schedule['spread_line'] >= schedule['away_score']).astype(int)
    schedule['under_covered'] = (schedule['home_score'] + schedule['away_score'] <= schedule['total_line']).astype(int)

    folded_df = schedule.drop(columns=['away_team_win', 'away_team_spread', 'total_target'])
    folded_df['ishome'] = folded_df['home_team']
    # Fold the DataFrame to treat home and away teams equally
    folded_df = df_rename_fold(folded_df, 'home_', 'away_')

    # Sort by team, season, and week
    folded_df = folded_df.sort_values(by=['team', 'season', 'week']).reset_index(drop=True)
    folded_df = folded_df.drop_duplicates(['season', 'week', 'team'])
    folded_df['ishome'] = folded_df['ishome'] == folded_df['team']

    # Calculate the rolling average of the last 10 games for covering the spread
    folded_df['rolling_team_cover'] = folded_df.groupby('team')['team_covered'].shift(1).rolling(10, min_periods=1).mean().reset_index(drop=True)

    # Calculate the rolling average of the last 10 games for going under the total
    folded_df['rolling_under_cover'] = folded_df.groupby('team')['under_covered'].shift(1).rolling(10, min_periods=1).mean().reset_index(drop=True)
    home_a = folded_df[folded_df.ishome == True][['season', 'week', 'team', 'rolling_team_cover', 'rolling_under_cover']].rename(columns={'team': 'home_team', 'rolling_team_cover': 'home_rolling_spread_cover', 'rolling_under_cover': 'home_rolling_under_cover'})
    away_a = folded_df[folded_df.ishome == False][['season', 'week', 'team', 'rolling_team_cover', 'rolling_under_cover']].rename(columns={'team': 'away_team', 'rolling_team_cover': 'away_rolling_spread_cover', 'rolling_under_cover': 'away_rolling_under_cover'})
    return away_a, home_a


def make_avg_penalty_group_features(data):
    """
    Calculate dynamic window avg for penalty attributes for both offense and defense.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data. (Filter data prior to calling this function)

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive penalty avg values.
    """
    data['offensive_penalty'] = data['penalty_team'] == data['posteam']
    data['defensive_penalty'] = data['penalty_team'] == data['defteam']
    data['offensive_penalty_yards'] = data['penalty_yards'] * data['offensive_penalty']
    data['defensive_penalty_yards'] = data['penalty_yards'] * data['defensive_penalty']
    features = pd.DataFrame()
    group_features_dict = {
        'offensive_penalty_yards': 'sum',
        'defensive_penalty_yards': 'sum',
        'offensive_penalty': 'sum',
        'defensive_penalty': 'sum'
    }
    for attr, agg_method in group_features_dict.items():
        penalty_df = data.groupby(['penalty_team', 'season', 'week'], as_index=False).agg({attr: agg_method})
        penalty_df[f'{attr}_shifted'] = penalty_df.groupby('penalty_team')[attr].shift()
        penalty_df[f'avg_{attr}'] = penalty_df.groupby('penalty_team').apply(dynamic_window_rolling_average, attr).values

        avgs = penalty_df[['penalty_team', 'season', 'week', f'avg_{attr}']].rename(columns={'penalty_team': 'team'})
        # Collect features for this attribute
        if features.shape[0] == 0:
            features = avgs
        else:
            features = pd.merge(features, avgs, on=['team', 'season', 'week'])

    return features


def make_avg_group_features(data, group_features_dict):
    """
    Calculate dynamic window avg for multiple attributes (like epa, rushing_yards, etc.) for both offense and defense.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data. (Filter data prior to calling this function)
        group_features: List of attributes to calculate dynamic window avg

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive avg values for each attribute.
    """

    features = pd.DataFrame()

    for attr, agg_method in group_features_dict.items():
        # Separate attribute values into rushing offense and defense
        offense = data.groupby(['posteam', 'season', 'week'], as_index=False).agg({attr: agg_method})
        defense = data.groupby(['defteam', 'season', 'week'], as_index=False).agg({attr: agg_method})

        # Lag attribute one period back
        offense[f'{attr}_shifted'] = offense.groupby('posteam')[attr].shift()
        defense[f'{attr}_shifted'] = defense.groupby('defteam')[attr].shift()

        # Calculate dynamic window EWMA for the attribute
        offense[f'avg_{attr}'] = offense.groupby('posteam').apply(dynamic_window_rolling_average, attr).values
        defense[f'avg_{attr}'] = defense.groupby('defteam').apply(dynamic_window_rolling_average, attr).values

        # Merge offense and defense attributes
        avgs = offense.rename(columns={'posteam': 'team'}).merge(
            defense.rename(columns={'defteam': 'team'}),
            on=['team', 'season', 'week'],
            suffixes=('_offense', '_defense')
        )
        avgs = avgs[['team', 'season', 'week', f'avg_{attr}_offense', f'avg_{attr}_defense']]
        # Collect features for this attribute
        if features.shape[0] == 0:
            features = avgs
        else:
            features = pd.merge(features, avgs, on=['team', 'season', 'week'])

    return features.drop_duplicates(subset=['team', 'season', 'week'])


def make_normal_play_group_features(data):
    ## Create General features for rolling avgs
    general_features_dict = {
        'first_down': 'sum',
        'third_down_converted': 'sum',
        'third_down_failed': 'sum',
        'fourth_down_converted': 'sum',
        'fourth_down_failed': 'sum',

        # Penalty features
        'first_down_penalty': 'sum',
        # 'penalty_yards': 'sum',
        # 'penalty': 'sum',
        # 'defensive_penalty': 'sum',

        # Add advanced general features here

        'shotgun': 'sum',
        'no_huddle': 'sum',
        'qb_dropback': 'sum',
        'qb_scramble': 'sum',
        'goal_to_go': 'sum',
        'is_redzone': 'sum',
        #'successful_two_point_conversion': 'sum',
        #'drive': 'nunique',  # Number of unique drives in the quarter
        #'series': 'nunique',  # Number of unique series in the quarter
    }
    general_features = data[
        (~data['down'].isna()) &
        (data['play_type'].isin(['pass', 'qb_kneel', 'qb_spike', 'run']))
        ].copy()
    general_features = make_avg_group_features(general_features, general_features_dict)
    ## make down percentages
    general_features['avg_third_down_percentage_offense'] = general_features.apply(
        lambda row: row['avg_third_down_converted_offense'] / (row['avg_third_down_converted_offense'] + row['avg_third_down_failed_offense'])
        if (row['avg_third_down_converted_offense'] + row['avg_third_down_failed_offense']) > 0 else 0,
        axis=1
    )
    general_features['avg_third_down_percentage_defense'] = general_features.apply(
        lambda row: row['avg_third_down_converted_defense'] / (row['avg_third_down_converted_defense'] + row['avg_third_down_failed_defense'])
        if (row['avg_third_down_converted_defense'] + row['avg_third_down_failed_defense']) > 0 else 0,
        axis=1
    )

    general_features['avg_fourth_down_percentage_offense'] = general_features.apply(
        lambda row: row['avg_fourth_down_converted_offense'] / (row['avg_fourth_down_converted_offense'] + row['avg_fourth_down_failed_offense'])
        if (row['avg_fourth_down_converted_offense'] + row['avg_fourth_down_failed_offense']) > 0 else 0,
        axis=1
    )
    general_features['avg_fourth_down_percentage_defense'] = general_features.apply(
        lambda row: row['avg_fourth_down_converted_defense'] / (row['avg_fourth_down_converted_defense'] + row['avg_fourth_down_failed_defense'])
        if (row['avg_fourth_down_converted_defense'] + row['avg_fourth_down_failed_defense']) > 0 else 0,
        axis=1
    )

    ## rename columns
    general_features = general_features.rename(columns={
        'avg_posteam_score_post_offense': 'avg_points_offense',
        'avg_posteam_score_post_defense': 'avg_points_defense',
        'avg_score_differential_post_offense': 'avg_point_differential_offense',
        'avg_score_differential_post_defense': 'avg_point_differential_defense',
    })
    return general_features

def make_general_group_features(data):
    """
    Unfiltered play by play data features for offense and defense

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive avg values.
    """

    #### Handles time of possession for offense and defense

    group_features_dict = {
        'posteam_score_post': 'last',
        'score_differential_post': 'last',
        'epa': 'sum',
        'wpa': 'sum',
        'time_of_possession': 'sum',
        'field_goal_made': 'sum',
        'field_goal_attempt':'sum',
        'field_goal_distance': 'mean',
        'extra_point_made': 'sum',
        'extra_point_attempt': 'sum',
        'turnover':'sum',
    }
    data['turnover'] = data['fumble_lost'] + data['interception']
    data['game_seconds_remaining'] = data['game_seconds_remaining'].fillna(0)
    # For each play in the game calculate the difference in the clock
    data['time_of_possession'] = data.groupby('game_id')['game_seconds_remaining'].diff(-1).abs()
    data['field_goal_made'] = data['field_goal_result'] == 'made'
    data['extra_point_made'] = data['extra_point_result'] == 'made'
    data['field_goal_distance'] = None
    data.loc[data['field_goal_attempt']==1, 'field_goal_distance'] = data.loc[data['field_goal_attempt']==1, 'kick_distance']

    general_features = make_avg_group_features(data, group_features_dict)
    general_features['avg_field_goal_percentage_offense'] = general_features.apply(
        lambda row: row['avg_field_goal_made_offense'] / (row['avg_field_goal_attempt_offense'])
        if (row['avg_field_goal_attempt_offense']) > 0 else 0,
        axis=1
    )
    general_features['avg_field_goal_percentage_defense'] = general_features.apply(
        lambda row: row['avg_field_goal_made_defense'] / (row['avg_field_goal_attempt_defense'])
        if (row['avg_field_goal_attempt_defense']) > 0 else 0,
        axis=1
    )
    general_features['avg_extra_point_percentage_offense'] = general_features.apply(
        lambda row: row['avg_extra_point_made_offense'] / (row['avg_extra_point_attempt_offense'])
        if (row['avg_extra_point_attempt_offense']) > 0 else 0,
        axis=1
    )
    general_features['avg_extra_point_percentage_defense'] = general_features.apply(
        lambda row: row['avg_extra_point_made_defense'] / (row['avg_extra_point_attempt_defense'])
        if (row['avg_extra_point_attempt_defense']) > 0 else 0,
        axis=1
    )
    ## rename columns
    general_features = general_features.rename(columns={
        'avg_posteam_score_post_offense': 'avg_points_offense',
        'avg_posteam_score_post_defense': 'avg_points_defense',
        'avg_score_differential_post_offense': 'avg_point_differential_offense',
        'avg_score_differential_post_defense': 'avg_point_differential_defense',
    })

    return general_features.drop_duplicates(subset=['team', 'season', 'week'])

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
        'qbr',
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


def calculate_ranks(df, group_by_col, rank_cols_methods):
    """
    Calculate ranks for specified columns in a DataFrame grouped by a column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        group_by_col (str): The column name to group by (e.g., 'week').
        rank_cols_methods (dict): A dictionary where keys are column names,
                                  and values are the ranking methods ('max' or 'min').

    Returns:
        pd.DataFrame: The original DataFrame with new rank columns.
    """
    rank_df_list = []

    # Calculate ranks for each column based on its corresponding method
    for col, method in rank_cols_methods.items():
        # Rank for each column based on its specific method
        rank_col_df = df.groupby(group_by_col, as_index=False)[col].rank(method=method, ascending=(method == 'min')).astype(int)
        rank_df_list.append(rank_col_df)  # Keep only the rank column

    # Combine all rank columns
    rank_df = pd.concat(rank_df_list, axis=1)

    # Add suffix to rank columns
    rank_df.columns = [f"{col}_rank" for col in rank_cols_methods.keys()]

    # Concatenate the rank columns back to the original dataframe
    df = pd.concat([df.reset_index(drop=True), rank_df.reset_index(drop=True)], axis=1)

    return df.drop(columns=[col for col in rank_cols_methods.keys()])


# Specify which columns should use 'max' and which should use 'min'


def make_rank_cols(team_fs):
    team_fs = team_fs.rename(columns={'spread_line': 'away_spread_line'})
    team_fs['home_spread_line'] = -team_fs['away_spread_line']
    team_fs['home_team_spread'] = -team_fs['away_team_spread']
    team_fs['home_team_win'] = team_fs['away_team_win'] == 0
    team_fs['home_team_covered_spread'] = team_fs['away_team_covered_spread'] == 0
    team_fs['ishome'] = team_fs['home_team']
    team_fs_df = df_rename_fold(team_fs, 'away_', 'home_')
    team_fs_df['ishome'] = team_fs_df['ishome'] == team_fs_df['team']
    df = team_fs[['away_team', 'home_team', 'week', 'season']].copy()
    rank_cols_methods_offense = {
        'elo_pre': 'max',
        'avg_points_offense': 'max',
        'avg_rushing_yards_offense': 'max',
        'avg_passing_yards_offense': 'max',
        'avg_total_yards_offense': 'max',
        'avg_yards_per_play_offense': 'max',
        'avg_total_turnovers_offense': 'min'  # Use 'min' for turnovers
    }

    rank_cols_methods_defense = {
        'avg_points_defense': 'min',
        'avg_rushing_yards_defense': 'min',
        'avg_passing_yards_defense': 'min',
        'avg_total_yards_defense': 'min',
        'avg_yards_per_play_defense': 'min',
        'avg_total_turnovers_defense': 'max'
    }

    # Calculate offensive ranks
    offensive_ranks = calculate_ranks(team_fs_df[['team', 'week', 'season'] + list(rank_cols_methods_offense.keys())], 'week', rank_cols_methods_offense)
    o_ranks = offensive_ranks.copy()
    o_ranks['offensive'] = o_ranks['avg_points_offense_rank'] + o_ranks['avg_total_yards_offense_rank']
    o_ranks = calculate_ranks(o_ranks[['team', 'week', 'season', 'offensive']], 'week', {'offensive': 'min'})
    defensive_ranks = calculate_ranks(team_fs_df[['team', 'week', 'season'] + list(rank_cols_methods_defense.keys())], 'week', rank_cols_methods_defense)
    d_ranks = defensive_ranks.copy()
    d_ranks['defensive'] = d_ranks['avg_points_defense_rank'] + d_ranks['avg_total_yards_defense_rank']
    d_ranks = calculate_ranks(d_ranks[['team', 'week', 'season', 'defensive']], 'week', {'defensive': 'min'})

    full_rank = pd.merge(offensive_ranks, defensive_ranks, on=['team', 'week', 'season'])
    full_rank = pd.merge(full_rank, o_ranks, on=['team', 'week', 'season'])
    full_rank = pd.merge(full_rank, d_ranks, on=['team', 'week', 'season'])
    full_rank['net_rank'] = (full_rank['offensive_rank'] + full_rank['defensive_rank']) / 2
    full_rank = full_rank.sort_values(by=['season', 'week', 'offensive_rank', ]).reset_index(drop=True)
    away_fs = full_rank.copy()
    away_fs.columns = ['away_' + col if col not in ['week', 'season'] else col for col in away_fs.columns]
    home_fs = full_rank.copy()
    home_fs.columns = ['home_' + col if col not in ['week', 'season'] else col for col in home_fs.columns]
    df = pd.merge(df, away_fs, on=['away_team', 'week', 'season'])
    df = pd.merge(df, home_fs, on=['home_team', 'week', 'season'])
    return df


def make_qtr_score_group_features(df):
    """
    Calculate score for a given groupby_cols. Uses the last value since play by play is sorted
    """
    groupby_cols = ['game_id', 'posteam', 'season', 'week', 'qtr']
    score = df[groupby_cols + ['posteam_score_post', 'defteam_score_post']].copy()
    score = score.groupby(groupby_cols).nth(-1)
    score = score.sort_values(groupby_cols)
    score['posteam_score_post'] = score.groupby(['game_id', 'posteam'])['posteam_score_post'].diff().fillna(score['posteam_score_post'])
    score['defteam_score_post'] = score.groupby(['game_id', 'posteam'])['defteam_score_post'].diff().fillna(score['defteam_score_post'])
    score['point_diff'] = score['posteam_score_post'] - score['defteam_score_post']
    score = score.rename(columns={'posteam_score_post': 'points', 'defteam_score_post': 'defteam_score'})
    score[['points', 'point_diff']] = score[['points', 'point_diff']].astype(int)
    score = score.drop(columns=['defteam_score'])

    # Pivot the table to create columns for each quarter
    score = score.reset_index()  # Reset index for easier manipulation
    score_pivot = score.pivot_table(index=['game_id', 'posteam', 'season', 'week'],
                                    columns='qtr',
                                    values=['points', 'point_diff'],
                                    aggfunc='first').fillna(0)

    # Flatten the multi-level column names
    score_pivot.columns = [f'q{int(qtr)}_{metric}' for metric, qtr in score_pivot.columns]

    # Reset index to get a clean DataFrame
    score_pivot = score_pivot.reset_index().drop(columns=['game_id'])
    score_pivot = pd.merge(df[['season', 'week', 'posteam', 'defteam']], score_pivot, on=['season', 'week', 'posteam', ], how='left')
    group_features_dict = {
        'q1_point_diff': 'mean',
        'q2_point_diff': 'mean',
        'q3_point_diff': 'mean',
        'q4_point_diff': 'mean',
        'q5_point_diff': 'mean',
        'q1_points': 'mean',
        'q2_points': 'mean',
        'q3_points': 'mean',
        'q4_points': 'mean',
        'q5_points': 'mean',
    }
    features = make_avg_group_features(score_pivot, group_features_dict)
    return features