import pandas as pd

from src.transforms.averages import dynamic_window_ewma
from src.transforms.general import make_avg_group_features


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