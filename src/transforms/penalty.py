import pandas as pd

from src.transforms.averages import dynamic_window_rolling_average


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