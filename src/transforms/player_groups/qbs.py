import pandas as pd

#from src.transforms.player import make_player_avg_group_features
from src.transforms.ranks import calculate_ranks

QB_COLS = [
    'player_id',
    'completions',
    'attempts',
    'passing_yards',
    'passing_tds',
    'interceptions',
    'sacks',
    'sack_yards',
    'sack_fumbles',
    'sack_fumbles_lost',
    'passing_air_yards',
    'passing_yards_after_catch',
    'passing_first_downs',
    'passing_epa',
    'passing_2pt_conversions',
    'pacr',
    'dakota',
    'carries',
    'rushing_yards',
    'rushing_tds',
    'rushing_fumbles',
    'rushing_fumbles_lost',
    'rushing_first_downs',
    'rushing_epa',
    'rushing_2pt_conversions',
    'fantasy_points',
    'fantasy_points_ppr'
]

def _calculate_raw_passer_value(df):
    ## takes a df, with properly named fields and returns a series w/ VALUE ##
    ## formula for reference ##
    ## https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/ ##
    ##      -2.2 * Pass Attempts +
    ##         3.7 * Completions +
    ##       (Passing Yards / 5) +
    ##        11.3 * Passing TDs –
    ##      14.1 * Interceptions –
    ##          8 * Times Sacked –
    ##       1.1 * Rush Attempts +
    ##       0.6 * Rushing Yards +
    ##        15.9 * Rushing TDs
    return (
        -2.2 * df['attempts'] +
        3.7 * df['completions'] +
        (df['passing_yards'] / 5) +
        11.3 * df['passing_tds'] -
        14.1 * df['interceptions'] -
        8 * df['sacks'] -
        1.1 * df['carries'] +
        0.6 * df['rushing_yards'] +
        15.9 * df['rushing_tds']
    )

def _calculate_passer_rating(df):
    # Step 1: Calculate each component
    a = ((df['completion_percentage']) - 0.3) * 5
    b = ((df['yards_per_pass']) - 3) * 0.25
    c = (df['tds_per_pass']) * 20
    d = 2.375 - (df['interceptions_per_pass']) * 25

    # Step 2: Cap each value between 0 and 2.375
    a = a.clip(0, 2.375)
    b = b.clip(0, 2.375)
    c = c.clip(0, 2.375)
    d = d.clip(0, 2.375)

    # Step 3: Calculate passer rating
    passer_rating = ((a + b + c + d) / 6) * 100

    return passer_rating

def _column_transforms(df):
    """

    :param df:
    :return: df
    """
    df['played'] = 1

    ## Passing
    df['completion_percentage'] = df['completions'] / df['attempts']
    df['yards_per_pass'] = df['passing_yards'] / df['attempts']
    df['adj_yards_per_pass'] = (df['passing_yards'] + 20*df['passing_tds']-45*df['interceptions']) / (df['attempts'])
    df['tds_per_pass'] = df['passing_tds'] / df['attempts']
    df['interceptions_per_pass'] = df['interceptions'] / df['attempts']

    ## Rushing
    df['yards_per_rush'] = df['rushing_yards'] / df['carries']
    df['tds_per_rush'] = df['rushing_tds'] / df['carries']
    df['yards_per_rush'] = df['rushing_yards'] / df['carries']
    return df

def make_qb_career(df):
    df = _column_transforms(df)
    df = _calculate_passer_rating(df)
    df = _calculate_raw_passer_value(df)
    group_features_dict = {
        'completions': 'mean',
        'attempts': 'mean',
        'passing_yards': 'mean',
        'passing_tds': 'mean',
        'interceptions': 'mean',
        'passing_epa': 'mean',
        'dakota': 'mean',
        'carries': 'mean',
        'rushing_yards': 'mean',
        'rushing_tds': 'mean',
        'rushing_epa': 'mean',
        'fantasy_points_ppr': 'mean',
        # 'qbr': 'mean',
        'completion_percentage': 'mean',
        # 'yards_per_pass': 'mean',
        'VALUE_ELO': 'mean',
        'passer_rating': 'mean'
    }
    qb_frame = df[['player_id', 'season', 'week'] + list(group_features_dict.keys())]
    for mode in ['season', 'form']:
        attrs_df = None
        if qb_frame.shape[0] == 0:
            qb_frame = attrs_df
        else:
            qb_frame = pd.merge(qb_frame, attrs_df, on=['player_id', 'season', 'week'])

    group_features_rank_dict = {f'season_avg_{i}': 'max' for i, val in list(group_features_dict.items())}
    qb_frame_ranks = calculate_ranks(qb_frame[['player_id', 'season', 'week'] + list(group_features_rank_dict.keys())].fillna(0).copy(), group_by_col=['season', 'week'], rank_cols_methods=group_features_rank_dict)
    qb_frame = pd.merge(qb_frame, qb_frame_ranks, on=['player_id', 'season', 'week'])

    return qb_frame

