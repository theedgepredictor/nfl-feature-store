
#####################################################################
## Stat Transforms that can be used for Players or Event level stats
# or just general useful transforms
#####################################################################
import pandas as pd

from src.extracts.player_stats import collect_weekly_ngs_receiving_data, collect_weekly_ngs_rushing_data, collect_weekly_ngs_passing_data, collect_weekly_espn_player_stats
from src.transforms.averages import dynamic_window_rolling_average


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
    b = ((df['yards_per_pass_attempt']) - 3) * 0.25
    c = (df['passing_tds'] / df['attempts']) * 20
    d = 2.375 - (df['interceptions'] / df['attempts']) * 25

    # Step 2: Cap each value between 0 and 2.375
    a = a.clip(0, 2.375)
    b = b.clip(0, 2.375)
    c = c.clip(0, 2.375)
    d = d.clip(0, 2.375)

    # Step 3: Calculate passer rating
    passer_rating = ((a + b + c + d) / 6) * 100

    return passer_rating

def stat_collection(year, season_type="REG", mode='team'):
    TEAM_STATS = [
        'team',
        'season',
        'week',
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
        'VALUE_ELO',
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

    passing_stats = [
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
        'avg_time_to_throw',
        'avg_completed_air_yards',
        'avg_intended_air_yards_passing',
        'avg_air_yards_differential',
        'aggressiveness',
        'max_completed_air_distance',
        'avg_air_yards_to_sticks',
        #'completion_percentage',
        'expected_completion_percentage',
        'completion_percentage_above_expectation',
        'avg_air_distance',
        'max_air_distance',
        #'qbr'
    ]

    rushing_stats = [
        'carries',
        'rushing_yards',
        'rushing_tds',
        'rushing_fumbles',
        'rushing_fumbles_lost',
        'rushing_first_downs',
        'rushing_epa',
        'rushing_2pt_conversions',
        'efficiency',
        'percent_attempts_gte_eight_defenders',
        'avg_time_to_los',
        'avg_rush_yards',
        'expected_rush_yards',
        'rush_yards_over_expected',
        'rush_yards_over_expected_per_att',
        'rush_pct_over_expected',
    ]

    receiving_stats = [
        'receptions',
        'targets',
        'receiving_yards',
        'receiving_tds',
        'receiving_fumbles',
        'receiving_fumbles_lost',
        'receiving_air_yards',
        'receiving_yards_after_catch',
        'receiving_first_downs',
        'receiving_epa',
        'receiving_2pt_conversions',
        'racr',
        'target_share',
        'air_yards_share',
        'wopr',
        'avg_cushion',
        'avg_separation',
        'avg_intended_air_yards_receiving',
        'percent_share_of_intended_air_yards',
        'catch_percentage',
        'avg_yac',
        'avg_expected_yac',
        'avg_yac_above_expectation',
    ]

    general_stats = [
        'special_teams_tds',
        'fantasy_points',
        'fantasy_points_ppr',
    ]

    player_stats_df = collect_weekly_espn_player_stats(year, season_type=season_type)
    player_stats_df = player_stats_df[player_stats_df.season==year].copy()
    ngs_passing_df = collect_weekly_ngs_passing_data([year], season_type=season_type)
    ngs_rushing_df = collect_weekly_ngs_rushing_data([year], season_type=season_type)
    ngs_receiving_df = collect_weekly_ngs_receiving_data([year], season_type=season_type)

    df = pd.merge(player_stats_df, ngs_passing_df, on=['player_id', 'season', 'week', 'recent_team'], how='left')
    df = pd.merge(df, ngs_rushing_df, on=['player_id', 'season', 'week', 'recent_team'], how='left')
    df = pd.merge(df, ngs_receiving_df, on=['player_id', 'season', 'week', 'recent_team'], how='left')

    if mode in ['team', 'opponent']:
        df = df.groupby(['season', 'week', 'recent_team' if mode == 'team' else 'opponent_team'])[passing_stats + rushing_stats + receiving_stats + general_stats].sum().reset_index()
        df['pass_to_rush_ratio'] = df['attempts'] / df['carries']
        df['pass_to_rush_first_down_ratio'] = df['passing_first_downs'] / df['rushing_first_downs']
        df = df.rename(columns={'recent_team' if mode == 'team' else 'opponent_team': 'team'})

    df['net_passing_yards'] = df['passing_yards'] - df['sack_yards']
    df['total_plays'] = df['attempts'] + df['sacks'] + df['carries']
    df['total_yards'] = df['rushing_yards'] + df['passing_yards']
    df['total_fumbles'] = df['rushing_fumbles'] + df['receiving_fumbles'] + df['sack_fumbles']
    df['total_fumbles_lost'] = df['rushing_fumbles_lost'] + df['receiving_fumbles_lost'] + df['sack_fumbles_lost']
    df['total_turnovers'] = df['total_fumbles_lost'] + df['interceptions']
    df['total_touchdowns'] = df['passing_tds'] + df['rushing_tds'] + df['special_teams_tds']
    df['total_first_downs'] = df['passing_first_downs'] + df['rushing_first_downs']
    df['yards_per_pass_attempt'] = df['passing_yards'] / df['attempts']
    df['completion_percentage'] = df['completions'] / df['attempts']
    df['sack_rate'] = df['sacks'] / df['attempts']
    df['fantasy_points_half_ppr'] = df['fantasy_points'] + df['receptions'] * 0.5

    df['yards_per_rush_attempt'] = df['rushing_yards'] / df['carries']
    df['touchdown_per_play'] = df['total_touchdowns'] / df['total_plays']
    df['yards_per_play'] = df['total_yards'] / df['total_plays']
    df['fantasy_point_per_play'] = df['fantasy_points'] / df['total_plays']

    df['air_yards_per_pass_attempt'] = df['receiving_air_yards'] / df['attempts']
    df['VALUE_ELO'] = _calculate_raw_passer_value(df)
    df['passer_rating'] = _calculate_passer_rating(df)
    return df[TEAM_STATS] if mode in ['team', 'opponent'] else df


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