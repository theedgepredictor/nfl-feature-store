import datetime

import numpy as np
import pandas as pd

def get_play_by_play(season):
    try:
        data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet')
        data.fillna(-1000000, inplace=True)
        data.replace(-1000000, None, inplace=True)
        ## Fixes
        data['quarter_seconds_remaining'] = data['quarter_seconds_remaining'].ffill()
        data['game_seconds_remaining'] = data['game_seconds_remaining'].ffill()

        #############################################################################################
        ## Attrs
        #############################################################################################
        data['is_redzone'] = data['yardline_100'] <= 20
        data['is_middle_8'] = ((data['qtr'] == 2) & (data['quarter_seconds_remaining'] <= 60 * 4)) | (data['qtr'] == 3) & (data['quarter_seconds_remaining'] >= (60 * 15) - (60 * 4))
        data['is_third_and_short'] = (data['down'] == 3) & (data['ydstogo'] < 3)
        data['is_third_and_medium'] = (data['down'] == 3) & (data['ydstogo'] >= 3) & (data['ydstogo'] < 7)
        data['is_third_and_long'] = (data['down'] == 3) & (data['ydstogo'] >= 7)
        data['successful_two_point_conversion'] = np.where(
            data['two_point_conv_result'] == 'success', 1,
            np.where(
                data['two_point_conv_result'].isna() & data['desc'].str.contains('ATTEMPT SUCCEEDS'), 1, 0
            )
        )
        data['sack_yards'] = None
        data.loc[data.sack == 1, 'sack_yards'] = data.loc[data.sack == 1, 'yards_gained']
        data = data.drop(columns=['two_point_conv_result'])
        ### Pass attrs
        data['short_left_pass'] = (data['pass_length'] == 'short') & (data['pass_location'] == 'left')
        data['short_middle_pass'] = (data['pass_length'] == 'short') & (data['pass_location'] == 'middle')
        data['short_right_pass'] = (data['pass_length'] == 'short') & (data['pass_location'] == 'right')
        data['deep_left_pass'] = (data['pass_length'] == 'deep') & (data['pass_location'] == 'left')
        data['deep_middle_pass'] = (data['pass_length'] == 'deep') & (data['pass_location'] == 'middle')
        data['deep_right_pass'] = (data['pass_length'] == 'deep') & (data['pass_location'] == 'right')
        data = data.drop(columns=['pass_length', 'pass_location'])

        ### Rush attrs
        data['left_end_rush'] = (data['run_location'] == 'left') & (data['run_gap'] == 'end')
        data['left_guard_rush'] = (data['run_location'] == 'left') & (data['run_gap'] == 'guard')
        data['left_tackle_rush'] = (data['run_location'] == 'left') & (data['run_gap'] == 'tackle')
        data['right_end_rush'] = (data['run_location'] == 'right') & (data['run_gap'] == 'end')
        data['right_guard_rush'] = (data['run_location'] == 'right') & (data['run_gap'] == 'guard')
        data['right_tackle_rush'] = (data['run_location'] == 'right') & (data['run_gap'] == 'tackle')
        data = data.drop(columns=['run_location', 'run_gap'])
        return data
    except:
        return pd.DataFrame()


def get_schedules(seasons):
    if min(seasons) < 1999:
        raise ValueError('Data not available before 1999.')

    scheds = pd.read_csv('http://www.habitatring.com/games.csv')
    scheds = scheds[scheds['season'].isin(seasons)].copy()
    return scheds


def get_elo(season):
    try:
        df = pd.read_parquet(f'https://github.com/theedgepredictor/elo-rating/raw/main/data/elo/football/nfl/{season}.parquet')
        return df
    except:
        return pd.DataFrame()


def collect_weekly_espn_player_stats(season, season_type="REG"):
    data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.parquet')
    data = data[((data.season_type == season_type))].copy()
    return data


def collect_weekly_ngs_passing_data(seasons, season_type="REG"):
    """
    Minimum 15 Pass Attempts in the week to be included
    """
    cols = [
        'season',
        # 'season_type',
        'week',
        # 'player_display_name',
        # 'player_position',
        'team_abbr',
        'avg_time_to_throw',
        'avg_completed_air_yards',
        'avg_intended_air_yards',
        'avg_air_yards_differential',
        'aggressiveness',
        'max_completed_air_distance',
        'avg_air_yards_to_sticks',
        # 'attempts',
        # 'pass_yards',
        # 'pass_touchdowns',
        # 'interceptions',
        'passer_rating',
        # 'completions',
        'completion_percentage',
        'expected_completion_percentage',
        'completion_percentage_above_expectation',
        'avg_air_distance',
        'max_air_distance',
        'player_gsis_id',
        # 'player_first_name',
        # 'player_last_name',
        # 'player_jersey_number',
        # 'player_short_name'
    ]
    ngs_passing_data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_passing.parquet')
    ngs_passing_data = ngs_passing_data[((ngs_passing_data.season.isin(seasons)) & (ngs_passing_data.week > 0) & (ngs_passing_data.season_type == season_type))].copy()
    return ngs_passing_data[cols].rename(columns={'player_gsis_id': 'player_id', 'team_abbr': 'recent_team', 'avg_intended_air_yards': 'avg_intended_air_yards_passing'})


def collect_weekly_ngs_rushing_data(seasons, season_type="REG"):
    """
    Minimum 10 Rush Attempts in the week to be included
    """
    cols = [
        'season',
        # 'season_type',
        'week',
        # 'player_display_name',
        # 'player_position',
        'team_abbr',
        'efficiency',
        'percent_attempts_gte_eight_defenders',
        'avg_time_to_los',
        # 'rush_attempts',
        # 'rush_yards',
        'avg_rush_yards',
        # 'rush_touchdowns',
        'player_gsis_id',
        # 'player_first_name',
        # 'player_last_name',
        # 'player_jersey_number',
        # 'player_short_name',
        'expected_rush_yards',
        'rush_yards_over_expected',
        'rush_yards_over_expected_per_att',
        'rush_pct_over_expected',
    ]
    ngs_rushing_data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_rushing.parquet')
    ngs_rushing_data = ngs_rushing_data[((ngs_rushing_data.season.isin(seasons)) & (ngs_rushing_data.week > 0) & (ngs_rushing_data.season_type == season_type))].copy()
    return ngs_rushing_data[cols].rename(columns={'player_gsis_id': 'player_id', 'team_abbr': 'recent_team'})


def collect_weekly_ngs_receiving_data(seasons, season_type="REG"):
    """
    Minimum 10 Rush Attempts in the week to be included
    """
    cols = [
        'season',
        # 'season_type',
        'week',
        # 'player_display_name',
        # 'player_position',
        'team_abbr',
        "avg_cushion",
        "avg_separation",
        "avg_intended_air_yards",
        "percent_share_of_intended_air_yards",
        # "receptions",
        # "targets",
        "catch_percentage",
        # "yards",
        # "rec_touchdowns",
        "avg_yac",
        "avg_expected_yac",
        "avg_yac_above_expectation",
        'player_gsis_id',
        # 'player_first_name',
        # 'player_last_name',
        # 'player_jersey_number',
        # 'player_short_name',

    ]
    ngs_receiving_data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_receiving.parquet')
    ngs_receiving_data = ngs_receiving_data[((ngs_receiving_data.season.isin(seasons)) & (ngs_receiving_data.week > 0) & (ngs_receiving_data.season_type == season_type))].copy()
    return ngs_receiving_data[cols].rename(columns={'player_gsis_id': 'player_id', 'team_abbr': 'recent_team', 'avg_intended_air_yards': 'avg_intended_air_yards_receiving'})


def collect_qbr(seasons, season_type="REG"):
    season_type = "Regular" if season_type == "REG" else "Playoffs"
    qbr = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_week_level.csv')
    qbr = qbr[qbr.season.isin(seasons)].copy()
    qbr = qbr[qbr.season_type == season_type].copy()
    p_id = pd.read_csv('https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv')
    p_id = p_id[p_id.espn_id.notnull()][['espn_id', 'gsis_id']]
    p_id.espn_id = p_id.espn_id.astype(int)
    p_id_dict = p_id.set_index('espn_id').to_dict()['gsis_id']

    qbr['player_id'] = qbr['player_id'].map(p_id_dict)
    return qbr[['season', 'week_num', 'team_abb', 'player_id', 'qbr_total']].rename(columns={'qbr_total': 'qbr', 'week_num': 'week', 'team_abb': 'recent_team'})


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
        'passer_rating',
        'completion_percentage',
        'expected_completion_percentage',
        'completion_percentage_above_expectation',
        'avg_air_distance',
        'max_air_distance',
        'qbr'
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
    ngs_passing_df = collect_weekly_ngs_passing_data([year], season_type=season_type)
    ngs_rushing_df = collect_weekly_ngs_rushing_data([year], season_type=season_type)
    ngs_receiving_df = collect_weekly_ngs_receiving_data([year], season_type=season_type)
    espn_qbr_df = collect_qbr([year], season_type=season_type)

    df = pd.merge(player_stats_df, ngs_passing_df, on=['player_id', 'season', 'week', 'recent_team'], how='left')
    df = pd.merge(df, ngs_rushing_df, on=['player_id', 'season', 'week', 'recent_team'], how='left')
    df = pd.merge(df, ngs_receiving_df, on=['player_id', 'season', 'week', 'recent_team'], how='left')
    df = pd.merge(df, espn_qbr_df, on=['player_id', 'season', 'week', 'recent_team'], how='left')

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
    # df['completion_percentage'] = df['completions'] / df['attempts']
    df['sack_rate'] = df['sacks'] / df['attempts']
    df['fantasy_points_half_ppr'] = df['fantasy_points'] + df['receptions'] * 0.5

    df['yards_per_rush_attempt'] = df['rushing_yards'] / df['carries']
    df['touchdown_per_play'] = df['total_touchdowns'] / df['total_plays']
    df['yards_per_play'] = df['total_yards'] / df['total_plays']
    df['fantasy_point_per_play'] = df['fantasy_points'] / df['total_plays']

    df['air_yards_per_pass_attempt'] = df['receiving_air_yards'] / df['attempts']
    return df[TEAM_STATS] if mode in ['team', 'opponent'] else df