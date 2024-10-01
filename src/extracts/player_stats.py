import pandas as pd

from src.formatters.reformat_team_name import team_id_repl
import numpy as np


def collect_weekly_espn_player_stats(season, season_type="REG"):
    data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.parquet')
    if season_type is not None:
        data = data[((data.season_type == season_type))].copy()
    data = team_id_repl(data)
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
        #'completion_percentage',
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
    ngs_passing_data = ngs_passing_data[((ngs_passing_data.season.isin(seasons)) & (ngs_passing_data.week > 0))].copy()
    if season_type is not None:
        ngs_passing_data = ngs_passing_data[(ngs_passing_data.season_type == season_type)].copy()

    ngs_passing_data = team_id_repl(ngs_passing_data)
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
    ngs_rushing_data = ngs_rushing_data[((ngs_rushing_data.season.isin(seasons)) & (ngs_rushing_data.week > 0))].copy()
    if season_type is not None:
        ngs_rushing_data = ngs_rushing_data[(ngs_rushing_data.season_type == season_type)].copy()
    ngs_rushing_data = team_id_repl(ngs_rushing_data)
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
    ngs_receiving_data = ngs_receiving_data[((ngs_receiving_data.season.isin(seasons)) & (ngs_receiving_data.week > 0))].copy()
    if season_type is not None:
        ngs_receiving_data = ngs_receiving_data[(ngs_receiving_data.season_type == season_type)].copy()
    ngs_receiving_data = team_id_repl(ngs_receiving_data)
    return ngs_receiving_data[cols].rename(columns={'player_gsis_id': 'player_id', 'team_abbr': 'recent_team', 'avg_intended_air_yards': 'avg_intended_air_yards_receiving'})
