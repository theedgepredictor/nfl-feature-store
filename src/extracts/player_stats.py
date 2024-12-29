import pandas as pd

from src.consts import POSITION_MAPPER, HIGH_POSITION_MAPPER
from src.formatters.reformat_team_name import team_id_repl
import numpy as np

## Add from nfl-madden-data pump
def get_madden_ratings(season):
    try:
        df = pd.read_csv(f'https://github.com/theedgepredictor/nfl-madden-data/raw/main/data/madden/processed/{season}.csv')
        return df
    except:
        return pd.DataFrame()

## Add from nfl-madden-data pump
def get_approximate_value(season):
    try:
        df = pd.read_csv(f'https://github.com/theedgepredictor/nfl-madden-data/raw/main/data/pfr/approximate_value/{season}.csv')
        return df
    except:
        return pd.DataFrame()

def get_player_regular_season_game_fs(season, group='off'):
    try:
        df = pd.read_parquet(f'../..//data/feature_store/player/{group}/regular_season_game/{season}.parquet')
        return df
    except:
        return pd.DataFrame()

def collect_depth_chart(season):
    data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.parquet')
    data = team_id_repl(data)
    data['position_group'] = data.position
    data.position_group = data.position_group.map(POSITION_MAPPER)
    data['depth_team'] = data['depth_team'].astype(int)
    return data

def collect_injuries(season):
    try:
        data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet')
        data = team_id_repl(data)
        data['position_group'] = data.position
        data.position_group = data.position_group.map(POSITION_MAPPER)
        return data.rename(columns={'gsis_id': 'player_id'})
    except Exception as e:
        return pd.DataFrame()

def collect_combine():
    data = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/combine/combine.parquet")
    data = team_id_repl(data)
    data['position_group'] = data.pos
    data.position_group = data.position_group.map(POSITION_MAPPER)
    return data.rename(columns={'player_name': 'name'})

def collect_players():
    data = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/players_components/players.parquet")
    data = team_id_repl(data)
    data['position_group'] = data.position
    data.position_group = data.position_group.map(POSITION_MAPPER)
    data['high_pos_group'] = data.position_group
    data.high_pos_group = data.high_pos_group.map(HIGH_POSITION_MAPPER)
    data['status_abbr'] = data.status
    data.status_abbr = data.status_abbr.fillna('N')
    data.status_abbr = data.status_abbr.apply(lambda x: x[0])
    data.status_abbr = data.status_abbr.replace(['W', 'E', 'I', 'N'], ['N', 'N', 'N', 'N'])
    data = data.rename(columns={'display_name': 'name', 'gsis_id': 'player_id'})
    def add_missing_draft_data(df):
        ## load missing draft data ##
        missing_draft = pd.read_csv(
            'https://github.com/greerreNFL/nfeloqb/raw/refs/heads/main/nfeloqb/Manual%20Data/missing_draft_data.csv',
        )
        ## groupby id to ensure no dupes ##
        missing_draft = missing_draft.groupby(['player_id']).head(1)
        ## rename the cols, which will fill if main in NA ##
        missing_draft = missing_draft.rename(columns={
            'rookie_year': 'rookie_season_fill',
            'draft_number': 'draft_pick_fill',
            'entry_year': 'draft_year_fill',
            'birth_date': 'birth_date_fill',
        })
        ## add to data ##
        df = pd.merge(
            df,
            missing_draft[[
                'player_id', 'rookie_season_fill', 'draft_pick_fill',
                'draft_year_fill', 'birth_date_fill'
            ]],
            on=['player_id'],
            how='left'
        )
        ## fill in missing data ##
        for col in [
            'rookie_season', 'draft_pick', 'draft_year', 'birth_date'
        ]:
            ## fill in missing data ##
            df[col] = df[col].combine_first(df[col + '_fill'])
            ## and then drop fill col ##
            df = df.drop(columns=[col + '_fill'])
        ## return ##
        return df

    data = add_missing_draft_data(data)

    return data

def collect_roster(year):
    try:
        player_nfld_df = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.parquet')
    except Exception as e:
        print(f'Cant get latest rosters for {year}...using latest player pull as week 1 data')
        player_nfld_df = collect_players()[['player_id', 'birth_date','position', 'latest_team','status_abbr', 'years_of_experience','jersey_number']]
        player_nfld_df = player_nfld_df.rename(
            columns={
                'latest_team': 'team',
                'years_of_experience': 'years_exp',
                'status_abbr': 'status_description_abbr',
                'player_id': 'gsis_id'
            })
        player_nfld_df['season'] = year
        player_nfld_df['week'] = 1

    player_nfld_df = team_id_repl(player_nfld_df)
    player_nfld_df = player_nfld_df[[
        'season',
        'week',
        'team',
        'position',
        #'depth_chart_position',
        'jersey_number',
        'birth_date',
        # 'status',
        'status_description_abbr',
        'gsis_id',
        # 'sportradar_id',
        # 'yahoo_id',
        # 'rotowire_id',
        # 'pff_id',
        # 'fantasy_data_id',
        # 'sleeper_id',
        'years_exp',
        # 'headshot_url',
        # 'ngs_position',
        # 'game_type',

        # 'football_name',
        # 'esb_id',
        # 'gsis_it_id',
        # 'smart_id',
    ]].rename(columns={'full_name': 'name', 'gsis_id': 'player_id'})
    player_nfld_df = player_nfld_df.loc[(
            (player_nfld_df.player_id.notnull()) & (player_nfld_df.birth_date.notnull()))].copy()
    player_nfld_df = player_nfld_df.drop(columns=['birth_date'])
    player_nfld_df = player_nfld_df.loc[player_nfld_df.player_id != ''].copy()
    player_nfld_df = player_nfld_df.rename(columns={'status_description_abbr': 'status_abbr'})
    player_nfld_df.status_abbr = player_nfld_df.status_abbr.fillna('N')
    player_nfld_df.status_abbr = player_nfld_df.status_abbr.apply(lambda x: x[0])
    player_nfld_df.status_abbr = player_nfld_df.status_abbr.replace(['W', 'E', 'I', 'N'], ['N', 'N', 'N', 'N'])
    player_nfld_df = player_nfld_df.reset_index().drop(columns='index')
    #player_nfld_df = player_nfld_df[player_nfld_df.week == 1].copy()
    player_nfld_df['position_group'] = player_nfld_df.position
    player_nfld_df.position_group = player_nfld_df.position_group.map(POSITION_MAPPER)
    player_nfld_df['high_pos_group'] = player_nfld_df.position_group
    player_nfld_df.high_pos_group = player_nfld_df.high_pos_group.map(HIGH_POSITION_MAPPER)
    return player_nfld_df

def collect_weekly_espn_player_stats(season, week=None, season_type=None,  group=''):
    if group in ['def', 'kicking']:
        group = '_' + group
    data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats{group}.parquet')
    if week is not None:
        data = data[((data.season < season) | ((data.season == season) & (data.week <= week)))].copy()
    else:
        data = data[data.season <= season].copy()
    if season_type is not None:
        data = data[((data.season_type == season_type))].copy()
    data = team_id_repl(data)
    data['position_group'] = data.position
    data.position_group = data.position_group.map(POSITION_MAPPER)
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
        #'passer_rating',
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
