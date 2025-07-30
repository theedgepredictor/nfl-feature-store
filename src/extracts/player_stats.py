import pandas as pd

from src.consts import POSITION_MAPPER, HIGH_POSITION_MAPPER, ESPN_ID_MAPPER
from src.extracts.games import get_schedules
from src.formatters.general import df_rename_fold
from src.formatters.reformat_team_name import team_id_repl
import numpy as np

def get_player_regular_season_game_fs(season, group='off'):
    try:
        df = pd.read_parquet(f'../../data/feature_store/player/{group}/regular_season_game/{season}.parquet')
        return df
    except:
        return pd.DataFrame()

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

## Add Starters from event-data-pump
def get_starters(season):
    try:
        df = pd.read_parquet(f"https://github.com/theedgepredictor/event-data-pump/raw/main/rosters/football/nfl/{season}.parquet")
        df = df.rename(columns={'player_id': 'espn_id', 'team_abbr': 'team'})
        df = team_id_repl(df)

        # Join to events for season and week
        events_df = df_rename_fold(get_schedules([season], season_type=None), 'away_', 'home_')[
            ['game_id', 'season', 'game_type', 'week', 'team', 'espn']
        ].rename(columns={'espn': 'event_id'})

        #print(set(list(df.team.unique())).symmetric_difference(set(list(events_df.team.unique()))))
        df = events_df.merge(df, how='left', on=['event_id', 'team']).drop(columns=['event_id', 'period', 'active','team_id'])
        df['starter'] = df.starter.astype(bool)
        df['did_not_play'] = df.did_not_play.astype(bool)
        df = df[df.espn_id.notnull()].copy()
        df['espn_id'] = df['espn_id'].astype(int)
        df['espn_id'] = df['espn_id'].astype(str) # to match nflverse
        return df
    except Exception as e:
        print(e)
        return pd.DataFrame()

def collect_depth_chart(season):
    try:
        data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.parquet')
        data = data.rename(columns={'club_code': 'team', 'depth_position': 'depth_chart_position', 'gsis_id': 'player_id'})
        data = team_id_repl(data)
        data = data[data.week.notnull()].copy()
        data.week = data.week.astype(int)
        data = data[[
            'season',
            'team',
            'week',
            'depth_team',
            'player_id',
            'position',
            'depth_chart_position',
        ]]
        data['position_group'] = data.position
        data.position_group = data.position_group.map(POSITION_MAPPER)
        data['depth_team'] = data['depth_team'].astype(int)
        return data
    except:
        return pd.DataFrame()


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

    data['filled_espn_id'] = data.player_id
    data['filled_espn_id'] = data['filled_espn_id'].map(ESPN_ID_MAPPER)
    data['espn_id'] = data['espn_id'].fillna(data['filled_espn_id'])
    data = data.drop(columns=['filled_espn_id'])

    return data


def _fill_pre_2002_roster(year):
    r_data = pd.read_parquet(f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.parquet")
    r_data = r_data[[
        'gsis_id',
        'season',
        'team',
        'depth_chart_position',
        'position',
        'jersey_number',
        'status',
        'years_exp',
        'birth_date',

    ]]
    rosters = []
    for week in range(1, 18 + 4):
        snapshot = r_data.copy()
        snapshot['week'] = week
        snapshot['status_description_abbr'] = snapshot['status']
        rosters.append(snapshot)
    return pd.concat(rosters, axis=0).reset_index(drop=True)

def collect_roster(year):
    try:
        if year < 2002:
            player_nfld_df = _fill_pre_2002_roster(year)
        else:
            player_nfld_df = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.parquet')
    except Exception as e:
        if year < 2024:
            return pd.DataFrame()
        print(f'Cant get latest rosters for {year}...using latest player pull as week 1 data')
        player_nfld_df = collect_players()[['player_id', 'birth_date','position', 'latest_team','status_abbr', 'status','years_of_experience','jersey_number']]
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
        'depth_chart_position',
        'jersey_number',
        'birth_date',
        'status',
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
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].astype(str)
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].str.extract('(\d+)') # Only numeric jersey numbers
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].fillna(-1).astype(int) # Fill with -1 to avoid convert to float
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].astype(str) # Convert to string
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].replace("-1", np.nan) # Convert -1 to NaN

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
    data['high_pos_group'] = data.position_group
    data.high_pos_group = data.high_pos_group.map(HIGH_POSITION_MAPPER)
    data['status'] = 'ACT'
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


def get_player_fantasy_projections(season, mode='weekly', group='OFF'):
    """
    Fetches fantasy projections for players based on position group and timeframe.
    """
    try:
        df = pd.read_parquet(f'https://github.com/theedgepredictor/fantasy-data-pump/raw/main/processed/season/football/nfl/{season}.parquet')
        df = team_id_repl(df)
        p_id = pd.read_csv('https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv')
        p_id = p_id[p_id.espn_id.notnull()][['espn_id', 'gsis_id']]
        p_id.espn_id = p_id.espn_id.astype(int)
        p_id_dict = p_id.set_index('espn_id').to_dict()['gsis_id']

        df['player_id'] = df['player_id'].map(p_id_dict)
        
        weekly_meta = [
            'season', 'week', 'player_id', 'name', 'position', 'team',
            'percent_owned', 'percent_started', 'projected_points'
        ]
        
        season_meta = [
            'season', 'player_id', 'name', 'position', 'team',
            'percent_owned', 'percent_started', 'total_points',
            'projected_total_points', 'avg_points', 'projected_avg_points'
        ]
        
        offensive_cols = [
            'projected_rushing_attempts', 'projected_rushing_yards',
            'projected_rushing_touchdowns', 'projected_rushing2_pt_conversions',
            'projected_rushing40_plus_yard_td', 'projected_rushing50_plus_yard_td',
            'projected_rushing100_to199_yard_game', 'projected_rushing200_plus_yard_game',
            'projected_rushing_yards_per_attempt', 'projected_receiving_yards',
            'projected_receiving_touchdowns', 'projected_receiving2_pt_conversions',
            'projected_receiving40_plus_yard_td', 'projected_receiving50_plus_yard_td',
            'projected_receiving_receptions', 'projected_receiving100_to199_yard_game',
            'projected_receiving200_plus_yard_game', 'projected_receiving_targets',
            'projected_receiving_yards_per_reception', 'projected_2_pt_conversions',
            'projected_fumbles', 'projected_lost_fumbles', 'projected_turnovers',
            'projected_passing_attempts', 'projected_passing_completions',
            'projected_passing_yards', 'projected_passing_touchdowns',
            'projected_passing_interceptions', 'projected_passing_completion_percentage'
        ]
        
        defensive_cols = [
            'projected_defensive_solo_tackles',
            'projected_defensive_total_tackles',
            'projected_defensive_interceptions',
            'projected_defensive_fumbles',
            'projected_defensive_blocked_kicks',
            'projected_defensive_safeties',
            'projected_defensive_sacks',
            'projected_defensive_touchdowns',
            'projected_defensive_forced_fumbles',
            'projected_defensive_passes_defensed',
            'projected_defensive_assisted_tackles',
            'projected_defensive_points_allowed',
            'projected_defensive_yards_allowed',
            'projected_defensive0_points_allowed',
            'projected_defensive1_to6_points_allowed',
            'projected_defensive7_to13_points_allowed',
            'projected_defensive14_to17_points_allowed',
            'projected_defensive18_to21_points_allowed',
            'projected_defensive22_to27_points_allowed',
            'projected_defensive28_to34_points_allowed',
            'projected_defensive35_to45_points_allowed',
            'projected_defensive45_plus_points_allowed',
            'projected_defensive100_to199_yards_allowed',
            'projected_defensive200_to299_yards_allowed',
            'projected_defensive300_to349_yards_allowed',
            'projected_defensive350_to399_yards_allowed',
            'projected_defensive400_to449_yards_allowed',
            'projected_defensive450_to499_yards_allowed',
            'projected_defensive500_to549_yards_allowed',
            'projected_defensive550_plus_yards_allowed'
        ]
        
        special_teams_cols = [
            'projected_made_field_goals', 'projected_attempted_field_goals',
            'projected_missed_field_goals', 'projected_made_extra_points',
            'projected_attempted_extra_points', 'projected_missed_extra_points',
            'projected_kickoff_return_touchdowns', 'projected_kickoff_return_yards',
            'projected_punt_return_touchdowns', 'projected_punt_return_yards',
            'projected_punts_returned', 'projected_made_field_goals_from50_plus',
            'projected_attempted_field_goals_from50_plus',
            'projected_made_field_goals_from40_to49',
            'projected_attempted_field_goals_from40_to49',
            'projected_made_field_goals_from_under40',
            'projected_attempted_field_goals_from_under40'
        ]
        
        if group == 'OFF':
            potential_stat_cols = offensive_cols
            positions = ['QB', 'RB', 'WR', 'TE']
        elif group == 'DEF':
            potential_stat_cols = defensive_cols
            positions = ['D/ST']
        elif group == 'ST':
            potential_stat_cols = special_teams_cols
            positions = ['K']
        else:
            ## All
            potential_stat_cols = offensive_cols + defensive_cols + special_teams_cols
            positions = ['QB', 'RB', 'WR', 'TE', 'D/ST', 'K']

        # Only include stat columns that exist in the DataFrame
        stat_cols = [col for col in potential_stat_cols if col in df.columns]

        meta_cols = season_meta if mode == 'season' else weekly_meta
        all_cols = meta_cols + stat_cols
        df = df[all_cols]
        df = df[df.position.isin(positions)].copy()
        
        if mode == 'season':
            meta_df = df[meta_cols].drop_duplicates(['player_id'])
            stats_df = df[['player_id'] + stat_cols].groupby(['player_id']).sum()
            df = pd.merge(meta_df, stats_df, on=['player_id'])
        return df
    except Exception as e:
        print(f"Error fetching fantasy projections: {e}")
        return pd.DataFrame()
    
    