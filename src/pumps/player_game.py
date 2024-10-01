import datetime

import numpy as np
import pandas as pd

###########################################################
## Loaders
###########################################################
from src.extracts import load_mult_lats, get_play_by_play, load_players
from src.utils import get_seasons_to_update

## From: https://github.com/nflverse/nflfastR/blob/master/R/aggregate_game_stats.R
## Converted from R to Python and additional stats needed for modeling from play by play data
### - redzone stats
### - passing location and distance stats
### - rushing gap location stats
### - receiving location and distance stats
### - success point stats

EXPERIMENT_SCORES = {}

def decode_gsis(new_id):
    if pd.isna(new_id) or len(new_id) != 36:
        return new_id
    else:
        # Extract and convert to GSIS ID
        to_decode = new_id[4:-8].replace("-", "")
        hex_bytes = [to_decode[i:i+2] for i in range(0, len(to_decode), 2)]
        decoded = ''.join([chr(int(byte, 16)) for byte in hex_bytes])
        return decoded

def custom_mode(x, na_rm=True):
    # Step 1: Remove NaN values if na_rm is True
    if na_rm:
        x = [i for i in x if pd.notna(i)]

    # Step 2: Find unique values
    unique_values = np.unique(x)

    # Step 3: Find the most frequent value
    if len(unique_values) > 0:
        mode_value = unique_values[np.argmax([x.count(val) for val in unique_values])]
        return mode_value
    else:
        return None  # Return None if there are no valid values


def decode_player_ids(data):
    # Load player information from a CSV or some data source
    players = pd.read_csv("https://github.com/nflverse/nflverse-data/releases/download/players/player_info.csv")

    # Create a dictionary of GSIS IDs to ESB IDs
    id_vector = dict(zip(players['esb_id'], players['gsis_id']))

    # Apply decoding to all relevant columns
    player_id_columns = [col for col in data.columns if col.endswith('player_id') or col in ['passer_id', 'rusher_id', 'receiver_id', 'id', 'fantasy_id']]

    for col in player_id_columns:
        data[col] = data[col].apply(lambda x: decode_gsis(x) if pd.notna(x) else None)

    return data


###########################################################
## Preprocessing
###########################################################
def filter_normal_plays(pbp):
    # Step 1: Filter for normal plays
    return pbp[
        (~pbp['down'].isna()) &
        (pbp['play_type'].isin(['pass', 'qb_kneel', 'qb_spike', 'run']))
        ].copy()


def filter_two_point_conversions(pbp):
    # Step 1: Filter rows where 'two_point_conv_result' equals 'success'
    return pbp[pbp['two_point_conv_result'] == 'success'][[
        'week', 'season', 'posteam', 'defteam',
        'pass_attempt', 'rush_attempt',
        'passer_player_name', 'passer_player_id',
        'rusher_player_name', 'rusher_player_id',
        'lateral_rusher_player_name', 'lateral_rusher_player_id',
        'receiver_player_name', 'receiver_player_id',
        'lateral_receiver_player_name', 'lateral_receiver_player_id'
    ]].copy()


def filter_passing_stats(data):
    pass_df = data[((data['play_type'].isin(['pass', 'qb_spike'])))].groupby(['passer_player_id', 'week', 'season']).agg(
        passing_yards_after_catch=pd.NamedAgg(column='passing_yards', aggfunc=lambda x: np.sum((x - data.loc[x.index, 'air_yards']) * data.loc[x.index, 'complete_pass'])),
        name_pass=pd.NamedAgg(column='passer_player_name', aggfunc='first'),
        team_pass=pd.NamedAgg(column='posteam', aggfunc='first'),
        opp_pass=pd.NamedAgg(column='defteam', aggfunc='first'),
        passing_yards=pd.NamedAgg(column='passing_yards', aggfunc='sum'),
        passing_tds=pd.NamedAgg(column='touchdown', aggfunc=lambda x: np.sum((x == 1) & (data.loc[x.index, 'td_team'] == data.loc[x.index, 'posteam']) & (data.loc[x.index, 'complete_pass'] == 1))),
        interceptions=pd.NamedAgg(column='interception', aggfunc='sum'),
        attempts=pd.NamedAgg(column='complete_pass', aggfunc=lambda x: np.sum((x == 1) | (data.loc[x.index, 'incomplete_pass'] == 1) | (data.loc[x.index, 'interception'] == 1))),
        completions=pd.NamedAgg(column='complete_pass', aggfunc=lambda x: np.sum(x == 1)),
        sack_fumbles=pd.NamedAgg(column='fumble', aggfunc=lambda x: np.sum((x == 1) & (data.loc[x.index, 'fumbled_1_player_id'] == data.loc[x.index, 'passer_player_id']))),
        sack_fumbles_lost=pd.NamedAgg(column='fumble_lost', aggfunc=lambda x: np.sum((x == 1) & (data.loc[x.index, 'fumbled_1_player_id'] == data.loc[x.index, 'passer_player_id']) & (data.loc[x.index, 'fumble_recovery_1_team'] != data.loc[x.index, 'posteam']))),
        passing_air_yards=pd.NamedAgg(column='air_yards', aggfunc='sum'),
        sacks=pd.NamedAgg(column='sack', aggfunc='sum'),
        sack_yards=pd.NamedAgg(column='yards_gained', aggfunc=lambda x: -np.sum(x * data.loc[x.index, 'sack'])),
        passing_first_downs=pd.NamedAgg(column='first_down_pass', aggfunc='sum'),
        passing_epa=pd.NamedAgg(column='qb_epa', aggfunc='sum')
    ).reset_index()

    # Calculate PACR, handling cases where passing_air_yards might be 0 or NaN
    pass_df['pacr'] = pass_df.apply(
        lambda row: None if pd.isna(row['passing_air_yards']) or row['passing_air_yards'] <= 0
        else row['passing_yards'] / row['passing_air_yards'],
        axis=1
    )

    # Rename columns
    pass_df = pass_df.rename(columns={'passer_player_id': 'player_id'})
    return pass_df


def filter_pass_two_point_conversions(two_points):
    # Step 1: Filter rows where 'pass_attempt' equals 1
    pass_two_points = two_points[two_points['pass_attempt'] == 1]

    # Step 2: Group by 'passer_player_id', 'week', and 'season'
    pass_two_points = pass_two_points.groupby(['passer_player_id', 'week', 'season']).agg(
        name_pass=('passer_player_name', custom_mode),  # Apply custom mode function
        team_pass=('posteam', custom_mode),
        opp_pass=('defteam', custom_mode),
        passing_2pt_conversions=('pass_attempt', 'count')  # Count the number of pass attempts
    ).reset_index()

    # Step 3: Rename 'passer_player_id' to 'player_id'
    pass_two_points = pass_two_points.rename(columns={'passer_player_id': 'player_id'})

    return pass_two_points


def process_pass_df(pass_df, pass_two_points):
    # Step 1: Perform a full join (outer merge)
    pass_df = pd.merge(
        pass_df, pass_two_points,
        on=["player_id", "week", "season", "name_pass", "team_pass", "opp_pass"],
        how="outer"
    )

    # Step 2: Replace NaN values in 'passing_2pt_conversions' with 0
    pass_df['passing_2pt_conversions'] = pass_df['passing_2pt_conversions'].fillna(0).astype(int)

    # Step 3: Filter out rows where 'player_id' is NaN
    pass_df = pass_df[~pass_df['player_id'].isna()]

    # Step 4: Handle missing values for specific columns ("passing_epa", "dakota", "pacr")
    columns_to_handle = ["passing_epa", "dakota", "pacr"]

    # Step 5: Set NaN values in these columns to 0
    pass_df[columns_to_handle] = pass_df[columns_to_handle].fillna(0)

    return pass_df


def filter_rush_stats(data):
    rush_df = (
        data[((data['play_type'].isin(['run', 'qb_kneel'])))]
            .groupby(['rusher_player_id', 'week', 'season'])
            .agg(
            name_rush=pd.NamedAgg(column='rusher_player_name', aggfunc='first'),
            team_rush=pd.NamedAgg(column='posteam', aggfunc='first'),
            opp_rush=pd.NamedAgg(column='defteam', aggfunc='first'),
            yards=pd.NamedAgg(column='rushing_yards', aggfunc='sum'),
            tds=pd.NamedAgg(column='td_player_id', aggfunc=lambda x: np.sum(x == data.loc[x.index, 'rusher_player_id'])),
            carries=pd.NamedAgg(column='rusher_player_id', aggfunc='count'),
            rushing_fumbles=pd.NamedAgg(column='fumble', aggfunc=lambda x: np.sum((x == 1) & (data.loc[x.index, 'fumbled_1_player_id'] == data.loc[x.index, 'rusher_player_id']) & pd.isna(data.loc[x.index, 'lateral_rusher_player_id']))),
            rushing_fumbles_lost=pd.NamedAgg(column='fumble_lost', aggfunc=lambda x: np.sum(
                (x == 1) & (data.loc[x.index, 'fumbled_1_player_id'] == data.loc[x.index, 'rusher_player_id']) & pd.isna(data.loc[x.index, 'lateral_rusher_player_id']) & (data.loc[x.index, 'fumble_recovery_1_team'] != data.loc[x.index, 'posteam']))),
            rushing_first_downs=pd.NamedAgg(column='first_down_rush', aggfunc=lambda x: np.sum((x == 1) & pd.isna(data.loc[x.index, 'lateral_rusher_player_id']))),
            rushing_epa=pd.NamedAgg(column='epa', aggfunc='sum')
        ).reset_index()
    )

    return rush_df


def filter_rush_lateral_stats(data, mult_lats):
    # Filter and group the lateral rushes data
    laterals = data[~data['lateral_rusher_player_id'].isna()].groupby(['lateral_rusher_player_id', 'week', 'season']).agg(
        lateral_yards=pd.NamedAgg(column='lateral_rushing_yards', aggfunc='sum'),
        lateral_fds=pd.NamedAgg(column='first_down_rush', aggfunc='sum'),
        lateral_tds=pd.NamedAgg(column='td_player_id', aggfunc=lambda x: np.sum(x == data.loc[x.index, 'lateral_rusher_player_id'])),
        lateral_att=pd.NamedAgg(column='lateral_rusher_player_id', aggfunc='count'),
        lateral_fumbles=pd.NamedAgg(column='fumble', aggfunc='sum'),
        lateral_fumbles_lost=pd.NamedAgg(column='fumble_lost', aggfunc='sum')
    ).reset_index()

    # Rename column to match rusher_player_id
    laterals = laterals.rename(columns={'lateral_rusher_player_id': 'rusher_player_id'})

    # Bind rows from `mult_lats`
    additional_laterals = mult_lats[
                              (mult_lats['type'] == 'lateral_rushing') &
                              (mult_lats['season'].isin(data['season'])) &
                              (mult_lats['week'].isin(data['week']))
                              ].loc[:, ['season', 'week', 'gsis_player_id', 'yards']].rename(
        columns={'gsis_player_id': 'rusher_player_id', 'yards': 'lateral_yards'}
    )

    # Add columns to match the original lateral structure
    additional_laterals['lateral_tds'] = 0
    additional_laterals['lateral_att'] = 1

    # Bind the additional laterals to the original laterals DataFrame
    laterals = pd.concat([laterals, additional_laterals], ignore_index=True)

    # Group again to summarize, ensuring one row per player per game
    laterals = laterals.groupby(['rusher_player_id', 'week', 'season']).sum(min_count=1).reset_index()
    return laterals


def filter_rush_two_point_conversions(two_points):
    # Step 1: Filter rows where 'rush_attempt' equals 1
    rush_two_points = two_points[two_points['rush_attempt'] == 1]

    # Step 2: Group by 'rusher_player_id', 'week', and 'season'
    rush_two_points = rush_two_points.groupby(['rusher_player_id', 'week', 'season']).agg(
        name_rush=('rusher_player_name', custom_mode),  # Apply custom mode function
        team_rush=('posteam', custom_mode),
        opp_rush=('defteam', custom_mode),
        rushing_2pt_conversions=('rush_attempt', 'count')  # Count the number of rush attempts
    ).reset_index()

    # Step 3: Rename 'rusher_player_id' to 'player_id'
    rush_two_points = rush_two_points.rename(columns={'rusher_player_id': 'player_id'})

    return rush_two_points


def process_rush_df(rushes, laterals, rush_two_points):
    rush_df = pd.merge(rushes, laterals, on=['rusher_player_id', 'week', 'season'], how='left')

    # Replace NaN values with defaults for lateral columns
    rush_df['lateral_yards'] = rush_df['lateral_yards'].fillna(0)
    rush_df['lateral_tds'] = rush_df['lateral_tds'].fillna(0).astype(int)
    rush_df['lateral_fumbles'] = rush_df['lateral_fumbles'].fillna(0)
    rush_df['lateral_fumbles_lost'] = rush_df['lateral_fumbles_lost'].fillna(0)
    rush_df['lateral_fds'] = rush_df['lateral_fds'].fillna(0)

    # Add the new columns by combining rushing and lateral stats
    rush_df['rushing_yards'] = rush_df['yards'] + rush_df['lateral_yards']
    rush_df['rushing_tds'] = rush_df['tds'] + rush_df['lateral_tds']
    rush_df['rushing_first_downs'] = rush_df['rushing_first_downs'] + rush_df['lateral_fds']
    rush_df['rushing_fumbles'] = rush_df['rushing_fumbles'] + rush_df['lateral_fumbles']
    rush_df['rushing_fumbles_lost'] = rush_df['rushing_fumbles_lost'] + rush_df['lateral_fumbles_lost']

    # Rename column and select required columns
    rush_df = rush_df.rename(columns={'rusher_player_id': 'player_id'}).loc[
              :, ['player_id', 'week', 'season', 'name_rush', 'team_rush', 'opp_rush',
                  'rushing_yards', 'carries', 'rushing_tds', 'rushing_fumbles',
                  'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa']
              ]
    # Full join (outer merge) with `rush_two_points`
    rush_df = pd.merge(rush_df, rush_two_points, how='outer', on=['player_id', 'week', 'season', 'name_rush', 'team_rush', 'opp_rush'])

    # Replace NaN values in `rushing_2pt_conversions` with 0
    rush_df['rushing_2pt_conversions'] = rush_df['rushing_2pt_conversions'].fillna(0).astype(int)

    # Filter rows where `player_id` is not NaN
    rush_df = rush_df[~rush_df['player_id'].isna()]

    # Identify the columns that contain NaN values
    rush_df_nas = rush_df.isna()

    # Find the index of the "rushing_epa" column
    epa_index = rush_df.columns.get_loc('rushing_epa')

    # Set the values in the "rushing_epa" column to `False` in the NaN tracker
    rush_df_nas.iloc[:, epa_index] = False

    # Replace NaN values with 0 for all columns except "rushing_epa"
    rush_df[rush_df_nas] = 0
    return rush_df


def filter_receiver_stats(data):
    rec_df = (
        data[data['receiver_player_id'].notna()]
            .groupby(['receiver_player_id', 'week', 'season'])
            .agg(
            name_receiver=pd.NamedAgg(column='receiver_player_name', aggfunc='first'),
            team_receiver=pd.NamedAgg(column='posteam', aggfunc='first'),
            opp_receiver=pd.NamedAgg(column='defteam', aggfunc='first'),
            yards=pd.NamedAgg(column='receiving_yards', aggfunc='sum'),
            receptions=pd.NamedAgg(column='complete_pass', aggfunc=lambda x: np.sum(x == 1)),
            targets=pd.NamedAgg(column='receiver_player_id', aggfunc='count'),
            tds=pd.NamedAgg(column='td_player_id', aggfunc=lambda x: np.sum(x == data.loc[x.index, 'receiver_player_id'])),
            receiving_fumbles=pd.NamedAgg(column='fumble', aggfunc=lambda x: np.sum((x == 1) & (data.loc[x.index, 'fumbled_1_player_id'] == data.loc[x.index, 'receiver_player_id']) & data.loc[x.index, 'lateral_receiver_player_id'].isna())),
            receiving_fumbles_lost=pd.NamedAgg(column='fumble_lost', aggfunc=lambda x: np.sum(
                (x == 1) & (data.loc[x.index, 'fumbled_1_player_id'] == data.loc[x.index, 'receiver_player_id']) & data.loc[x.index, 'lateral_receiver_player_id'].isna() & (data.loc[x.index, 'fumble_recovery_1_team'] != data.loc[x.index, 'posteam']))),
            receiving_air_yards=pd.NamedAgg(column='air_yards', aggfunc='sum'),
            receiving_yards_after_catch=pd.NamedAgg(column='yards_after_catch', aggfunc='sum'),
            receiving_first_downs=pd.NamedAgg(column='first_down_pass', aggfunc=lambda x: np.sum((x == 1) & data.loc[x.index, 'lateral_receiver_player_id'].isna())),
            receiving_epa=pd.NamedAgg(column='epa', aggfunc='sum')
        ).reset_index()
    )
    return rec_df


def filter_receiver_lateral_stats(data, mult_lats):
    laterals = data[data['lateral_receiver_player_id'].notna()].groupby(['lateral_receiver_player_id', 'week', 'season']).agg(
        lateral_yards=pd.NamedAgg(column='lateral_receiving_yards', aggfunc='sum'),
        lateral_tds=pd.NamedAgg(column='td_player_id', aggfunc=lambda x: np.sum(x == data.loc[x.index, 'lateral_receiver_player_id'])),
        lateral_att=pd.NamedAgg(column='lateral_receiver_player_id', aggfunc='count'),
        lateral_fds=pd.NamedAgg(column='first_down_pass', aggfunc='sum'),
        lateral_fumbles=pd.NamedAgg(column='fumble', aggfunc='sum'),
        lateral_fumbles_lost=pd.NamedAgg(column='fumble_lost', aggfunc='sum')
    ).reset_index()

    laterals = laterals.rename(columns={'lateral_receiver_player_id': 'receiver_player_id'})

    additional_laterals = mult_lats[
        (mult_lats['type'] == 'lateral_receiving') &
        (mult_lats['season'].isin(data['season'])) &
        (mult_lats['week'].isin(data['week']))
        ][['season', 'week', 'gsis_player_id', 'yards']].rename(
        columns={'gsis_player_id': 'receiver_player_id', 'yards': 'lateral_yards'}
    )

    additional_laterals['lateral_tds'] = 0
    additional_laterals['lateral_att'] = 1

    laterals = pd.concat([laterals, additional_laterals], ignore_index=True)
    laterals = laterals.groupby(['receiver_player_id', 'week', 'season']).sum(min_count=1).reset_index()

    return laterals


def filter_receiver_two_point_conversions(two_points):
    rec_two_points = two_points[two_points['pass_attempt'] == 1].groupby(['receiver_player_id', 'week', 'season']).agg(
        name_receiver=('receiver_player_name', custom_mode),
        team_receiver=('posteam', custom_mode),
        opp_receiver=('defteam', custom_mode),
        receiving_2pt_conversions=('pass_attempt', 'count')
    ).reset_index()

    rec_two_points = rec_two_points.rename(columns={'receiver_player_id': 'player_id'})
    return rec_two_points


def filter_team_receiving_stats(data):
    rec_team = data[data['receiver_player_id'].notna()].groupby(['posteam', 'week', 'season']).agg(
        team_targets=pd.NamedAgg(column='receiver_player_id', aggfunc='count'),
        team_air_yards=pd.NamedAgg(column='air_yards', aggfunc='sum')
    ).reset_index()
    return rec_team


def process_receiver_df(rec, laterals, rec_team, rec_two_points, racr_ids):
    rec_df = pd.merge(rec, laterals, on=['receiver_player_id', 'week', 'season'], how='left')
    rec_df = pd.merge(rec_df, rec_team, left_on=['team_receiver', 'week', 'season'], right_on=['posteam', 'week', 'season'], how='left')

    rec_df['lateral_yards'] = rec_df['lateral_yards'].fillna(0)
    rec_df['lateral_tds'] = rec_df['lateral_tds'].fillna(0).astype(int)
    rec_df['lateral_fumbles'] = rec_df['lateral_fumbles'].fillna(0)
    rec_df['lateral_fumbles_lost'] = rec_df['lateral_fumbles_lost'].fillna(0)
    rec_df['lateral_fds'] = rec_df['lateral_fds'].fillna(0)

    rec_df['receiving_yards'] = rec_df['yards'] + rec_df['lateral_yards']
    rec_df['receiving_tds'] = rec_df['tds'] + rec_df['lateral_tds']
    rec_df['receiving_yards_after_catch'] = rec_df['receiving_yards_after_catch'] + rec_df['lateral_yards']
    rec_df['receiving_first_downs'] = rec_df['receiving_first_downs'] + rec_df['lateral_fds']
    rec_df['receiving_fumbles'] = rec_df['receiving_fumbles'] + rec_df['lateral_fumbles']
    rec_df['receiving_fumbles_lost'] = rec_df['receiving_fumbles_lost'] + rec_df['lateral_fumbles_lost']

    # Initialize RACR column with NaN values
    rec_df['racr'] = np.nan

    # Calculate RACR safely, avoiding division by zero
    # Only perform division where receiving_air_yards is not zero
    rec_df['racr'] = rec_df.apply(
        lambda row: row['receiving_yards'] / row['receiving_air_yards'] if row['receiving_air_yards'] != 0 else 0,
        axis=1
    )

    # Apply additional conditions from the R code logic
    # Set RACR to 0 for specific players when receiving_air_yards is negative
    rec_df['racr'] = np.where(
        (rec_df['receiving_air_yards'] < 0) & (~rec_df['receiver_player_id'].isin(racr_ids)),  # Apply the player condition
        0,
        rec_df['racr']  # Keep RACR as calculated for other cases
    )

    # Replace NaN values in RACR with None
    rec_df['racr'] = rec_df['racr'].replace({np.nan: None})

    rec_df['target_share'] = rec_df['targets'] / rec_df['team_targets']
    rec_df['air_yards_share'] = rec_df.apply(
        lambda row: row['receiving_air_yards'] / row['team_air_yards'] if row['team_air_yards'] != 0 else 0,
        axis=1
    )
    rec_df['wopr'] = rec_df.apply(
        lambda row: 1.5 * row['target_share'] + 0.7 * row['air_yards_share'] if row['air_yards_share'] != 0 else 0,
        axis=1
    )
    #rec_df['air_yards_share'] = rec_df['receiving_air_yards'] / rec_df['team_air_yards']
    #rec_df['wopr'] = 1.5 * rec_df['target_share'] + 0.7 * rec_df['air_yards_share']

    rec_df = rec_df.rename(columns={'receiver_player_id': 'player_id'})[[
        'player_id', 'week', 'season', 'name_receiver', 'team_receiver', 'opp_receiver',
        'receiving_yards', 'receiving_air_yards', 'receiving_yards_after_catch',
        'receptions', 'targets', 'receiving_tds', 'receiving_fumbles',
        'receiving_fumbles_lost', 'receiving_first_downs', 'receiving_epa',
        'racr', 'target_share', 'air_yards_share', 'wopr'
    ]]

    rec_df = pd.merge(rec_df, rec_two_points, on=['player_id', 'week', 'season', 'name_receiver', 'team_receiver', 'opp_receiver'], how='outer')
    rec_df['receiving_2pt_conversions'] = rec_df['receiving_2pt_conversions'].fillna(0).astype(int)

    rec_df = rec_df[rec_df['player_id'].notna() & rec_df['name_receiver'].notna()]

    rec_df_nas = rec_df.isna()
    epa_index = rec_df.columns.get_loc('receiving_epa')
    rec_df_nas.iloc[:, epa_index] = False

    rec_df[rec_df_nas] = 0
    return rec_df


def combine_all_stats(pass_df, rush_df, rec_df, st_tds, s_type):
    # Full joins for combining the dataframes
    player_df = pd.merge(pass_df, rush_df, on=['player_id', 'week', 'season'], how='outer')
    player_df = pd.merge(player_df, rec_df, on=['player_id', 'week', 'season'], how='outer')
    player_df = pd.merge(player_df, st_tds, on=['player_id', 'week', 'season'], how='outer')
    player_df = pd.merge(player_df, s_type, on=['season', 'week'], how='left')

    # Mutate step to create player_name, recent_team, and opponent_team based on conditions
    player_df['player_name'] = np.where(
        player_df['name_pass'].notna(), player_df['name_pass'],
        np.where(player_df['name_rush'].notna(), player_df['name_rush'],
                 np.where(player_df['name_receiver'].notna(), player_df['name_receiver'], player_df['name_st']))
    )

    player_df['recent_team'] = np.where(
        player_df['team_pass'].notna(), player_df['team_pass'],
        np.where(player_df['team_rush'].notna(), player_df['team_rush'],
                 np.where(player_df['team_receiver'].notna(), player_df['team_receiver'], player_df['team_st']))
    )

    player_df['opponent_team'] = np.where(
        player_df['opp_pass'].notna(), player_df['opp_pass'],
        np.where(player_df['opp_rush'].notna(), player_df['opp_rush'],
                 np.where(player_df['opp_receiver'].notna(), player_df['opp_receiver'], player_df['opp_st']))
    )

    # Select the columns
    columns_to_select = [
        # ID information
        'player_id', 'player_name', 'recent_team', 'season', 'week', 'season_type', 'opponent_team',

        # Passing stats
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
        'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards', 'passing_yards_after_catch',
        'passing_first_downs', 'passing_epa', 'passing_2pt_conversions', 'pacr', 'dakota',

        # Rushing stats
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
        'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions',

        # Receiving stats
        'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
        'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch',
        'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions', 'racr',
        'target_share', 'air_yards_share', 'wopr',

        # Special teams
        'special_teams_tds'
    ]

    player_df = player_df[columns_to_select]

    # Filter out rows where player_id or player_name is missing
    player_df = player_df[player_df['player_id'].notna() & player_df['player_name'].notna()]

    # Handle NA values in specific columns
    player_df_nas = player_df.isna()

    epa_columns = ['passing_epa', 'rushing_epa', 'receiving_epa', 'dakota', 'racr', 'target_share', 'air_yards_share', 'wopr', 'pacr']
    epa_indices = [player_df.columns.get_loc(col) for col in epa_columns if col in player_df.columns]

    # Set False for those epa columns
    player_df_nas.iloc[:, epa_indices] = False

    # Replace remaining NA values with 0
    player_df[player_df_nas] = 0

    # Calculate fantasy points and fantasy points with PPR
    player_df['fantasy_points'] = (
            (1 / 25 * player_df['passing_yards']) +
            (4 * player_df['passing_tds']) +
            (-2 * player_df['interceptions']) +
            (1 / 10 * (player_df['rushing_yards'] + player_df['receiving_yards'])) +
            (6 * (player_df['rushing_tds'] + player_df['receiving_tds'] + player_df['special_teams_tds'])) +
            (2 * (player_df['passing_2pt_conversions'] + player_df['rushing_2pt_conversions'] + player_df['receiving_2pt_conversions'])) +
            (-2 * (player_df['sack_fumbles_lost'] + player_df['rushing_fumbles_lost'] + player_df['receiving_fumbles_lost']))
    )

    player_df['fantasy_points_ppr'] = player_df['fantasy_points'] + player_df['receptions']

    # Sort the dataframe
    player_df = player_df.sort_values(by=['player_id', 'season', 'week'])

    return player_df

def calculate_player_stats(pbp, weekly=False):
    mult_lats = load_mult_lats()
    data = filter_normal_plays(pbp)
    two_points = filter_two_point_conversions(pbp)

    # # we need this column for the special teams tds
    if 'special' not in pbp.columns:
        pbp['special'] = pbp['play_type'].apply(
            lambda x: 1 if x in ["extra_point", "field_goal", "kickoff", "punt"] else 0
        )
    # Select distinct rows based on 'season', 'season_type', and 'week'
    s_type = pbp[['season', 'season_type', 'week']].drop_duplicates()

    #Load the player data
    player_info = load_players()
    #Select specific columns and rename them
    player_info = player_info[[
        'gsis_id', 'display_name', 'short_name', 'position', 'position_group', 'headshot'
    ]].rename(columns={
        'gsis_id': 'player_id',
        'display_name': 'player_display_name',
        'short_name': 'player_name',
        'headshot': 'headshot_url'
    })
    # Filter players for specific positions (RB, FB, HB)
    racr_ids = player_info[player_info['position'].isin(['RB', 'FB', 'HB'])][['player_id']]

    # Passing stats -----------------------------------------------------------
    pass_df = filter_passing_stats(data)

    pass_df['dakota'] = 0 ## TODO: Find way to add dakota

    pass_two_points = filter_pass_two_point_conversions(two_points)
    pass_df = process_pass_df(pass_df, pass_two_points)

    # Rushing stats -----------------------------------------------------------
    rush_df = filter_rush_stats(data)
    rush_lateral_df = filter_rush_lateral_stats(data, mult_lats)
    rush_two_points = filter_rush_two_point_conversions(two_points)
    rush_df = process_rush_df(rush_df, rush_lateral_df, rush_two_points)

    # Receiving stats ---------------------------------------------------------
    receiving_df = filter_receiver_stats(data)
    receiving_lateral_df = filter_receiver_lateral_stats(data, mult_lats)
    receiving_two_points = filter_receiver_two_point_conversions(two_points)
    rec_team = filter_team_receiving_stats(data)
    receiving_df = process_receiver_df(receiving_df, receiving_lateral_df,rec_team, receiving_two_points, racr_ids)

    # Special Teams -----------------------------------------------------------
    # Filter, group, and summarize the data
    st_tds = (
        pbp[pbp['special'] == 1]  # Filter where special == 1
        .dropna(subset=['td_player_id'])  # Drop rows where td_player_id is NaN
        .groupby(['td_player_id', 'week', 'season'])  # Group by player_id, week, season
        .agg(
            name_st=('td_player_name', custom_mode),
            team_st=('td_team', custom_mode),
            opp_st=('defteam', custom_mode),
            special_teams_tds=('touchdown', 'sum')  # Summarize touchdowns
        )
        .reset_index()  # Reset the index to get a DataFrame
        .rename(columns={"td_player_id": "player_id"})  # Rename the player_id column
    )

    player_df = combine_all_stats(pass_df, rush_df, receiving_df, st_tds, s_type)

    # Handle weekly flag
    if not weekly:
        player_df['tgts'] = player_df['targets']
        player_df['rec_air_yds'] = player_df['receiving_air_yards']

        player_df = player_df.groupby('player_id').agg({
            'player_name': custom_mode,
            'recent_team': 'last',
            'completions': 'sum',
            'attempts': 'sum',
            'passing_yards': 'sum',
            'passing_tds': 'sum',
            'interceptions': 'sum',
            'sacks': 'sum',
            'sack_yards': 'sum',
            'sack_fumbles': 'sum',
            'sack_fumbles_lost': 'sum',
            'passing_air_yards': 'sum',
            'passing_yards_after_catch': 'sum',
            'passing_first_downs': 'sum',
            'passing_epa': lambda x: np.nan if x.isna().all() else np.sum(x),
            'passing_2pt_conversions': 'sum',
            'pacr': lambda x: np.nan if np.isnan(x).all() else np.sum(x),
            'carries': 'sum',
            'rushing_yards': 'sum',
            'rushing_tds': 'sum',
            'rushing_fumbles': 'sum',
            'rushing_fumbles_lost': 'sum',
            'rushing_first_downs': 'sum',
            'rushing_epa': lambda x: np.nan if x.isna().all() else np.sum(x),
            'rushing_2pt_conversions': 'sum',
            'receptions': 'sum',
            'targets': 'sum',
            'receiving_yards': 'sum',
            'receiving_tds': 'sum',
            'receiving_fumbles': 'sum',
            'receiving_fumbles_lost': 'sum',
            'receiving_air_yards': 'sum',
            'receiving_yards_after_catch': 'sum',
            'receiving_first_downs': 'sum',
            'receiving_epa': lambda x: np.nan if x.isna().all() else np.sum(x),
            'receiving_2pt_conversions': 'sum',
            'racr': lambda x: np.nan if np.isnan(x).all() else np.sum(x),
            'target_share': lambda x: np.nan if np.isnan(x).all() else np.sum(player_df['tgts']) / np.sum(player_df['tgts'] / player_df['target_share']),
            'air_yards_share': lambda x: np.nan if np.isnan(x).all() else np.sum(player_df['rec_air_yds']) / np.sum(player_df['rec_air_yds'] / player_df['air_yards_share']),
            'wopr': lambda x: 1.5 * player_df['target_share'] + 0.7 * player_df['air_yards_share'],
            'special_teams_tds': 'sum',
            'fantasy_points': 'sum',
            'fantasy_points_ppr': 'sum'
        }).reset_index()

        player_df['racr'] = np.where(
            player_df['receiving_air_yards'] == 0, 0,
            player_df['receiving_yards'] / player_df['receiving_air_yards']
        )
        player_df['pacr'] = np.where(
            player_df['passing_air_yards'] <= 0, 0,
            player_df['passing_yards'] / player_df['passing_air_yards']
        )

    # Join with player info
    player_df = player_df.drop(columns='player_name')
    player_df = pd.merge(player_df, player_info, on='player_id', how='left')

    return player_df



def make_player_game_feature_store(load_seasons):
    fs = []
    for season in load_seasons:
        pbp = get_play_by_play(season)

        print(f"    Preprocessing player game feature store {datetime.datetime.now()}")

        player_df = calculate_player_stats(pbp=pbp, weekly=True)
        fs.append(player_df)

    return pd.concat(fs, ignore_index=True)
