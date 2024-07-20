import numpy as np
import pandas as pd

from src.consts import NFL_PLAYER_URL, player_cols_raw, NFL_PLAYER_STATS_URL, PLAYER_POINTS_COLUMNS, PLAYER_BOXSCORE_COLUMNS
from src.data_types import fs_apply_type


def etl_player_df():
    player_df = pd.read_parquet(NFL_PLAYER_URL)
    player_df = player_df[player_cols_raw]
    player_df = player_df[player_df.position_group.isin(['QB', 'RB', 'WR', 'TE'])].copy()
    player_df = player_df.rename(columns={'gsis_id': 'player_id'})
    return player_df

def etl_player_info_df():
    player_stats_df = pd.read_parquet(NFL_PLAYER_STATS_URL)
    player_stats_df = player_stats_df[player_stats_df.position_group.isin(['QB', 'RB', 'WR', 'TE'])].copy()
    player_stats_df = player_stats_df[player_stats_df.season_type == 'REG'].drop(columns=['player_name', 'player_display_name', 'headshot_url', 'recent_team', 'season_type', 'opponent_team']).copy()
    player_stats_df['games_played'] = 1
    return player_stats_df


def target_feature_store(player_weekly_stats_df):
    position_group_df = player_weekly_stats_df.drop_duplicates(['player_id'])[['player_id', 'position_group']]
    player_points_df = player_weekly_stats_df[PLAYER_POINTS_COLUMNS].groupby(['player_id', 'season'])['fantasy_points'].sum().reset_index()
    player_points_df['fantasy_points_ppr'] = player_weekly_stats_df[PLAYER_POINTS_COLUMNS].groupby(['player_id', 'season'])['fantasy_points_ppr'].sum().reset_index()['fantasy_points_ppr']
    player_points_df = pd.merge(player_points_df, position_group_df, on=['player_id'], how='left')
    player_points_df['position_rank'] = player_points_df.groupby(['position_group', 'season'])['fantasy_points'].rank(ascending=False, method='first')
    player_points_df['position_rank'] = player_points_df['position_rank'].astype("Int32")
    player_points_df['ppr_position_rank'] = player_points_df.groupby(['position_group', 'season'])['fantasy_points_ppr'].rank(ascending=False, method='first')
    player_points_df['ppr_position_rank'] = player_points_df['ppr_position_rank'].astype("Int32")
    player_points_df = player_points_df.drop(columns=['position_group'])
    return player_points_df


def season_total_feature_store(player_weekly_stats_df):
    # Group by player and season to get total stats
    position_group_df = player_weekly_stats_df.drop_duplicates(['player_id'])[['player_id', 'position_group']]
    total_stats = player_weekly_stats_df[PLAYER_BOXSCORE_COLUMNS + ['games_played']].groupby(['player_id', 'season']).sum().reset_index()
    total_stats = total_stats[(total_stats.games_played > 3)].copy()
    player_info_df = pd.DataFrame()
    for n_last in [0, 1, 2]:
        t_copy = total_stats.copy()
        prefix = 'last_year_' if n_last == 0 else f'{n_last + 1}_years_ago_'
        t_copy['season'] = t_copy['season'] + n_last
        t_copy = pd.merge(t_copy, position_group_df, on=['player_id'], how='left')
        t_copy['position_rank'] = t_copy.groupby(['position_group', 'season'])['fantasy_points'].rank(ascending=False, method='first')
        t_copy['position_rank'] = t_copy['position_rank'].astype("Int32")
        t_copy['ppr_position_rank'] = t_copy.groupby(['position_group', 'season'])['fantasy_points_ppr'].rank(ascending=False, method='first')
        t_copy['ppr_position_rank'] = t_copy['ppr_position_rank'].astype("Int32")
        t_copy = t_copy.drop(columns=['position_group'])
        t_copy.columns = [f'total_{prefix}{col}' if col not in ['player_id', 'season'] else col for col in t_copy.columns]
        join_cols = ['player_id', 'season'] if 'season' in player_info_df.columns else ['player_id']
        if player_info_df.shape[0] != 0:
            player_info_df = pd.merge(player_info_df, t_copy, on=join_cols, how='left')
        else:
            player_info_df = t_copy
    t_copy = total_stats.copy()
    t_copy = t_copy.sort_values(by=['player_id', 'season'])
    t_copy.columns = [f'total_career_{col}' if col not in ['player_id', 'season'] else col for col in t_copy.columns]
    feature_cols = [i for i in t_copy.columns if i not in ['player_id', 'season']]
    t_copy[feature_cols] = t_copy.groupby('player_id')[feature_cols].cumsum()
    player_info_df = pd.merge(player_info_df, t_copy, on=join_cols, how='left')
    del t_copy
    player_info_df['season'] = player_info_df['season'].fillna(-1)
    player_info_df = player_info_df[player_info_df.season != -1].copy()
    player_info_df['season'] = player_info_df['season'] + 1
    return player_info_df.sort_values(by=['player_id', 'season'])


def season_avg_feature_store(player_weekly_stats_df):
    # Compute average stats per season
    avg_stats = player_weekly_stats_df[PLAYER_BOXSCORE_COLUMNS + ['games_played']].groupby(['player_id', 'season']).sum().reset_index()
    avg_stats = avg_stats[(avg_stats.games_played > 3)].copy()
    stat_columns = avg_stats.columns.difference(['player_id', 'season', 'games_played'])
    avg_stats[stat_columns] = avg_stats[stat_columns].div(avg_stats['games_played'], axis=0)
    avg_stats.drop(columns=['games_played'], inplace=True)
    player_info_df = pd.DataFrame()
    # Include last 1, 2, and 3 years averages
    for n_last in [0, 1, 2]:
        t_copy = avg_stats.copy()
        prefix = 'last_year_' if n_last == 0 else f'{n_last + 1}_years_ago_'
        t_copy['season'] = t_copy['season'] + n_last
        t_copy.columns = [f'avg_{prefix}{col}' if col not in ['player_id', 'season'] else col for col in t_copy.columns]
        join_cols = ['player_id', 'season'] if 'season' in player_info_df.columns else ['player_id']
        if player_info_df.shape[0] != 0:
            player_info_df = pd.merge(player_info_df, t_copy, on=join_cols, how='left')
        else:
            player_info_df = t_copy
    # Compute cumulative average stats
    t_copy = avg_stats.copy()
    t_copy = t_copy.sort_values(by=['player_id', 'season'])
    t_copy.columns = [f'avg_career_{col}' if col not in ['player_id', 'season'] else col for col in t_copy.columns]
    feature_cols = [i for i in t_copy.columns if i not in ['player_id', 'season']]
    for col in feature_cols:
        t_copy[col] = t_copy.groupby('player_id')[col].expanding().mean().reset_index(level=0, drop=True)
    # Merge cumulative average stats with player_info_df
    join_cols = ['player_id', 'season']
    player_info_df = pd.merge(player_info_df, t_copy, on=join_cols, how='left')
    # Final adjustments
    player_info_df['season'] = player_info_df['season'].fillna(-1)
    player_info_df = player_info_df[player_info_df.season != -1].copy()
    player_info_df['season'] = player_info_df['season'] + 1
    return player_info_df.sort_values(by=['player_id', 'season'])


def make_season_feature_store():
    player_df = etl_player_df()
    player_weekly_stats_df = etl_player_info_df()
    player_points_df = target_feature_store(player_weekly_stats_df)
    player_season_total_fs_df = season_total_feature_store(player_weekly_stats_df)
    player_season_avg_fs_df = season_avg_feature_store(player_weekly_stats_df)
    fs_df = pd.merge(player_season_total_fs_df, player_season_avg_fs_df, on=['player_id', 'season'], how='left')
    del player_season_total_fs_df, player_season_avg_fs_df
    fs_df = pd.merge(fs_df, player_points_df, on=['player_id', 'season'], how='left')
    fs_df = pd.merge(fs_df, player_df, on=['player_id'], how='left')

    fs_df['years_of_experience'] = fs_df['season'] - fs_df['entry_year']
    fs_df.fillna(-1, inplace=True)

    # rank by position and season for ranked by total fantasy points and make a column called position_rank

    fs_df = fs_df[(fs_df.fantasy_points > 0) | (fs_df.fantasy_points_ppr > 0) | (fs_df.season == fs_df.season.max())].copy()
    fs_df = fs_apply_type(fs_df, method='contains')
    return fs_df