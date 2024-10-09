import pandas as pd

from src.formatters.general import df_rename_fold


def calculate_ranks(df, group_by_col, rank_cols_methods):
    """
    Calculate ranks for specified columns in a DataFrame grouped by a column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        group_by_col (str): The column name to group by (e.g., 'week').
        rank_cols_methods (dict): A dictionary where keys are column names,
                                  and values are the ranking methods ('max' or 'min').

    Returns:
        pd.DataFrame: The original DataFrame with new rank columns.
    """
    rank_df_list = []

    # Calculate ranks for each column based on its corresponding method
    for col, method in rank_cols_methods.items():
        # Rank for each column based on its specific method
        rank_col_df = df.groupby(group_by_col, as_index=False)[col].rank(method=method, ascending=(method == 'min')).astype(int)
        rank_df_list.append(rank_col_df)  # Keep only the rank column

    # Combine all rank columns
    rank_df = pd.concat(rank_df_list, axis=1)

    # Add suffix to rank columns
    rank_df.columns = [f"{col}_rank" for col in rank_cols_methods.keys()]

    # Concatenate the rank columns back to the original dataframe
    df = pd.concat([df.reset_index(drop=True), rank_df.reset_index(drop=True)], axis=1)

    return df.drop(columns=[col for col in rank_cols_methods.keys()])


# Specify which columns should use 'max' and which should use 'min'


def make_rank_cols(team_fs):
    team_fs['ishome'] = team_fs['home_team']
    team_fs_df = df_rename_fold(team_fs, 'away_', 'home_')
    team_fs_df['ishome'] = team_fs_df['ishome'] == team_fs_df['team']
    df = team_fs[['away_team', 'home_team', 'week', 'season']].copy()
    rank_cols_methods_offense = {
        #'elo_pre': 'max',
        'avg_points_offense': 'max',
        'avg_rushing_yards_offense': 'max',
        'avg_passing_yards_offense': 'max',
        'avg_total_yards_offense': 'max',
        'avg_yards_per_play_offense': 'max',
        'avg_total_turnovers_offense': 'min'  # Use 'min' for turnovers
    }

    rank_cols_methods_defense = {
        'avg_points_defense': 'min',
        'avg_rushing_yards_defense': 'min',
        'avg_passing_yards_defense': 'min',
        'avg_total_yards_defense': 'min',
        'avg_yards_per_play_defense': 'min',
        'avg_total_turnovers_defense': 'max'
    }

    ## Handle null state agg removal (first week is always null in agg we grab multiple seasons back to avoid errors in calculation but need to drop null state)

    team_fs_df = team_fs_df[team_fs_df['avg_points_offense'].notnull()].copy()

    # Calculate offensive ranks
    offensive_ranks = calculate_ranks(team_fs_df[['team', 'week', 'season'] + list(rank_cols_methods_offense.keys())], ['season','week'], rank_cols_methods_offense)
    o_ranks = offensive_ranks.copy()
    o_ranks['offensive'] = o_ranks['avg_points_offense_rank'] + o_ranks['avg_total_yards_offense_rank']
    o_ranks = calculate_ranks(o_ranks[['team', 'week', 'season', 'offensive']], ['season','week'], {'offensive': 'min'})
    defensive_ranks = calculate_ranks(team_fs_df[['team', 'week', 'season'] + list(rank_cols_methods_defense.keys())], ['season','week'], rank_cols_methods_defense)
    d_ranks = defensive_ranks.copy()
    d_ranks['defensive'] = d_ranks['avg_points_defense_rank'] + d_ranks['avg_total_yards_defense_rank']
    d_ranks = calculate_ranks(d_ranks[['team', 'week', 'season', 'defensive']], ['season','week'], {'defensive': 'min'})

    full_rank = pd.merge(offensive_ranks, defensive_ranks, on=['team', 'week', 'season'])
    full_rank = pd.merge(full_rank, o_ranks, on=['team', 'week', 'season'])
    full_rank = pd.merge(full_rank, d_ranks, on=['team', 'week', 'season'])
    full_rank['net_rank'] = (full_rank['offensive_rank'] + full_rank['defensive_rank']) / 2
    full_rank = full_rank.sort_values(by=['season', 'week', 'offensive_rank', ]).reset_index(drop=True)
    away_fs = full_rank.copy()
    away_fs.columns = ['away_' + col if col not in ['week', 'season'] else col for col in away_fs.columns]
    home_fs = full_rank.copy()
    home_fs.columns = ['home_' + col if col not in ['week', 'season'] else col for col in home_fs.columns]
    df = pd.merge(df, away_fs, on=['away_team', 'week', 'season'])
    df = pd.merge(df, home_fs, on=['home_team', 'week', 'season'])
    return df