from src.formatters.general import df_rename_fold


def make_cover_feature(schedule):
    """
    Calculate the cover feature for both the team (home or away) and whether the game went under.

    Parameters:
        schedule (DataFrame): DataFrame containing the schedule, scores, spread, and total line.

    Returns:
        DataFrame: DataFrame with added columns for rolling average of team covering and under cover.
    """
    # Calculate if the away team covered the spread and if the game went under
    schedule['away_team_covered'] = (schedule['actual_away_score'] + schedule['spread_line'] >= schedule['actual_home_score']).astype(int)
    schedule['home_team_covered'] = (schedule['actual_home_score'] - schedule['spread_line'] >= schedule['actual_away_score']).astype(int)
    schedule['under_covered'] = (schedule['actual_home_score'] + schedule['actual_away_score'] <= schedule['total_line']).astype(int)

    folded_df = schedule.drop(columns=['actual_away_team_win', 'actual_away_spread', 'actual_point_total','actual_away_team_covered_spread','actual_under_covered'])
    folded_df['ishome'] = folded_df['home_team']
    # Fold the DataFrame to treat home and away teams equally
    folded_df = df_rename_fold(folded_df, 'home_', 'away_')

    # Sort by team, season, and week
    folded_df = folded_df.sort_values(by=['team', 'season', 'week']).reset_index(drop=True)
    folded_df = folded_df.drop_duplicates(['season', 'week', 'team'])
    folded_df['ishome'] = folded_df['ishome'] == folded_df['team']

    # Calculate the rolling average of the last 10 games for covering the spread
    folded_df['rolling_team_cover'] = folded_df.groupby('team')['team_covered'].shift(1).rolling(10, min_periods=1).mean().reset_index(drop=True)

    # Calculate the rolling average of the last 10 games for going under the total
    folded_df['rolling_under_cover'] = folded_df.groupby('team')['under_covered'].shift(1).rolling(10, min_periods=1).mean().reset_index(drop=True)
    home_a = folded_df[folded_df.ishome == True][['season', 'week', 'team', 'rolling_team_cover', 'rolling_under_cover']].rename(columns={'team': 'home_team', 'rolling_team_cover': 'home_rolling_spread_cover', 'rolling_under_cover': 'home_rolling_under_cover'})
    away_a = folded_df[folded_df.ishome == False][['season', 'week', 'team', 'rolling_team_cover', 'rolling_under_cover']].rename(columns={'team': 'away_team', 'rolling_team_cover': 'away_rolling_spread_cover', 'rolling_under_cover': 'away_rolling_under_cover'})
    return away_a, home_a