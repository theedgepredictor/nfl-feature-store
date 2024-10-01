from src.transforms.averages import dynamic_window_ewma


def make_rushing_epa(data):
    """
    Calculate the rushing EPA for both offense and defense with dynamic window EWMA.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive rushing EPA values.
    """
    # Separate EPA into rushing offense and defense
    rushing_offense_epa = data.loc[(data['play_type'].isin(['run', 'qb_kneel'])), :] \
        .groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()

    rushing_defense_epa = data.loc[data['rush_attempt'] == 1, :] \
        .groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()

    # Lag EPA one period back
    rushing_offense_epa['epa_shifted'] = rushing_offense_epa.groupby('posteam')['epa'].shift()
    rushing_defense_epa['epa_shifted'] = rushing_defense_epa.groupby('defteam')['epa'].shift()

    # Calculate dynamic window EWMA
    rushing_offense_epa['ewma_rushing'] = rushing_offense_epa.groupby('posteam') \
        .apply(dynamic_window_ewma).values

    rushing_defense_epa['ewma_rushing'] = rushing_defense_epa.groupby('defteam') \
        .apply(dynamic_window_ewma).values

    # Merge offense and defense EPA
    rushing_epa = rushing_offense_epa.rename(columns={'posteam': 'team'}).merge(
        rushing_defense_epa.rename(columns={'defteam': 'team'}),
        on=['team', 'season', 'week'],
        suffixes=('_offense', '_defense')
    )
    features = [column for column in rushing_epa.columns if 'ewma' in column] + ['team', 'season', 'week']

    return rushing_epa[features]


def make_passing_epa(data):
    """
    Calculate the passing EPA for both offense and defense with dynamic window EWMA.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive passing EPA values.
    """
    # Separate EPA into passing offense and defense
    passing_offense_epa = data.loc[(data['play_type'].isin(['pass', 'qb_spike'])), :] \
        .groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()

    passing_defense_epa = data.loc[(data['play_type'].isin(['pass', 'qb_spike'])), :] \
        .groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()

    # Lag EPA one period back
    passing_offense_epa['epa_shifted'] = passing_offense_epa.groupby('posteam')['epa'].shift()
    passing_defense_epa['epa_shifted'] = passing_defense_epa.groupby('defteam')['epa'].shift()

    passing_offense_epa['ewma_passing'] = passing_offense_epa.groupby('posteam') \
        .apply(dynamic_window_ewma).values

    passing_defense_epa['ewma_passing'] = passing_defense_epa.groupby('defteam') \
        .apply(dynamic_window_ewma).values

    # Merge offense and defense EPA
    passing_epa = passing_offense_epa.rename(columns={'posteam': 'team'}).merge(
        passing_defense_epa.rename(columns={'defteam': 'team'}),
        on=['team', 'season', 'week'],
        suffixes=('_offense', '_defense')
    )
    features = [column for column in passing_epa.columns if 'ewma' in column] + ['team', 'season', 'week']

    return passing_epa[features]