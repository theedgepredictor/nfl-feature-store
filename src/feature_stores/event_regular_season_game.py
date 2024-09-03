import datetime

import numpy as np
import pandas as pd

###########################################################
## Loaders
###########################################################
from src.utils import get_seasons_to_update

EXPERIMENT_SCORES = {}


def get_play_by_play(season):
    try:
        df = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet')
        df.fillna(-1000000, inplace=True)
        df.replace(-1000000, None, inplace=True)
        return df
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

def load_data(load_seasons):
    print(f"    Loading play-by-play data {datetime.datetime.now()}")

    data = pd.concat([get_play_by_play(season) for season in load_seasons])
    data = data[(data.season_type=='REG')].copy()
    print(f"    Loading schedule data {datetime.datetime.now()}")

    schedule = get_schedules(load_seasons)

    print(f"    Loading elo data {datetime.datetime.now()}")

    elo = pd.concat([get_elo(season) for season in load_seasons])
    return data, schedule, elo

###########################################################
## Preprocessing
###########################################################

def dynamic_window_ewma(x):
    """
    Calculate rolling exponentially weighted EPA with a dynamic window size
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x.epa_shifted[:i + 1]
        if row.week > 10:
            values[i] = epa.ewm(min_periods=1, span=row.week).mean().values[-1]
        else:
            values[i] = epa.ewm(min_periods=1, span=10).mean().values[-1]

    return pd.Series(values, index=x.index)


def make_rushing_epa(data):
    """
    Calculate the rushing EPA for both offense and defense with dynamic window EWMA.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive rushing EPA values.
    """
    # Separate EPA into rushing offense and defense
    rushing_offense_epa = data.loc[data['rush_attempt'] == 1, :] \
        .groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()

    rushing_defense_epa = data.loc[data['rush_attempt'] == 1, :] \
        .groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()

    # Lag EPA one period back
    rushing_offense_epa['epa_shifted'] = rushing_offense_epa.groupby('posteam')['epa'].shift()
    rushing_defense_epa['epa_shifted'] = rushing_defense_epa.groupby('defteam')['epa'].shift()

    # Calculate dynamic window EWMA
    rushing_offense_epa['ewma_dynamic_window_rushing'] = rushing_offense_epa.groupby('posteam') \
        .apply(dynamic_window_ewma).values

    rushing_defense_epa['ewma_dynamic_window_rushing'] = rushing_defense_epa.groupby('defteam') \
        .apply(dynamic_window_ewma).values

    # Merge offense and defense EPA
    rushing_epa = rushing_offense_epa.rename(columns={'posteam': 'team'}).merge(
        rushing_defense_epa.rename(columns={'defteam': 'team'}),
        on=['team', 'season', 'week'],
        suffixes=('_offense', '_defense')
    )
    features = [column for column in rushing_epa.columns if 'ewma' in column and 'dynamic' in column] + ['team', 'season', 'week']

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
    passing_offense_epa = data.loc[data['pass_attempt'] == 1, :] \
        .groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()

    passing_defense_epa = data.loc[data['pass_attempt'] == 1, :] \
        .groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()

    # Lag EPA one period back
    passing_offense_epa['epa_shifted'] = passing_offense_epa.groupby('posteam')['epa'].shift()
    passing_defense_epa['epa_shifted'] = passing_defense_epa.groupby('defteam')['epa'].shift()

    passing_offense_epa['ewma_dynamic_window_passing'] = passing_offense_epa.groupby('posteam') \
        .apply(dynamic_window_ewma).values

    passing_defense_epa['ewma_dynamic_window_passing'] = passing_defense_epa.groupby('defteam') \
        .apply(dynamic_window_ewma).values

    # Merge offense and defense EPA
    passing_epa = passing_offense_epa.rename(columns={'posteam': 'team'}).merge(
        passing_defense_epa.rename(columns={'defteam': 'team'}),
        on=['team', 'season', 'week'],
        suffixes=('_offense', '_defense')
    )
    features = [column for column in passing_epa.columns if 'ewma' in column and 'dynamic' in column] + ['team', 'season', 'week']

    return passing_epa[features]


def make_score_feature(data):
    """
    Calculate the score for both offense and defense with dynamic window EWMA.

    Parameters:
        schedule (DataFrame): Dataframe containing the schedule and scores.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive score values with EWMA.
    """
    # Separate EPA into passing offense and defense
    passing_offense_epa = data \
        .groupby(['posteam', 'season', 'week'], as_index=False)['posteam_score_post'].last()

    passing_defense_epa = data \
        .groupby(['defteam', 'season', 'week'], as_index=False)['defteam_score_post'].last()

    # Lag EPA one period back
    passing_offense_epa['epa_shifted'] = passing_offense_epa.groupby('posteam')['posteam_score_post'].shift()
    passing_defense_epa['epa_shifted'] = passing_defense_epa.groupby('defteam')['defteam_score_post'].shift()

    passing_offense_epa['ewma_dynamic_window_score'] = passing_offense_epa.groupby('posteam') \
        .apply(dynamic_window_ewma).values

    passing_defense_epa['ewma_dynamic_window_score'] = passing_defense_epa.groupby('defteam') \
        .apply(dynamic_window_ewma).values

    # Merge offense and defense EPA
    passing_epa = passing_offense_epa.rename(columns={'posteam': 'team'}).merge(
        passing_defense_epa.rename(columns={'defteam': 'team'}),
        on=['team', 'season', 'week'],
        suffixes=('_offense', '_defense')
    )
    features = [column for column in passing_epa.columns if 'ewma' in column and 'dynamic' in column] + ['team', 'season', 'week']

    return passing_epa[features]


def make_cover_feature(schedule):
    """
    Calculate the cover feature for both the team (home or away) and whether the game went under.

    Parameters:
        schedule (DataFrame): DataFrame containing the schedule, scores, spread, and total line.

    Returns:
        DataFrame: DataFrame with added columns for rolling average of team covering and under cover.
    """
    # Calculate if the away team covered the spread and if the game went under
    schedule['away_team_covered'] = (schedule['away_score'] + schedule['spread_line'] >= schedule['home_score']).astype(int)
    schedule['home_team_covered'] = (schedule['home_score'] - schedule['spread_line'] >= schedule['away_score']).astype(int)
    schedule['under_covered'] = (schedule['home_score'] + schedule['away_score'] <= schedule['total_line']).astype(int)

    folded_df = schedule.drop(columns=['home_team_win', 'away_team_spread', 'total_target'])
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
    home_a = folded_df[folded_df.ishome == True][['season', 'week', 'team', 'rolling_team_cover', 'rolling_under_cover']].rename(columns={'team': 'home_team', 'rolling_team_cover': 'rolling_home_team_spread_cover', 'rolling_under_cover': 'rolling_home_team_under_cover'})
    away_a = folded_df[folded_df.ishome == False][['season', 'week', 'team', 'rolling_team_cover', 'rolling_under_cover']].rename(columns={'team': 'away_team', 'rolling_team_cover': 'rolling_away_team_spread_cover', 'rolling_under_cover': 'rolling_away_team_under_cover'})
    return away_a, home_a


def preprocess(data, schedule, elo):
    schedule['away_team'] = schedule['away_team'].str.replace("SD", "LAC").str.replace("OAK", "LV").str.replace("STL", "LA")
    schedule['home_team'] = schedule['home_team'].str.replace("SD", "LAC").str.replace("OAK", "LV").str.replace("STL", "LA")

    s = schedule[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'spread_line', 'total_line']].drop_duplicates().reset_index(drop=True) \
        .assign(
        home_team_win=lambda x: (x.home_score > x.away_score),
        away_team_spread=lambda x: (x.home_score - x.away_score),
        total_target=lambda x: (x.home_score + x.away_score),
    )

    away_a, home_a = make_cover_feature(s)

    s['away_team_covered_spread'] = (s['away_score'] + s['spread_line'] >= s['home_score'])

    # Calculate if the game covered the under
    s['under_covered'] = (s['home_score'] + s['away_score'] <= s['total_line'])

    epa = pd.merge(
        make_rushing_epa(data),
        make_passing_epa(data),
        on=['team', 'season', 'week'],
    )

    epa = pd.merge(
        epa,
        make_score_feature(data),
        on=['team', 'season', 'week'],
    )

    df = s.merge(
        epa.rename(columns={'team': 'home_team'}),
        on=['home_team', 'season', 'week'],
        how='left'
    ).merge(
        epa.rename(columns={'team': 'away_team'}),
        on=['away_team', 'season', 'week'],
        how='left',
        suffixes=('_home', '_away')
    )

    df = df.merge(away_a, on=['season', 'week', 'away_team'], how='left').merge(home_a, on=['season', 'week', 'home_team'], how='left')

    # Remove the first week of the dataset since it is used as aggregate
    df = df[~((df.season == df.season.min()) & (df.week == df.week.min()))].copy()
    # Remove where the spread or total line is missing and games havent happened yet
    df = df.dropna()
    df[['home_team_win', 'away_team_spread', 'total_target', 'away_team_covered_spread', 'under_covered']] = df[['home_team_win', 'away_team_spread', 'total_target', 'away_team_covered_spread', 'under_covered']].astype(int)

    # Make Inference set
    inference_df = s[((s.home_score.isnull()) & (s.away_score.isnull()) & (s.spread_line.notna()) & (s.total_line.notna()))].copy()
    latest_epa = epa.groupby('team').nth(-1).drop(columns=['week', 'season'])

    inference_df = inference_df.merge(
        latest_epa.rename(columns={'team': 'home_team'}),
        on=['home_team'],
        how='left'
    ).merge(
        latest_epa.rename(columns={'team': 'away_team'}),
        on=['away_team'],
        how='left',
        suffixes=('_home', '_away')
    )

    df = pd.concat([df, inference_df])
    elo = pd.merge(elo[['id', 'away_elo_pre', 'away_elo_prob', 'home_elo_pre', 'home_elo_prob']], schedule[['espn', 'away_team', 'home_team', 'season', 'week']].rename(columns={'espn': 'id'}), on=['id'], how='inner').drop(columns=['id'])

    df = df.merge(elo, on=['season', 'week', 'away_team', 'home_team'], how='left')

    return df


def df_rename_fold(df, t1_prefix, t2_prefix):
    '''
    The reverse of a df_rename_pivot
    Fold two prefixed column types into one generic type
    Ex: away_team_id and home_team_id -> team_id
    '''
    try:
        t1_all_cols = [i for i in df.columns if t2_prefix not in i]
        t2_all_cols = [i for i in df.columns if t1_prefix not in i]

        t1_cols = [i for i in df.columns if t1_prefix in i]
        t2_cols = [i for i in df.columns if t2_prefix in i]
        t1_new_cols = [i.replace(t1_prefix, '') for i in df.columns if t1_prefix in i]
        t2_new_cols = [i.replace(t2_prefix, '') for i in df.columns if t2_prefix in i]

        t1_df = df[t1_all_cols].rename(columns=dict(zip(t1_cols, t1_new_cols)))
        t2_df = df[t2_all_cols].rename(columns=dict(zip(t2_cols, t2_new_cols)))

        df_out = pd.concat([t1_df, t2_df]).reset_index().drop(columns='index')
        return df_out
    except Exception as e:
        print("--df_rename_fold-- " + str(e))
        print(f"columns in: {df.columns}")
        print(f"shape: {df.shape}")
        return df

def make_event_regular_season_feature_store(load_seasons):
    data, schedule, elo = load_data(load_seasons)

    print(f"    Preprocessing event regular season feature store {datetime.datetime.now()}")
    return preprocess(data, schedule, elo)
