import datetime

import numpy as np
import pandas as pd

###########################################################
## Loaders
###########################################################
from src.extract import get_play_by_play, get_schedules, get_elo, stat_collection
from src.transform import make_cover_feature, make_general_group_features, make_weekly_avg_group_features, make_rushing_epa, make_passing_epa, make_avg_penalty_group_features, make_score_feature

EXPERIMENT_SCORES = {}

def load_data(load_seasons):
    """
    Extracting play by play data, schedules, elo and weekly offensive and defensive player metrics (rolled up into total team metrics).
    Each of these data groups are extracted and loaded for the given seasons and filtered for the regular season
    :param load_seasons:
    :return:
    """
    print(f"    Loading play-by-play data {datetime.datetime.now()}")

    data = pd.concat([get_play_by_play(season) for season in load_seasons])
    data = data[(data.season_type == 'REG')].copy()
    print(f"    Loading schedule data {datetime.datetime.now()}")

    schedule = get_schedules(load_seasons)

    print(f"    Loading elo data {datetime.datetime.now()}")

    elo = pd.concat([get_elo(season) for season in load_seasons])

    print(f"    Loading offensive player weekly data {datetime.datetime.now()}")
    off_weekly = pd.concat([stat_collection(season, season_type="REG", mode='team') for season in load_seasons])

    print(f"    Loading defensive player weekly data {datetime.datetime.now()}")
    def_weekly = pd.concat([stat_collection(season, season_type="REG", mode='opponent') for season in load_seasons])
    return data, schedule, elo, off_weekly, def_weekly


###########################################################
## Preprocessing
###########################################################

def preprocess(data, schedule, elo, off_weekly, def_weekly):
    schedule['away_team'] = schedule['away_team'].str.replace("SD", "LAC").str.replace("OAK", "LV").str.replace("STL", "LA")
    schedule['home_team'] = schedule['home_team'].str.replace("SD", "LAC").str.replace("OAK", "LV").str.replace("STL", "LA")

    s = schedule[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'spread_line', 'total_line']].drop_duplicates(subset=['season', 'week', 'home_team', 'away_team']).reset_index(drop=True) \
        .assign(
        away_team_win=lambda x: (x.home_score < x.away_score),
        away_team_spread=lambda x: (x.home_score - x.away_score),
        total_target=lambda x: (x.home_score + x.away_score),
    )

    away_a, home_a = make_cover_feature(s)

    s['away_team_covered_spread'] = (s['away_score'] + s['spread_line'] >= s['home_score'])

    # Calculate if the game covered the under
    s['under_covered'] = (s['home_score'] + s['away_score'] <= s['total_line'])

    epa = make_score_feature(data)

    a = make_weekly_avg_group_features(off_weekly, def_weekly)
    b = make_rushing_epa(data)
    c = make_passing_epa(data)
    d = make_avg_penalty_group_features(data)
    e = make_general_group_features(data)

    groups = [
        a,b,c,d,e
    ]

    for group in groups:
        epa = pd.merge(epa, group, on=['team', 'season', 'week'], how='left')

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

    # Suffix to prefix
    df.columns = [
        'home_' + col.replace('_home', '') if '_home' in col else
        'away_' + col.replace('_away', '') if '_away' in col else
        col
        for col in df.columns
    ]

    df = df.merge(away_a, on=['season', 'week', 'away_team'], how='left').merge(home_a, on=['season', 'week', 'home_team'], how='left')

    # Remove the first week of the dataset since it is used as aggregate
    if df.season.min() <= 2002:
        df = df[~((df.season == df.season.min()) & (df.week == df.week.min()))].copy()
    # Remove where the spread or total line is missing and games havent happened yet
    df = df.dropna(subset=['home_score','away_score','spread_line','total_line'])
    df[['away_team_win', 'away_team_spread', 'total_target', 'away_team_covered_spread', 'under_covered']] = df[['away_team_win', 'away_team_spread', 'total_target', 'away_team_covered_spread', 'under_covered']].astype(int)

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

    # Suffix to prefix
    inference_df.columns = [
        'home_' + col.replace('_home', '') if '_home' in col else
        'away_' + col.replace('_away', '') if '_away' in col else
        col
        for col in inference_df.columns
    ]

    df = pd.concat([df, inference_df])
    elo = pd.merge(elo[['id', 'away_elo_pre', 'away_elo_prob', 'home_elo_pre', 'home_elo_prob']], schedule[['espn', 'away_team', 'home_team', 'season', 'week']].rename(columns={'espn': 'id'}), on=['id'], how='inner').drop(columns=['id'])

    df = df.merge(elo, on=['season', 'week', 'away_team', 'home_team'], how='left')
    # print(df.columns)
    return df


def make_event_regular_season_feature_store(load_seasons):
    data, schedule, elo, off_weekly, def_weekly = load_data(load_seasons)

    print(f"    Preprocessing event regular season feature store {datetime.datetime.now()}")
    return preprocess(data, schedule, elo, off_weekly, def_weekly)

