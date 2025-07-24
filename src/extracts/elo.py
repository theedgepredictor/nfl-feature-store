import pandas as pd

from src.extracts.games import get_schedules
from src.formatters.general import df_rename_fold
from src.formatters.reformat_qb_names import fix_elo_qb_names
from src.formatters.reformat_team_name import team_id_repl


def get_elo(season):
    try:
        df = pd.read_parquet(f'https://github.com/theedgepredictor/elo-rating/raw/main/data/elo/football/nfl/{season}.parquet')
        return df
    except:
        return pd.DataFrame()


def get_qb_elo(seasons, season_type='REG'):
    df = get_schedules(seasons, season_type=None)[['game_id', 'game_type', 'season', 'week', 'home_team', 'away_team']]

    elo_df = pd.read_csv("https://raw.githubusercontent.com/greerreNFL/nfeloqb/main/qb_elos.csv")
    elo_df = elo_df[elo_df.season.isin(seasons)].copy()
    elo_df = team_id_repl(elo_df)
    elo_df = fix_elo_qb_names(elo_df)
    elo_df = df_rename_fold(elo_df, t1_prefix='1', t2_prefix='2')
    elo_df['is_postseason'] = elo_df.playoff.notnull()
    

    # Only keep relevant columns and rename
    elo_df = elo_df[[
        'season',
        'week',
        'team',
        'qbelo_pre',
        'qbelo_prob',
        'qbelo_post',
    ]].rename(columns={
        'qbelo_pre': 'elo_pre',
        'qbelo_prob': 'elo_prob',
        'qbelo_post': 'elo_post',
    })
    elo_df['week'] = elo_df['week'].astype(int)

    # Project elos for any week with scheduled games but missing elo data
    for season in seasons:
        season_weeks = df[df['season'] == season]['week'].unique()
        for week in sorted(season_weeks):
            if not ((elo_df['season'] == season) & (elo_df['week'] == week)).any():
                if week == 1:
                    # 1/3 regression for week 1
                    prev_season = season - 1
                    prev_season_df = elo_df[elo_df['season'] == prev_season]
                    if not prev_season_df.empty:
                        last_week = prev_season_df['week'].max()
                        last_elo = prev_season_df[prev_season_df['week'] == last_week][['team', 'elo_post']]
                        mean_elo = last_elo['elo_post'].mean()
                        last_elo['elo_pre'] = mean_elo + (last_elo['elo_post'] - mean_elo) * (2/3)
                        last_elo['elo_prob'] = None
                        last_elo['elo_post'] = last_elo['elo_pre']
                        last_elo['season'] = season
                        last_elo['week'] = 1
                        elo_df = pd.concat([elo_df, last_elo[['season', 'week', 'team', 'elo_pre', 'elo_prob', 'elo_post']]], ignore_index=True)
                else:
                    # For other weeks, use previous week's elo_post as elo_pre
                    prev_week_df = elo_df[(elo_df['season'] == season) & (elo_df['week'] == week - 1)][['team', 'elo_post']]
                    if not prev_week_df.empty:
                        next_elo = prev_week_df.copy()
                        next_elo['elo_pre'] = next_elo['elo_post']
                        next_elo['elo_prob'] = None
                        next_elo['elo_post'] = next_elo['elo_pre']
                        next_elo['season'] = season
                        next_elo['week'] = week
                        elo_df = pd.concat([elo_df, next_elo[['season', 'week', 'team', 'elo_pre', 'elo_prob', 'elo_post']]], ignore_index=True)

    away_elo_df = elo_df.rename(columns={
        'team': 'away_team',
        'elo_pre': 'away_elo_pre',
        'elo_prob': 'away_elo_prob',
        'elo_post': 'away_elo_post',
    })

    home_elo_df = elo_df.rename(columns={
        'team': 'home_team',
        'elo_pre': 'home_elo_pre',
        'elo_prob': 'home_elo_prob',
        'elo_post': 'home_elo_post',
    })

    df = pd.merge(df, home_elo_df, on=['season', 'week', 'home_team'], how='left')
    df = pd.merge(df, away_elo_df, on=['season', 'week', 'away_team'], how='left')
    if season_type == 'REG':
        df = df[df.game_type == 'REG'].copy()
    return df.drop(columns=['game_type'])