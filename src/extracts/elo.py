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
    elo_df = elo_df[[
        'season',
        'week',
        'team',
        # 'elo_pre',
        # 'elo_prob',
        # 'elo_post',
        'qbelo_pre',
        # 'qb',
        # 'qb_value_pre',
        # 'qb_adj',
        'qbelo_prob',
        # 'qb_game_value',
        # 'qb_value_post',
        'qbelo_post',
    ]].rename(columns={
        'qbelo_pre': 'elo_pre',
        'qbelo_prob': 'elo_prob',
        'qbelo_post': 'elo_post',
    })
    elo_df['week'] = elo_df['week'].astype(int)

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