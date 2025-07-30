import pandas as pd

from src.feature_stores.event_regular_season_game import VEGAS, META, TARGETS, POINT_FEATURES, KICKING_FEATURES, RANKING_FEATURES, COMMON_RUSHING_FEATURES, COMMON_PASSING_FEATURES, PENALTY_FEATURES, COMMON_FEATURES, FANTASY_FEATURES, DOWN_FEATURES, ROLLING_COVER_FEATURES, EWMA_FEATURES
from src.formatters.general import df_rename_shift, df_rename_exavg, df_rename_fold

def get_event_feature_store(season):
    #return pd.read_parquet(f'../nfl-feature-store/data/feature_store/event/regular_season_game/{season}.parquet')
    return pd.read_parquet(f'https://github.com/theedgepredictor/nfl-feature-store/raw/main/data/feature_store/event/regular_season_game/{season}.parquet')

def load_exavg_event_feature_store(seasons):
    event_fs = pd.concat([get_event_feature_store(season) for season in seasons], ignore_index=True)
    event_fs = event_fs[event_fs.away_elo_pre.notnull()].copy()

    columns_for_base = META + ['home_elo_pre', 'away_elo_pre'] + VEGAS + TARGETS + ['away_offensive_rank','away_defensive_rank','home_offensive_rank','home_defensive_rank',]
    columns_for_shift = ['team', 'season', 'week', 'is_home'] + POINT_FEATURES + KICKING_FEATURES + RANKING_FEATURES + COMMON_RUSHING_FEATURES + COMMON_PASSING_FEATURES + COMMON_FEATURES + FANTASY_FEATURES + DOWN_FEATURES + EWMA_FEATURES
    shifted_df = event_fs.copy()
    base_dataset_df = event_fs[columns_for_base].copy()

    del event_fs

    #### Shift Features
    shifted_df = df_rename_shift(shifted_df)[columns_for_shift]

    #### Rename for Expected Average
    t1_cols = [i for i in shifted_df.columns if '_offense' in i and (i not in TARGETS + META) and i.replace('home_', '') in columns_for_shift]
    t2_cols = [i for i in shifted_df.columns if '_defense' in i and (i not in TARGETS + META) and i.replace('away_', '') in columns_for_shift]

    #### Apply Expected Average
    expected_features_df = df_rename_exavg(shifted_df, '_offense', '_defense', t1_cols=t1_cols, t2_cols=t2_cols)

    #### Rename back into home and away features
    home_exavg_features_df = expected_features_df[expected_features_df['is_home'] == 1].copy().drop(columns='is_home')
    away_exavg_features_df = expected_features_df[expected_features_df['is_home'] == 0].copy().drop(columns='is_home')
    home_exavg_features_df.columns = ["home_" + col if 'exavg_' in col or col == 'team' else col for col in home_exavg_features_df.columns]
    away_exavg_features_df.columns = ["away_" + col if 'exavg_' in col or col == 'team' else col for col in away_exavg_features_df.columns]

    #### Merge home and away Expected Average features into base as dataset_df
    dataset_df = pd.merge(base_dataset_df, home_exavg_features_df, on=['home_team', 'season', 'week'], how='left')
    dataset_df = pd.merge(dataset_df, away_exavg_features_df, on=['away_team', 'season', 'week'], how='left')
    dataset_df['game_id'] = dataset_df.apply(lambda x: f"{x['season']}_{x['week']}_{x['away_team']}_{x['home_team']}", axis=1)

    #### Fold base from away and home into team
    folded_dataset_df = base_dataset_df.copy()
    folded_dataset_df['game_id'] = folded_dataset_df.apply(lambda x: f"{x['season']}_{x['week']}_{x['away_team']}_{x['home_team']}", axis=1)
    folded_dataset_df = folded_dataset_df.rename(columns={'spread_line': 'away_spread_line'})
    folded_dataset_df['home_spread_line'] = - folded_dataset_df['away_spread_line']
    folded_dataset_df['actual_home_spread'] = -folded_dataset_df['actual_away_spread']
    folded_dataset_df['actual_home_team_win'] = folded_dataset_df['actual_away_team_win'] == 0
    folded_dataset_df['actual_home_team_covered_spread'] = folded_dataset_df['actual_away_team_covered_spread'] == 0
    folded_dataset_df = df_rename_fold(folded_dataset_df, 'away_', 'home_')
    folded_dataset_df = pd.merge(folded_dataset_df, expected_features_df, on=['team', 'season', 'week'], how='left')
    dataset_df.index = dataset_df.game_id

    # Customize Column names from feature store into friendly_names
    dataset_df['expected_spread'] = dataset_df['home_exavg_avg_points'] - dataset_df['away_exavg_avg_points']
    dataset_df['expected_total'] = dataset_df['home_exavg_avg_points'] + dataset_df['away_exavg_avg_points']

    folded_dataset_df['elo_pre'] = folded_dataset_df['elo_pre'].astype(int)
    folded_dataset_df['exavg_avg_time_of_possession'] = folded_dataset_df['exavg_avg_time_of_possession'].apply(lambda x: f"{int(x // 60)}:{int(x % 60):02}")
    return dataset_df, folded_dataset_df