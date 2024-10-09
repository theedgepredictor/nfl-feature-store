import pandas as pd

from src.formatters.reformat_pbp import plays_formatting
from src.formatters.reformat_team_name import team_id_repl


def get_play_by_play(season, season_type=None):
    print(f'   --- Loading {season}')

    try:
        data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet')
        data = plays_formatting(data)
        data = team_id_repl(data)
        if season_type is not None:
            data = data[(data['season_type'] == season_type)]
        return data
    except:
        return pd.DataFrame()

def load_mult_lats():
    mult_lats = pd.read_csv("https://github.com/nflverse/nflverse-data/releases/download/misc/multiple_lateral_yards.csv")

    # Step 1: Extract 'season' and 'week' from 'game_id'
    mult_lats['season'] = mult_lats['game_id'].str.slice(0, 4).astype(int)
    mult_lats['week'] = mult_lats['game_id'].str.slice(5, 7).astype(int)

    # Step 2: Filter rows where 'yards' is not 0
    mult_lats = mult_lats[mult_lats['yards'] != 0]

    # Step 3: Group by 'game_id' and 'play_id' and remove the last entry in each group
    mult_lats = mult_lats.groupby(['game_id', 'play_id']).apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

    # Step 4: Handle cases where a player collects lateral yards multiple times
    # Group by 'season', 'week', 'type', and 'gsis_player_id' and aggregate the 'yards'
    mult_lats_aggregated = (
        mult_lats.groupby(['season', 'week', 'type', 'gsis_player_id'])
            .agg({'yards': 'sum'})
            .reset_index()
    )
    return mult_lats_aggregated

