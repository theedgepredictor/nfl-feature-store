import pandas as pd

from src.formatters.reformat_team_name import team_id_repl


def load_players():
    df = pd.read_parquet('https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet')
    df = team_id_repl(df)
    return df