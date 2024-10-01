import pandas as pd

from src.formatters.reformat_game_scores import score_clean
from src.formatters.reformat_team_name import team_id_repl


def get_schedules(seasons, season_type='REG'):
    if min(seasons) < 1999:
        raise ValueError('Data not available before 1999.')
    ## apply ##
    scheds = pd.read_csv('http://www.habitatring.com/games.csv')
    scheds = score_clean(scheds)
    scheds = scheds[scheds['season'].isin(seasons)].copy()
    if season_type == 'REG':
        scheds = scheds[scheds.game_type=='REG'].copy()
    scheds = team_id_repl(scheds)
    return scheds
