import pandas as pd

from src.formatters.reformat_team_name import team_id_repl


def collect_qbr(seasons, season_type="REG"):
    season_type = "Regular" if season_type == "REG" else "Playoffs"
    qbr = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_week_level.csv')
    qbr = qbr[qbr.season.isin(seasons)].copy()
    qbr = qbr[qbr.season_type == season_type].copy()
    qbr = team_id_repl(qbr)
    p_id = pd.read_csv('https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv')
    p_id = p_id[p_id.espn_id.notnull()][['espn_id', 'gsis_id']]
    p_id.espn_id = p_id.espn_id.astype(int)
    p_id_dict = p_id.set_index('espn_id').to_dict()['gsis_id']

    qbr['player_id'] = qbr['player_id'].map(p_id_dict)
    return qbr[['season', 'week_num', 'team_abb', 'player_id', 'qbr_total']].rename(columns={'qbr_total': 'qbr', 'week_num': 'week', 'team_abb': 'team'})