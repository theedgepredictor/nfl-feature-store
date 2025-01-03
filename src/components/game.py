import datetime
import pandas as pd
from src.extracts.elo import get_qb_elo
from src.extracts.games import get_schedules
from src.transforms.targets import event_targets
from src.transforms.vegas_lines import make_cover_feature


class GameComponent:
    def __init__(self, load_seasons, season_type=None):
        self.load_seasons = load_seasons
        self.season_type = season_type
        self.db = self.extract()
        self.df = self.run_pipeline()

    def extract(self):
        """
        Extracting play by play data, schedules, elo and weekly offensive and defensive player metrics (rolled up into total team metrics).
        Each of these data groups are extracted and loaded for the given seasons and filtered for the regular season
        :param load_seasons:
        :return:
        """
        print(f"    Loading schedule data {datetime.datetime.now()}")

        schedule = get_schedules(self.load_seasons, self.season_type)
        elo = get_qb_elo(self.load_seasons, self.season_type)

        return {
            'games': schedule,
            'elo': elo,
        }

    def run_pipeline(self):
        df = self._game_pipeline()
        groups = [
            self._target_pipeline(),
            self._lines_pipeline()
        ]
        for group in groups:
            df = pd.merge(df, group, on=['home_team', 'away_team', 'season', 'week'], how='left')
        df = self._add_rolling_cover_pipeline(df.copy())
        df = self._add_elo_pipeline(df.copy())
        return df[[
            'home_team',
            'away_team',
            'season',
            'week',
            'home_rest',
            'away_rest',
            'actual_away_team_win',
            'actual_away_spread',
            'actual_point_total',
            'actual_away_team_covered_spread',
            'actual_under_covered',
            'actual_home_score',
            'actual_away_score',
            'spread_line',
            'total_line',
            'home_moneyline',
            'away_moneyline',
            'home_rolling_spread_cover',
            'away_rolling_spread_cover',
            'home_rolling_under_cover',
            'away_rolling_under_cover',
            'home_elo_pre',
            'home_elo_prob',
            'away_elo_pre',
            'away_elo_prob'
        ]]

    def _game_pipeline(self):
        """

        :return:
        """

        df = self.db['games'][
            [
                'season',
                'week',
                'home_team',
                'away_team',
                'home_rest',
                'away_rest',
            ]
        ].copy().drop_duplicates(subset=['season', 'week', 'home_team', 'away_team']).reset_index(drop=True)
        df['game_id'] = df.apply(lambda x: f"{x['season']}_{x['week']}_{x['away_team']}_{x['home_team']}", axis=1)
        return df

    def _lines_pipeline(self):
        df = self.db['games'][
            [
                'season',
                'week',
                'home_team',
                'away_team',
                'spread_line',
                'total_line',
                'away_moneyline',
                'home_moneyline',
            ]
        ].copy().drop_duplicates(subset=['season', 'week', 'home_team', 'away_team']).reset_index(drop=True)
        return df

    def _target_pipeline(self):
        return event_targets(self.db['games'].copy())

    def _add_rolling_cover_pipeline(self, df):
        away_a, home_a = make_cover_feature(df)
        df = df.merge(away_a, on=['season', 'week', 'away_team'], how='left').merge(home_a, on=['season', 'week', 'home_team'], how='left')
        return df

    def _add_elo_pipeline(self, df):
        df = df.merge(self.db['elo'], on=['season', 'week', 'away_team', 'home_team'], how='left')
        return df

