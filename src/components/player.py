import datetime

import pandas as pd

from src.extracts.games import get_schedules
from src.extracts.pbp import get_play_by_play
from src.extracts.player_stats import collect_roster, collect_injuries
from src.formatters.general import df_rename_fold
from src.transforms.general import stat_collection
from src.transforms.player import make_player_avg_group_features


class GamePlayerComponent:
    def __init__(self, load_seasons, season_type=None):
        self.load_seasons = load_seasons
        self.season_type = season_type
        self.db = self.extract()
        self.df = self.run_pipeline()

    def extract(self):
        """
        Extracting play by play data and weekly offensive player metrics
        Each of these data groups are extracted and loaded for the given seasons and filtered for the regular season
        :param load_seasons:
        :return:
        """
        ### Schedule
        print(f"    Loading schedule data {datetime.datetime.now()}")
        schedule = get_schedules(self.load_seasons, self.season_type)

        ### Rosters
        print(f"    Loading weekly player roster data {datetime.datetime.now()}")
        rosters = pd.concat([collect_roster(season) for season in self.load_seasons])

        ### Injury Reports
        print(f"    Loading weekly player injury report data {datetime.datetime.now()}")
        injuries = pd.concat([collect_injuries(season) for season in self.load_seasons])

        return {
            'games': schedule,
            'rosters': rosters,
            'injuries': injuries,
        }

    def run_pipeline(self):
        games_df = self._game_pipeline()

        df = self._roster_pipeline()

        df = games_df.merge(df, how='left', on=[
            'season',
            'team',
            'week',
        ])

        df = self._injury_pipeline(df)
        df = df[df.position_group=='quarterback'].copy()

        return df

    def _game_pipeline(self):
        """
        Add datetime reference and event context
        :return:
        """

        df = self.db['games'][
            [
                'season',
                'week',
                'home_team',
                'away_team',
                'gameday',
                'gametime',
                'game_type',
                'result'
            ]
        ].copy().drop_duplicates(subset=['season', 'week', 'home_team', 'away_team']).reset_index(drop=True)
        df['datetime'] = df['gameday'] + ' ' + df['gametime']
        df.datetime = pd.to_datetime(df.datetime)
        df = df.drop(columns=['gameday','gametime'])
        ### Add 4 hrs to datetime column and then convert back to datetime in utc since were in ET
        df.datetime = df.datetime + pd.Timedelta(hours=4)
        df.datetime = pd.to_datetime(df.datetime, utc=True)
        df['game_id'] = df.apply(lambda x: f"{x['season']}_{x['week']}_{x['away_team']}_{x['home_team']}", axis=1)
        df['future'] = 1
        df.loc[df.result.notnull(),'future'] = 0
        df = df[df.game_type == self.season_type].copy().drop(columns=['game_type', 'result'])

        return df_rename_fold(df, 'away_','home_')

    def _roster_pipeline(self):
        """
        Adds roster context and roster position and status
        :return:
        """
        df = self.db['rosters'][[
            'player_id',
            'season',
            'week',
            'team',
            'high_pos_group',
            'position_group',
            'position',
            'jersey_number',
            'status_abbr',
        ]]
        df['game_type'] ='REG'
        df.loc[(((df.season >= 2021) & (df.week > 18)) | ((df.season < 2021) & (df.week > 17))), 'game_type'] = 'POST'
        df = df[df.game_type==self.season_type].copy().drop(columns='game_type')
        return df

    def _injury_pipeline(self, df):
        injury_df = self.db['injuries']
        if injury_df.shape[0] == 0:
            for fill_col in ['pre_kickoff_injury_designation','report_primary_injury','report_secondary_injury','report_status','practice_primary_injury','practice_secondary_injury','practice_status','date_modified']:
                df[fill_col] = None
            return df

        injury_df = injury_df[[
            'season',
            'game_type',
            'team',
            'week',
            'player_id',
            'report_primary_injury',
            'report_secondary_injury',
            'report_status',
            'practice_primary_injury',
            'practice_secondary_injury',
            'practice_status',
            'date_modified',
        ]]

        injury_df = injury_df[injury_df.game_type == self.season_type].copy().drop(columns='game_type')
        df = df.merge(injury_df, how='left', on=[
            'player_id',
            'season',
            'team',
            'week',
        ])
        df['datetime'] = pd.to_datetime(df.datetime)
        df['date_modified'] = pd.to_datetime(df.date_modified)
        df['pre_kickoff_injury_designation'] = df['datetime'] > df['date_modified']
        return df

if __name__ == '__main__':
    game_player_component = GamePlayerComponent([2023,2024], season_type='REG')
    df = game_player_component.run_pipeline()