import datetime

import pandas as pd

from src.extracts.games import get_schedules
from src.extracts.pbp import get_play_by_play
from src.extracts.player_stats import collect_roster, collect_injuries, get_player_fantasy_projections
from src.formatters.general import df_rename_fold
from src.transforms.general import stat_collection
from src.transforms.player import make_player_avg_group_features


class WeeklyPlayerStatComponent:
    def __init__(self, load_seasons, season_type=None, group = 'off'):
        self.load_seasons = load_seasons
        self.season_type = season_type
        self.group = group
        self.db = self.extract()
        self.df = self.run_pipeline()

    def extract(self):
        """
        Extracting play by play data and weekly offensive player metrics
        Each of these data groups are extracted and loaded for the given seasons and filtered for the regular season
        :param load_seasons:
        :return:
        """

        if self.group == 'off':
            print(f"    Loading offensive player weekly data {datetime.datetime.now()}")
            stats_weekly = pd.concat([stat_collection(season, season_type=self.season_type, mode='player') for season in self.load_seasons])
        else:
            stats_weekly = None

        ## Get Defense

        ## Get Special Teams

        ## Add play by play for rush location distributions and pass location distributions
        ### - We dont have a lot of data for OLINE otherwise

        return {
            'player_stats': stats_weekly,
        }

    def run_pipeline(self):
        if self.group == 'off':
            #### Run offensive player weekly feature creation
            player_df = self._offensive_player_pipeline()
        else:
            #### Run defensive player weekly feature creation
            #player_df = None
            #### Run special teams player weekly feature creation
            player_df = None

        return player_df

    def _offensive_player_pipeline(self):
        filters = [
            'player_id',
            'season',
            'week',
        ]
        passing_stats = [
            'completions',
            'attempts',
            'passing_yards',
            'passing_tds',
            'interceptions',
            'sacks',
            'sack_yards',
            'sack_fumbles',
            'sack_fumbles_lost',
            'passing_air_yards',
            'passing_yards_after_catch',
            'passing_first_downs',
            'passing_epa',
            'passing_2pt_conversions',
            'pacr',
            'dakota',
            'avg_time_to_throw',
            'avg_completed_air_yards',
            'avg_intended_air_yards_passing',
            'avg_air_yards_differential',
            'aggressiveness',
            'max_completed_air_distance',
            'avg_air_yards_to_sticks',
            'passer_rating',
            'VALUE_ELO',
            'completion_percentage',
            'expected_completion_percentage',
            'completion_percentage_above_expectation',
            'avg_air_distance',
            'max_air_distance',
            'net_passing_yards',
            'yards_per_pass_attempt',
            'sack_rate',
            'air_yards_per_pass_attempt',
            # 'qbr'
        ]

        rushing_stats = [
            'carries',
            'rushing_yards',
            'rushing_tds',
            'rushing_fumbles',
            'rushing_fumbles_lost',
            'rushing_first_downs',
            'rushing_epa',
            'rushing_2pt_conversions',
            'efficiency',
            'percent_attempts_gte_eight_defenders',
            'avg_time_to_los',
            'avg_rush_yards',
            'expected_rush_yards',
            'rush_yards_over_expected',
            'rush_yards_over_expected_per_att',
            'rush_pct_over_expected',
            'yards_per_rush_attempt',
        ]

        receiving_stats = [
            'receptions',
            'targets',
            'receiving_yards',
            'receiving_tds',
            'receiving_fumbles',
            'receiving_fumbles_lost',
            'receiving_air_yards',
            'receiving_yards_after_catch',
            'receiving_first_downs',
            'receiving_epa',
            'receiving_2pt_conversions',
            'racr',
            'target_share',
            'air_yards_share',
            'wopr',
            'avg_cushion',
            'avg_separation',
            'avg_intended_air_yards_receiving',
            'percent_share_of_intended_air_yards',
            'catch_percentage',
            'avg_yac',
            'avg_expected_yac',
            'avg_yac_above_expectation',
        ]

        general_stats = [
            'special_teams_tds',
            #'fantasy_points',
            #'fantasy_points_half_ppr',
            'fantasy_points_ppr',
            'total_plays',
            'total_yards',
            'total_fumbles',
            'total_fumbles_lost',
            'total_turnovers',
            'total_touchdowns',
            'total_first_downs',
            'touchdown_per_play',
            'yards_per_play',
            'fantasy_point_per_play',
        ]
        attrs = passing_stats+rushing_stats+receiving_stats+general_stats
        #attrs = ['fantasy_points_ppr','VALUE_ELO','passing_epa','attempts','total_plays']
        group_features_dict = {i: 'sum' for i in attrs}
        cols = filters+attrs

        off_frame = self.db['player_stats'][['position_group']+cols]
        off_frame = off_frame[off_frame.position_group=='quarterback'].copy()
        #droppers = [i for i in df.columns if i not in cols]

        for mode in ['season_avg', 'season_total', 'form']:
            attrs_df = make_player_avg_group_features(off_frame, group_features_dict=group_features_dict, mode=mode)
            if off_frame.shape[0] == 0:
                off_frame = attrs_df
            else:
                off_frame = pd.merge(off_frame, attrs_df, on=['player_id', 'season', 'week'])

        ## Ranks (Scrapping ranks for now. Going to normalize features already)
        '''
        group_features_rank_dict = {f'season_avg_{i}': 'max' for i, val in list(group_features_dict.items())}
        off_frame_ranks = calculate_ranks(off_frame[['position_group','player_id','season','week']+list(group_features_rank_dict.keys())].fillna(0).copy(), group_by_col=['position_group','season', 'week'], rank_cols_methods=group_features_rank_dict).drop(columns=['position_group'])
        off_frame = pd.merge(off_frame, off_frame_ranks, on=['player_id', 'season', 'week'])'''
        return off_frame




if __name__ == '__main__':
    player_stat_component = WeeklyPlayerStatComponent([2002,2003], season_type='REG', group='off')
    df = player_stat_component.run_pipeline()