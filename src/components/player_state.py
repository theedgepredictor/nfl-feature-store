import datetime

import pandas as pd

from src.extracts.games import get_schedules
from src.extracts.player_stats import collect_roster, collect_injuries, get_starters, collect_players, collect_combine, collect_weekly_espn_player_stats, collect_depth_chart, get_madden_ratings, get_approximate_value
from src.formatters.general import df_rename_fold
from src.utils import find_year_for_season


class PlayerStateDataComponent:
    """
    Main class for extracting, merging, and building weekly states for NFL players.
    Handles roster, starter, injury, depth chart, and game participation data.
    For PoC, focuses on QBs.
    """
    def __init__(self, load_seasons, season_type=None):
        self.load_seasons = load_seasons
        self.season_type = season_type
        self.db = self.extract()
        self.players = self.get_static_players()
        self.team_events = self.static_team_events()
        #self.playerverse = self.init_playerverse()
        #self.df = self.run_pipeline()

        ### Create Player State

        ### Create Team State
        #### - GameID -> QB, RB, WR

    def extract(self):
        """
        Loads all raw data sources for the given seasons: schedule, rosters, injuries, starters, depth charts, stats, players, combine.
        Returns a dictionary of DataFrames.
        """
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

        ### Starters
        print(f"    Loading weekly player starter data {datetime.datetime.now()}")
        starters = pd.concat([get_starters(season) for season in self.load_seasons])

        ### Depth Charts
        print(f"    Loading weekly player depth chart data {datetime.datetime.now()}")
        depth_charts = pd.concat([collect_depth_chart(season) for season in self.load_seasons])

        ### Stats Weekly
        print(f"    Loading weekly player stats data {datetime.datetime.now()}")
        stats_weekly = collect_weekly_espn_player_stats(max(self.load_seasons), season_type=self.season_type).rename(columns={'recent_team': 'team'})

        print(f"    Loading defensive weekly player stats data {datetime.datetime.now()}")
        def_stats_weekly = collect_weekly_espn_player_stats(max(self.load_seasons), season_type=self.season_type, group='def')

        print(f"    Loading special teams weekly player stats data {datetime.datetime.now()}")
        kicking_stats_weekly = collect_weekly_espn_player_stats(max(self.load_seasons), season_type=self.season_type, group='kicking')

        ### Players
        print(f"    Loading players data {datetime.datetime.now()}")
        players = collect_players()

        combine = collect_combine()


        return {
            'games': schedule,
            'rosters': rosters,
            'injuries': injuries,
            'starters': starters,
            'depth_charts': depth_charts,
            'player_stats': stats_weekly,
            'def_player_stats': def_stats_weekly,
            'kicking_player_stats': kicking_stats_weekly,
            'players': players,
            'combine': combine
        }

    def run_pipeline(self):
        """
        Main pipeline to process and merge all player data into weekly states.
        Returns a DataFrame of player-week states.
        """

        print(f"    Making Playerverse {datetime.datetime.now()}")
        df = self.init_players()

        ### Add game participants
        df = self.add_game_participants(df)

        ### Add events
        df = self.add_valid_games(df)

        ### Add injuries
        df = self.add_injuries(df)

        ## Add status
        df = self.transform_status(df)

        df = df[[
            'game_id',
            'player_id',
            'season',
            'week',
            'team',
            'high_pos_group',
            'position_group',
            'position',
            'starter',
            'status',
            'report_status',
            'playerverse_status'
        ]]
        return df


    def get_static_players(self):
        """
        Merges player metadata with combine results for static player attributes.
        Returns a DataFrame of player metadata.
        """
        """
        Pull from player and combine data
        """
        df = self.db['players']

        ### Combine Extractor (First Come)
        combine_df = self.db['combine']
        # combine_df = combine_df[combine_df.position_group == position_group].copy()
        valid_combine_df = combine_df[combine_df.pfr_id.notnull()].copy()[[
            'pfr_id',
            'forty',
            'bench',
            'vertical',
            'broad_jump',
            'cone',
            'shuttle'
        ]]
        valid_combine_df = pd.merge(valid_combine_df, df[['player_id', 'pfr_id']], on='pfr_id', how='left').drop(columns=['pfr_id'])
        invalid_combine_df = combine_df[combine_df.pfr_id.isnull()].copy()[[
            'name',
            'position_group',
            'forty',
            'bench',
            'vertical',
            'broad_jump',
            'cone',
            'shuttle'
        ]]
        invalid_combine_df = pd.merge(invalid_combine_df, df[['name', 'position_group', 'player_id']], on=['name', 'position_group'], how='left').drop(columns=['name', 'position_group'])
        combine_df = pd.concat([valid_combine_df, invalid_combine_df], axis=0).reset_index(drop=True)
        df = df.merge(combine_df, on='player_id', how='left')
        df = df[[
            'player_id',
            'name',
            # 'common_first_name',
            'first_name',
            'last_name',
            # 'short_name',
            # 'football_name',
            # 'suffix',
            # 'esb_id',
            # 'nfl_id',
            'pfr_id',
            # 'pff_id',
            # 'otc_id',
            'espn_id',
            # 'smart_id',
            'birth_date',
            # 'high_pos_group',
            # 'position_group',
            # 'position',
            'height',
            'weight',
            'headshot',
            'college_name',
            'college_conference',
            # 'jersey_number',
            'rookie_season',
            # 'last_season',
            # 'latest_team',
            # 'status',
            # 'status_abbr',
            # 'ngs_status',
            # 'ngs_status_short_description',
            # 'years_of_experience',
            # 'pff_position',
            # 'pff_status',
            'draft_year',
            'draft_round',
            'draft_pick',
            'draft_team',
            'forty',
            'bench',
            'vertical',
            'broad_jump',
            'cone',
            'shuttle'
        ]]
        # df['last_updated'] = pd.to_datetime('now')
        return df

    def static_team_events(self):
        """
        Processes schedule data into team-level events, including game datetime and opponent info.
        Returns a DataFrame of team events.
        """
        event_df = self.db['games'].sort_values(
            by=['season', 'week', 'game_id'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        event_df['datetime'] = event_df['gameday'] + ' ' + event_df['gametime']
        event_df.datetime = pd.to_datetime(event_df.datetime)
        event_df['datetime'] = event_df['datetime'].fillna(event_df['gameday'])
        event_df.datetime = pd.to_datetime(event_df.datetime)
        event_df = event_df.drop(columns=['gametime'])
        ### Add 4 hrs to datetime column and then convert back to datetime in utc since were in ET
        event_df.datetime = event_df.datetime + pd.Timedelta(hours=4)
        event_df.datetime = pd.to_datetime(event_df.datetime, utc=True)
        event_df['away_opponent'] = event_df['home_team']
        event_df['home_opponent'] = event_df['away_team']
        event_df['is_home'] = event_df['home_team']

        fold_df = df_rename_fold(event_df, 'away_', 'home_').sort_values(
            by=['season', 'week', 'game_id'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        fold_df['is_home'] = fold_df['is_home'] == fold_df['team']

        return fold_df[[
            'game_id',
            'season',
            'game_type',
            'week',
            'datetime',
            'team',
            'opponent',
            'score',
            'rest',
            'qb_id',
            #'qb_name',
            #'coach',
            'is_home'
        ]]

    def init_players(self):
        """
        Creates a base DataFrame of all players for all weeks/seasons, based on rookie season.
        Returns a DataFrame of player-week snapshots.
        """
        players_df = self.players[[
            'player_id',
            'espn_id',
            'pfr_id',
            'birth_date',
            # 'height',
            # 'weight',
            'rookie_season',
            'draft_year',
        ]]
        playerverse = []
        ### Create snapshot for each player for each valid season week pair based on rookie season
        for season in range(1999, find_year_for_season()+1):
            for week in range(1, (18 + 1 + 4 if season >= 2021 else 17 + 1 + 4)):
                snapshot = players_df[players_df.rookie_season <= season].copy()
                snapshot['season'] = season
                snapshot['week'] = week
                playerverse.append(snapshot)
        df = pd.concat(playerverse, axis=0).reset_index(drop=True)
        return df

    def init_playerverse(self):
        """
        Builds the playerverse by merging in game participants, valid games, and injuries.
        Returns a DataFrame of enriched player-week states.
        """
        print(f"    Making Playerverse {datetime.datetime.now()}")
        df = self.init_players()

        ### Add game participants
        df = self.add_game_participants(df)

        ### Add events
        df = self.add_valid_games(df)

        ### Add injuries
        df = self.add_injuries(df)

        ## Add status
        df = self.transform_status(df)

        df = df[[
            'game_id',
            'player_id',
            'season',
            'week',
            'team',
            'high_pos_group',
            'position_group',
            'position',
            'jersey_number',
            'starter',
            'status',
            'report_status',
            'playerverse_status'
        ]]

        POSITION_GROUPS = [
            'd_line',
            'd_lb',
            'd_field',
            'o_line',
            'o_pass',
            'o_rush',
            'o_te',
            'NA',
            'quarterback',
            'special_teams',
        ]

        POSITION_GROUPS = ['quarterback']
        for pos in POSITION_GROUPS:
            for season in self.load_seasons:
                for week in range(1, (18 + 1 + 4 if season >= 2021 else 17 + 1 + 4)):
                    sub_verse = df[((df.season == season) & (df.week == week) & (df.position_group==pos))].copy()

                    ## Need to pull the game stats for all players for each week we want to make sure that we have the latest playerverse_status labeled correctly for each player

        return df

    def add_game_participants(self, df):
        """
        Merges roster, starter, and stats info for each player-week-team.
        Returns a DataFrame with participation flags and metadata.
        """
        espn_ids = df[['espn_id', 'player_id']].drop_duplicates()

        ### Rosters

        rosters_df = self.db['rosters'][[
            'player_id',
            'season',
            'week',
            'team',
            'high_pos_group',
            'position_group',
            'position',
            'depth_chart_position',
            'jersey_number',
            'status',
        ]]

        rosters_df = rosters_df.merge(espn_ids, how='left', on=['player_id'])
        starters_df = self.db['starters'].drop(columns=['game_id', 'game_type'])
        starters_df['espn_id'] = starters_df['espn_id'].astype(int)
        starters_df['espn_id'] = starters_df['espn_id'].astype(str)
        starters_df['played'] = starters_df['did_not_play'] == 0
        starters_df = starters_df.drop(columns=['did_not_play'])

        ## Will join starter in after we join roster + game participants
        rosters_df = rosters_df.merge(starters_df.drop(columns=['starter']), how='left', on=[
            'espn_id',
            'season',
            'week',
            'team'
        ]
                                      )
        rosters_df['rostered'] = True
        rosters_df = rosters_df.drop(columns=['espn_id'])

        jersey_numbers = rosters_df[['player_id', 'season', 'week', 'team', 'jersey_number', 'depth_chart_position']].drop_duplicates(['player_id', 'season', 'week', 'team', 'jersey_number', 'depth_chart_position', ]).copy()

        ### Game Stats

        stats_df = pd.DataFrame()
        for groups in ['player_stats', 'def_player_stats', 'kicking_player_stats']:
            s = self.db[groups][[
                'player_id',
                'season',
                'week',
                'team',
                'high_pos_group',
                'position_group',
                'position',
                'status',
            ]].copy()
            s['played'] = True
            s['rostered'] = True
            stats_df = pd.concat([stats_df, s], axis=0).reset_index(drop=True)
        stats_df = stats_df.merge(jersey_numbers, how='left', on=['player_id', 'season', 'week', 'team'])

        stats_df = pd.concat([stats_df, rosters_df], axis=0).drop_duplicates(['player_id', 'season', 'week', 'team', 'jersey_number', 'position_group'], keep='first').reset_index(drop=True)
        df = df.merge(stats_df, how='left', on=[
            'player_id',
            'season',
            'week'])

        ### Add depth charts

        depth_chart_df = self.db['depth_charts'][[
            'player_id',
            'season',
            'week',
            'team',
            'position',
            'depth_team',
            'depth_chart_position',
        ]].rename(columns={'position': 'depth_chart_position_raw'})

        df = df.merge(depth_chart_df, how='left', on=[
            'player_id',
            'season',
            'week',
            'team',
            'depth_chart_position',
        ])

        df = df.drop(columns=['depth_chart_position_raw'])

        ### Add starters

        qb_starters_df = self.team_events[[
            'season',
            'week',
            'team',
            'qb_id',
        ]].rename(columns={'qb_id': 'player_id'})
        qb_starters_df['starter'] = True

        starters_df = pd.concat([qb_starters_df, starters_df.drop(columns=['played', 'espn_id'])], axis=0).drop_duplicates(['player_id', 'season', 'week', 'team'], keep='first').reset_index(drop=True)

        df = df.merge(starters_df, how='left', on=[
            'player_id',
            'season',
            'week',
            'team'
        ])

        return df

    def add_valid_games(self, df):
        """
        Flags valid games for each player-week-team by merging with team events.
        Returns a DataFrame with is_valid_game column.
        """
        events_df = self.team_events[[
            'game_id',
            'season',
            'week',
            'team',
            'datetime'
        ]].rename(columns={'datetime': 'game_datetime'})
        events_df['is_valid_game'] = True
        df = df.merge(events_df, how='left', on=[
            'season',
            'week',
            'team'
        ])
        return df

    def add_injuries(self, df):
        """
        Merges injury info and flags pre-kickoff injury designations for each player-week-team.
        Returns a DataFrame with injury columns.
        """
        if self.db['injuries'].shape[0] == 0:
            df['injury_designation'] = False
            df['pre_kickoff_injury_designation'] = False
            df['report_status'] = None
            #df['injury_report_date_modified'] = None
            return df



        injury_df = self.db['injuries'][[
            'player_id',
            'season',
            'team',
            'week',
            'report_primary_injury',
            'report_secondary_injury',
            'report_status',
            'practice_primary_injury',
            'practice_secondary_injury',
            'practice_status',
            'date_modified',
        ]].copy().rename(columns={
            'date_modified': 'injury_report_date_modified',
        })
        injury_df['injury_designation'] = True

        df = df.merge(injury_df, how='left', on=[
            'player_id',
            'season',
            'team',
            'week',
        ])
        df['game_datetime'] = pd.to_datetime(df.game_datetime)
        df['injury_report_date_modified'] = pd.to_datetime(df.injury_report_date_modified)
        df['pre_kickoff_injury_designation'] = df['game_datetime'] > df['injury_report_date_modified']
        ## Drop for now until we setup injury type
        df = df.drop(columns=['injury_report_date_modified', 'practice_status', 'practice_primary_injury', 'practice_secondary_injury', 'report_status', 'report_primary_injury', 'report_secondary_injury'])

        ### Need to clean this up and make injury type but for now this will do

        return df

    def transform_status(self, df):
        """
        Sets a status for each player-week (PLAYED, INJURED, ROSTERED, FREE_AGENT, RETIRED, NO_GAME).
        Returns a DataFrame with playerverse_status column. Allows for retirement, coming back from retirement, status is always captured for players for every week
        """
        ## status (PLAYED, INJURED, ROSTERED, FREE_AGENT, RETIRED)

        ## Create default status
        df['playerverse_status'] = 'NONE'
        ## Create known statuses for overrides

        ## If there is no game_id associated with the player default them to NO_GAME (can be overrided by RETIRED, FREE_AGENT)
        df.loc[((df['playerverse_status'] == 'NONE') & (df['game_id'].isnull())), 'playerverse_status'] = 'NO_GAME'
        ## If the player has the status of RET then they are retired
        df.loc[((df['status'] == 'RET')), 'playerverse_status'] = 'RETIRED'
        ## If the player has the report_status of OUT then they are injured
        df.loc[(((df['report_status'] == 'Out') & (df['pre_kickoff_injury_designation'] == True))), 'playerverse_status'] = 'INJURED'
        ## If the player has stats then they played
        df.loc[((df['played'] == True)), 'playerverse_status'] = 'PLAYED'
        df.loc[((df['playerverse_status'] == 'NONE') & (df['status'].isin(['ACT','RES']) )), 'playerverse_status'] = 'ROSTERED' # leaving like this in case I want to split out status
        df.loc[((df['playerverse_status'] == 'NONE') & (df['rostered'] == True)), 'playerverse_status'] = 'ROSTERED'
        df.loc[((df['playerverse_status'] == 'NONE')), 'playerverse_status'] = 'FREE_AGENT'
        return df

    def add_event_qb_starters(self, df):
        """
        Flags QBs who started games using team events.
        Returns a DataFrame with starter flags for QBs.
        """

        starters_df = self.team_events[[
            'season',
            'week',
            'team',
            'qb_id',
        ]].rename(columns={'qb_id': 'player_id'})
        starters_df['events_qbs_played'] = True
        starters_df['events_qbs_starter'] = True
        df = df.merge(starters_df, how='left', on=[
            'player_id',
            'season',
            'week',
            'team'
        ]
        )
        # df['played'] = df['played'].fillna(df['events_qbs_played'])
        df = df.drop(columns=['events_qbs_played'])
        df['starter'] = df['starter'].fillna(df['events_qbs_starter'])
        df = df.drop(columns=['events_qbs_starter'])
        return df


if __name__ == '__main__':
    pdc = PlayerStateDataComponent(list(range(1999, 2025)))
    df = pdc.playerverse