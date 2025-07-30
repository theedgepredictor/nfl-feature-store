import datetime
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from src.components.player_state import PlayerStateDataComponent
from src.extracts.events import load_exavg_event_feature_store
from src.extracts.player_stats import get_player_regular_season_game_fs
from src.transforms.player import get_preseason_players, impute_base_player_ratings, adjust_preseason_ratings



@dataclass
class PlayerState:
    # Core identifiers
    player_id: str
    game_id: int
    season: int
    week: int
    
    # Team and position info
    team: str
    high_pos_group: str
    position_group: str
    position: str
    
    # Status fields
    starter: bool
    status: str
    report_status: str
    playerverse_status: str
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return f"Player({self.player_id}, {self.position_group}, {self.team}, Week {self.week})"

@dataclass
class PreSeasonPlayer(PlayerState):
    # Madden identifiers
    madden_id: str
    years_exp: int
    is_rookie: bool
    last_season_av: int
    
    # Overall rating
    overallrating: int
    
    # Physical attributes
    agility: int
    acceleration: int
    speed: int
    stamina: int
    strength: int
    toughness: int
    injury: int
    awareness: int
    jumping: int
    
    # Position-specific ratings
    # Rushing/Ball Carrying
    trucking: int = 0
    carrying: int = 0
    ballcarriervision: int = 0
    stiffarm: int = 0
    spinmove: int = 0
    jukemove: int = 0
    
    # Passing
    throwpower: int = 0
    throwaccuracyshort: int = 0
    throwaccuracymid: int = 0
    throwaccuracydeep: int = 0
    playaction: int = 0
    throwonrun: int = 0
    
    # Receiving
    catching: int = 0
    shortrouterunning: int = 0
    midrouterunning: int = 0
    deeprouterunning: int = 0
    spectacularcatch: int = 0
    catchintraffic: int = 0
    release: int = 0
    
    # Blocking
    runblocking: int = 0
    passblocking: int = 0

@dataclass
class PlayerRatingMatrix:
    player_state: PlayerState
    metrics: Dict[str, float] = field(default_factory=dict)
    kpi_weights: Dict[str, float] = field(default_factory=dict)
    
    def calculate_rating(self) -> float:
        rating = 0.0
        for metric, value in self.metrics.items():
            if metric in self.kpi_weights:
                rating += value * self.kpi_weights[metric]
        return rating

@dataclass
class QuarterbackPlayerRatingMatrix(PlayerRatingMatrix):
    def __init__(self, player_state: PlayerState, metrics: Dict[str, float]):
        super().__init__(player_state, metrics)
        self.kpi_weights = {
            # Efficiency Metrics (40% weight)
            'completion_percentage': 0.15,
            'yards_per_pass_attempt': 0.15,
            'passer_rating': 0.10,
            'VALUE_ELO': 0.10,
            'dakota': 0.10,

            # Production Metrics (30% weight)
            'passing_epa': 0.15,
            'passing_yards': 0.10,
            'passing_tds': 0.10,
            'passing_first_downs': 0.05,

            # Decision Making (30% weight)
            'touchdown_per_play': 0.15,
            'interceptions': -0.10,
            'sack_rate': -0.05
        }

class PlayerRatingComponent:
    """
    Calculates weekly and overall ratings for players (PoC: QBs) using normalized stats, rolling averages, and regression to mean.
    Integrates Madden ratings and performance metrics.
    """
    def __init__(self, load_seasons, season_type=None):
        """
        Initializes the PlayerRatingComponent with seasons and season type.
        Loads data and runs the rating pipeline.
        """
        self.load_seasons = load_seasons
        self.season_type = season_type
        self.db = self.extract()
        self.df = self.run_pipeline()
        self.latest_player_ratings = None
        self.player_ratings = None
        self.team_ratings = None

    def extract(self):
        """
        ### Input Data
        1. **Static Players**
        ```python
        'static_players': player_state_component.players
        ```
        - Contains unchanging player metadata
        - Used for joining against rating system
        - Provides player names and additional visualization data
        - Fields: player_id, name, birth_date, college, draft info, etc.

        2. **Player States**
        ```python
        'player_states': player_states
        ```
        - Weekly player state information
        - Contains merged weekly player data
        - Tracks player STATUS (PLAYED, INJURED, etc.)
        - Used to determine player availability and game participation

        3. **Offensive Weekly Player Stats**
        ```python
        'off_weekly_player_stats': off_players
        ```
        - Weekly offensive performance metrics
        - Contains game-level statistics
        - Used for rating calculations
        - Stats include: passing_yards, passing_tds, interceptions, completion_pct

        4. **Team Ratings**
        ```python
        'team_ratings': team_ratings
        ```
        - Team-level performance metrics
        - Used for contextualizing player performances
        - Loaded from event feature store

        5. **Preseason Players**
        ```python
        'preseason_players': preseason_players
        ```
        - Initial player ratings before season starts
        - Used for week 1 rating calculations
        - Includes Madden ratings and other baseline metrics
        """

        print(f"    Loading player data {datetime.datetime.now()}")
        player_state_component = PlayerStateDataComponent(self.load_seasons, season_type=self.season_type)
        player_states = player_state_component.run_pipeline()

        print(f"    Loading offensive aggregated player stat data {datetime.datetime.now()}")
        off_players = pd.concat([get_player_regular_season_game_fs(season, group='off') for season in self.load_seasons])

        print(f"    Loading event feature store {datetime.datetime.now()}")
        _,team_ratings = load_exavg_event_feature_store(self.load_seasons)

        print(f"    Loading preseason player rating data {datetime.datetime.now()}")
        preseason_players = pd.concat([get_preseason_players(season) for season in self.load_seasons])


        return {
            'static_players': player_state_component.players,
            'player_states': player_states,
            'off_weekly_player_stats': off_players,
            'team_ratings': team_ratings,
            'preseason_players': preseason_players,
        }

    def init_ratings(self):
        """
        Initialize player ratings by creating PlayerState and PreSeasonPlayer objects,
        then setting up the rating matrices for each position group.
        """
        init_season = 2003

        # Create player states from the player states DataFrame
        player_states = []
        players_df = self.db['player_states']
        
        for _, row in players_df.iterrows():
            player_state = PlayerState(
                player_id=row['player_id'],
                game_id=row['game_id'],
                season=row['season'],
                week=row['week'],
                team=row['team'],
                high_pos_group=row['high_pos_group'],
                position_group=row['position_group'],
                position=row['position'],
                starter=row['starter'],
                status=row['status'],
                report_status=row.get('report_status', ''),
                playerverse_status=row.get('playerverse_status', '')
            )
            player_states.append(player_state)

        # Create preseason players from the preseason DataFrame
        preseason_players = []
        preseason_df = self.db['preseason_players']
        
        for _, row in preseason_df.iterrows():
            # Map state data from player states
            matching_state = next(
                (ps for ps in player_states 
                 if ps.player_id == row['player_id'] 
                 and ps.season == row['season']), None)
            
            if matching_state:
                preseason_player = PreSeasonPlayer(
                    # Inherit from PlayerState
                    player_id=matching_state.player_id,
                    game_id=matching_state.game_id,
                    season=matching_state.season,
                    week=1,  # Preseason players are always week 1
                    team=matching_state.team,
                    high_pos_group=matching_state.high_pos_group,
                    position_group=matching_state.position_group,
                    position=matching_state.position,
                    starter=matching_state.starter,
                    status=matching_state.status,
                    report_status=matching_state.report_status,
                    playerverse_status=matching_state.playerverse_status,
                    
                    # PreSeasonPlayer specific fields
                    madden_id=row.get('madden_id', ''),
                    years_exp=row['years_exp'],
                    is_rookie=row['is_rookie'],
                    last_season_av=row.get('last_season_av', 0),
                    overallrating=row['overallrating'],
                    
                    # Physical attributes
                    agility=row.get('agility', 0),
                    acceleration=row.get('acceleration', 0),
                    speed=row.get('speed', 0),
                    stamina=row.get('stamina', 0),
                    strength=row.get('strength', 0),
                    toughness=row.get('toughness', 0),
                    injury=row.get('injury', 0),
                    awareness=row.get('awareness', 0),
                    jumping=row.get('jumping', 0),
                    
                    # Position-specific ratings
                    trucking=row.get('trucking', 0),
                    carrying=row.get('carrying', 0),
                    ballcarriervision=row.get('ballcarriervision', 0),
                    stiffarm=row.get('stiffarm', 0),
                    spinmove=row.get('spinmove', 0),
                    jukemove=row.get('jukemove', 0),
                    throwpower=row.get('throwpower', 0),
                    throwaccuracyshort=row.get('throwaccuracyshort', 0),
                    throwaccuracymid=row.get('throwaccuracymid', 0),
                    throwaccuracydeep=row.get('throwaccuracydeep', 0),
                    playaction=row.get('playaction', 0),
                    throwonrun=row.get('throwonrun', 0),
                    catching=row.get('catching', 0),
                    shortrouterunning=row.get('shortrouterunning', 0),
                    midrouterunning=row.get('midrouterunning', 0),
                    deeprouterunning=row.get('deeprouterunning', 0),
                    spectacularcatch=row.get('spectacularcatch', 0),
                    catchintraffic=row.get('catchintraffic', 0),
                    release=row.get('release', 0),
                    runblocking=row.get('runblocking', 0),
                    passblocking=row.get('passblocking', 0)
                )
                preseason_players.append(preseason_player)

        # Store the initialized objects
        self.player_states = player_states
        self.preseason_players = preseason_players
        
        # Create player rating matrices
        self.player_rating_matrices = []
        off_weekly_stats = self.db['off_weekly_player_stats']
        
        for player_state in player_states:
            # Get player's stats from off_weekly_stats
            player_stats = off_weekly_stats[
                (off_weekly_stats['player_id'] == player_state.player_id) &
                (off_weekly_stats['season'] == player_state.season) &
                (off_weekly_stats['week'] == player_state.week)
            ].iloc[0] if len(off_weekly_stats) > 0 else {}
            
            # Convert stats to metrics dictionary
            metrics = {col: player_stats.get(col, 0.0) for col in player_stats.index}
            
            # Create appropriate rating matrix based on position group
            if player_state.position == "QB":
                matrix = QuarterbackPlayerRatingMatrix(player_state, metrics)
            else:
                matrix = PlayerRatingMatrix(player_state, metrics)
            
            self.player_rating_matrices.append(matrix)
        
        return self.player_rating_matrices



    def run_pipeline(self):
        """
        Main pipeline to process and calculate ratings for each season and week.
        Returns a DataFrame of player ratings.
        """
        ratings_df = self.init_ratings()

        ### Generate Averages

        ### Ge

        ### Create ratings for each season
        ratings_df = pd.DataFrame()
        for season in self.load_seasons:
            if 'season' in ratings_df.columns and ratings_df.shape[0] != 0:
                previous_rating_df = ratings_df[ratings_df.season < season].copy()
            else:
                previous_rating_df = pd.DataFrame()
            season_ratings_df = self.player_rating_season_pipeline(season, previous_rating_df)
            ratings_df = pd.concat([ratings_df, season_ratings_df], ignore_index=True)



    def player_rating_season_pipeline(self, season, previous_rating_df):
        """
        Calculates ratings for all players in a season, using previous ratings for regression and rolling averages.
        Returns a DataFrame of season ratings.
        """
        if previous_rating_df.shape[0] != 0:
            p = previous_rating_df.sort_values(['player_id', 'season', 'week']).drop_duplicates(['player_id'], keep='last').rename(columns={
                    'season': 'last_rating_season',
                    'week': 'last_rating_week',
                }).copy()
        else:
            p = previous_rating_df
        season_ratings_df = []
        events_df = self.db['players'].drop_duplicates(['game_id'],keep='first')
        weeks = events_df[events_df.season==season].copy().week.unique()
        for week in list(weeks):
            p = self._weekly_player_pipeline(season, week, p)


    def _weekly_player_pipeline(self, season, week, previous_rating_df, position_group='quarterback'):
        """
        Calculates weekly ratings for QBs using normalized stats and combines them into a weighted rating.
        Applies regression to mean for preseason and rolling average for subsequent weeks.
        Returns a DataFrame of weekly QB ratings.
        """
        df = self.db['players']
        player_df = self.db['off_players']
        weekly_player_df = self.db['preseason_players'] if week == 1 else previous_rating_df

        df = df[((df['season'] == season) & (df.week == week))].copy()
        player_df = player_df[(
            (player_df['season'] < season-1) | ((player_df['season'] == season) & (player_df.week == week))
        )].drop(columns=['position_group']).sort_values(['player_id', 'season', 'week']).drop_duplicates(['player_id'], keep='last').rename(columns={
            'season': 'last_stat_season',
            'week': 'last_stat_week',
        }).copy()

        weekly_player_df = pd.merge(weekly_player_df, df, on=['player_id', 'season'], how='left')
        weekly_player_df = weekly_player_df[weekly_player_df.position_group == position_group].copy()
        weekly_player_df = pd.merge(weekly_player_df, player_df, on=['player_id'], how='left')

        weekly_player_df['birth_date'] = pd.to_datetime(weekly_player_df.birth_date)
        weekly_player_df['birth_date'] = weekly_player_df['birth_date'] + pd.Timedelta(hours=4)
        weekly_player_df.birth_date = pd.to_datetime(weekly_player_df.birth_date, utc=True)
        weekly_player_df['age'] = (weekly_player_df['datetime'] - weekly_player_df['birth_date']).dt.days / 365

        # Clean up columns
        weekly_player_df = weekly_player_df.drop(columns=[
            'common_first_name', 
            'first_name', 
            'last_name', 
            'short_name', 
            'football_name', 
            'suffix', 
            'esb_id', 
            'nfl_id', 
            'pff_id', 
            'otc_id', 
            'espn_id', 
            'smart_id',
            'headshot', 
            'college_name', 
            'college_conference', 
            'rookie_season', 
            'draft_team'
        ], errors='ignore')
        weekly_player_df = weekly_player_df.drop_duplicates(['player_id', 'season'])

        # --- QB Weekly Rating Calculation ---
        # Only keep QBs who played or were starters
        qbs = weekly_player_df.copy()
        # Example stat columns: passing_yards, passing_tds, interceptions, completion_pct
        stat_cols = ['passing_yards', 'passing_tds', 'interceptions', 'completion_pct']
        # Normalize stats (z-score)
        for col in stat_cols:
            if col in qbs.columns:
                mean = qbs[col].mean()
                std = qbs[col].std()
                if std > 0:
                    qbs[f'{col}_norm'] = (qbs[col] - mean) / std
                else:
                    qbs[f'{col}_norm'] = 0
            else:
                qbs[f'{col}_norm'] = 0

        # Weighted sum for weekly rating
        qbs['weekly_rating'] = (
            qbs['passing_yards_norm'] * 0.3 +
            qbs['passing_tds_norm'] * 0.4 -
            qbs['interceptions_norm'] * 0.2 +
            qbs['completion_pct_norm'] * 0.1
        )

        # Update overall rating (rolling average or regression)
        if week == 1:
            # Apply regression to mean for preseason
            def rating_regression(preseason_players_df, previous_rating_df):
                if previous_rating_df.shape[0] == 0:
                    return preseason_players_df
                # Simple regression: new_rating = mean + (old_rating - mean) * 2/3
                mean_rating = previous_rating_df['overallrating'].mean() if 'overallrating' in previous_rating_df.columns else 70
                preseason_players_df['overallrating'] = mean_rating + (preseason_players_df['overallrating'] - mean_rating) * (2/3)
                return preseason_players_df
            qbs = rating_regression(qbs, previous_rating_df)
        else:
            # Rolling average: combine previous and current week
            if previous_rating_df is not None and 'overallrating' in previous_rating_df.columns:
                prev = previous_rating_df[['player_id', 'overallrating']].set_index('player_id')
                qbs['overallrating'] = qbs.apply(
                    lambda row: (row['weekly_rating'] + prev.loc[row['player_id'], 'overallrating']) / 2 if row['player_id'] in prev.index else row['weekly_rating'],
                    axis=1
                )
            else:
                qbs['overallrating'] = qbs['weekly_rating']

        # Impute Missing Base Ratings
        qbs = impute_base_player_ratings(qbs)
        qbs = qbs.sort_values(['overallrating'], ascending=False)

        # Adjust preseason ratings if needed
        if week == 1:
            qbs = adjust_preseason_ratings(qbs)

        return qbs






if __name__ == '__main__':
    player_rating_component = PlayerRatingComponent([2003,2004], season_type='REG')
    df = player_rating_component.run_pipeline()