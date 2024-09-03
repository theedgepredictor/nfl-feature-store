import numpy as np

NFLVERSEGITHUB = "https://github.com/nflverse/nflverse-data/releases/download/"
NFL_PLAYER_URL = f"{NFLVERSEGITHUB}players/players.parquet"
NFL_PLAYER_STATS_URL = f"{NFLVERSEGITHUB}player_stats/player_stats.parquet"
NFL_PBP = NFLVERSEGITHUB + "pbp/play_by_play_{season}.parquet"

player_cols_raw = [
    'display_name',
    'gsis_id',
    #'first_name',
    #'last_name',
    #'esb_id',
    'status',
    'birth_date',
    'college_name',
    'position_group',
    'position',
    #'jersey_number',
    'height',
    'weight',
    #'years_of_experience',
    #'team_abbr',
    #'team_seq',
    #'current_team_id',
    #'football_name',
    'entry_year',
    #'rookie_year',
    #'draft_club',
    'draft_number',
    'college_conference',
    #'status_description_abbr',
    #'status_short_description',
    #'gsis_it_id',
    #'short_name',
    #'smart_id',
    'headshot',
    #'suffix',
    #'uniform_number',
    #'draft_round',
    #'season'
]

PLAYER_BOXSCORE_COLUMNS = [
    'player_id',
    'season',
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
    'carries',
    'rushing_yards',
    'rushing_tds',
    'rushing_fumbles',
    'rushing_fumbles_lost',
    'rushing_first_downs',
    'rushing_epa',
    'rushing_2pt_conversions',
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
    'special_teams_tds',
    'fantasy_points',
    'fantasy_points_ppr'
]

PLAYER_POINTS_COLUMNS = [
    'player_id',
    'season',
    'fantasy_points',
    'fantasy_points_ppr'
]

DATA_TYPES = [
    "Integer",
    "Float",
    "String",
    "Timestamp",
    "Boolean"
]

FEATURE_STORE_METADATA = [
    {"name": "completions", "friendly_name": "Completions", "description": "The number of completed passes.", "dtype": "Int32", "type_group": "Passing", "selectable": False},
    {"name": "attempts", "friendly_name": "Attempts", "description": "The number of pass attempts as defined by the NFL.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "passing_yards", "friendly_name": "Passing Yards", "description": "Yards gained on pass plays.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "passing_tds", "friendly_name": "Passing Touchdowns", "description": "The number of passing touchdowns.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "interceptions", "friendly_name": "Interceptions", "description": "The number of interceptions thrown.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "sacks", "friendly_name": "Sacks", "description": "The Number of times sacked.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "sack_yards", "friendly_name": "Sack Yards", "description": "Yards lost on sack plays.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "sack_fumbles", "friendly_name": "Sack Fumbles", "description": "The number of sacks with a fumble.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "sack_fumbles_lost", "friendly_name": "Sack Fumbles Lost", "description": "The number of sacks with a lost fumble.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "passing_air_yards", "friendly_name": "Passing Air Yards", "description": "Passing air yards (includes incomplete passes).", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "passing_yards_after_catch", "friendly_name": "Passing Yards After Catch", "description": "Yards after the catch gained on plays in which player was the passer (this is an unofficial stat and may differ slightly between different sources).", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "passing_first_downs", "friendly_name": "Passing First Downs", "description": "First downs on pass attempts.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "passing_epa", "friendly_name": "Passing EPA", "description": "Total expected points added on pass attempts and sacks. NOTE: this uses the variable qb_epa, which gives QB credit for EPA for up to the point where a receiver lost a fumble after a completed catch and makes EPA work more like passing yards on plays with fumbles.", "dtype": "Float32", "type_group": "Passing", "selectable": True},
    {"name": "passing_2pt_conversions", "friendly_name": "Passing 2pt Conversions", "description": "Two-point conversion passes.", "dtype": "Int32", "type_group": "Passing", "selectable": True},
    {"name": "pacr", "friendly_name": "PACR", "description": "Passing Air Conversion Ratio. PACR = passing_yards / passing_air_yards", "dtype": "Float32", "type_group": "Passing", "selectable": True},
    {"name": "dakota", "friendly_name": "Dakota", "description": "Dakota rating for passing efficiency.", "dtype": "Float32", "type_group": "Passing", "selectable": True},

    {"name": "carries", "friendly_name": "Carries", "description": "The number of official rush attempts (incl. scrambles and kneel downs). Rushes after a lateral reception don't count as carry.", "dtype": "Int32", "type_group": "Rushing", "selectable": True},
    {"name": "rushing_yards", "friendly_name": "Rushing Yards", "description": "Yards gained when rushing with the ball (incl. scrambles and kneel downs). Also includes yards gained after obtaining a lateral on a play that started with a rushing attempt.", "dtype": "Int32", "type_group": "Rushing", "selectable": True},
    {"name": "rushing_tds", "friendly_name": "Rushing Touchdowns", "description": "The number of rushing touchdowns (incl. scrambles). Also includes touchdowns after obtaining a lateral on a play that started with a rushing attempt.", "dtype": "Int32", "type_group": "Rushing", "selectable": True},
    {"name": "rushing_fumbles", "friendly_name": "Rushing Fumbles", "description": "The number of rushes with a fumble.", "dtype": "Int32", "type_group": "Rushing", "selectable": True},
    {"name": "rushing_fumbles_lost", "friendly_name": "Rushing Fumbles Lost", "description": "The number of rushes with a lost fumble.", "dtype": "Int32", "type_group": "Rushing", "selectable": True},
    {"name": "rushing_first_downs", "friendly_name": "Rushing First Downs", "description": "First downs on rush attempts (incl. scrambles).", "dtype": "Int32", "type_group": "Rushing", "selectable": True},
    {"name": "rushing_epa", "friendly_name": "Rushing EPA", "description": "Expected points added on rush attempts (incl. scrambles and kneel downs).", "dtype": "Float32", "type_group": "Rushing", "selectable": True},
    {"name": "rushing_2pt_conversions", "friendly_name": "Rushing 2pt Conversions", "description": "Two-point conversion rushes.", "dtype": "Int32", "type_group": "Rushing", "selectable": True},

    {"name": "receptions", "friendly_name": "Receptions", "description": "The number of pass receptions. Lateral receptions officially don't count as reception.", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "targets", "friendly_name": "Targets", "description": "The number of pass plays where the player was the targeted receiver.", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_yards", "friendly_name": "Receiving Yards", "description": "Yards gained after a pass reception. Includes yards gained after receiving a lateral on a play that started as a pass play.", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_tds", "friendly_name": "Receiving Touchdowns", "description": "The number of touchdowns following a pass reception. Also includes touchdowns after receiving a lateral on a play that started as a pass play.", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_air_yards", "friendly_name": "Receiving Air Yards", "description": "Receiving air yards (incl. incomplete passes).", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_yards_after_catch", "friendly_name": "Receiving Yards After Catch", "description": "Yards after the catch gained on plays in which player was receiver (this is an unofficial stat and may differ slightly between different sources).", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_first_downs", "friendly_name": "Receiving First Downs", "description": "First downs by reception.", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_epa", "friendly_name": "Receiving EPA", "description": "The expected points added by receiving.", "dtype": "Float32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_fumbles", "friendly_name": "Receiving Fumbles", "description": "The number of fumbles after a pass reception.", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_fumbles_lost", "friendly_name": "Receiving Fumbles Lost", "description": "The number of fumbles lost after a pass reception.", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "receiving_2pt_conversions", "friendly_name": "Receiving 2pt Conversions", "description": "Two-point conversion receptions.", "dtype": "Int32", "type_group": "Receiving", "selectable": True},
    {"name": "racr", "friendly_name": "RACR", "description": "Receiver Air Conversion Ratio. RACR = receiving_yards / receiving_air_yards.", "dtype": "Float32", "type_group": "Receiving", "selectable": True},
    {"name": "target_share", "friendly_name": "Target Share", "description": "The share of targets of the player in all targets of his team.", "dtype": "Float32", "type_group": "Receiving", "selectable": True},
    {"name": "air_yards_share", "friendly_name": "Air Yards Share", "description": "The share of receiving_air_yards of the player in all air_yards of his team.", "dtype": "Float32", "type_group": "Receiving", "selectable": True},
    {"name": "wopr", "friendly_name": "WOPR", "description": "Weighted Opportunity Rating. WOPR = 1.5 × target_share + 0.7 × air_yards_share.", "dtype": "Float32", "type_group": "Receiving", "selectable": True},

    {"name": "special_teams_tds", "friendly_name": "Special Teams Touchdowns", "description": "The number of touchdowns scored on special teams.", "dtype": "Int32", "type_group": "SpecialTeams", "selectable": True},

    {"name": "player_id", "friendly_name": "Player ID", "description": "ID of the player. Use this to join to other sources.", "dtype": "string", "type_group": "Identifier", "selectable": False},
    {"name": "display_name", "friendly_name": "Display Name", "description": "The display name of the player.", "dtype": "string", "type_group": "Identifier", "selectable": False},
    {"name": "status", "friendly_name": "Status", "description": "Player Status (ACT, RET, CUT)", "dtype": "string", "type_group": "Identifier", "selectable": False},
    {"name": "position", "friendly_name": "Position", "description": "Position of the player.", "dtype": "string", "type_group": "Identifier", "selectable": False},
    {"name": "position_group", "friendly_name": "Position Group", "description": "High Level Position Group of the player.", "dtype": "string", "type_group": "Identifier", "selectable": False},

    {"name": "season", "friendly_name": "Season", "description": "The season year.", "dtype": "Int32", "type_group": "Identifier", "selectable": True},
    {"name":"games_played", "friendly_name": "Games Played", "description": "Games the player played in that season", "dtype": "Int32", "type_group": "Identifier", "selectable": False},
    {"name": "entry_year", "friendly_name": "Entry Year", "description": "The year the player entered the league.", "dtype": "Int32", "type_group": "Identifier", "selectable": True},
    {"name": "birth_date", "friendly_name": "Birth Date", "description": "The birth date of the player.", "dtype": "string", "type_group": "Identifier", "selectable": False},
    {"name": "draft_number", "friendly_name": "Draft Number", "description": "The draft number of the player.", "dtype": "Int32", "type_group": "Identifier", "selectable": True},
    {"name": "college_name", "friendly_name": "College Name", "description": "The college the player attended.", "dtype": "string", "type_group": "Identifier", "selectable": False},
    {"name": "college_conference", "friendly_name": "College Conference", "description": "The conference the college is a part of", "dtype": "string", "type_group": "Identifier", "selectable": False},
{"name": "headshot", "friendly_name": "Headshot", "description": "The URL to the player's headshot photo", "dtype": "string", "type_group": "Identifier", "selectable": False},
    {"name": "years_of_experience", "friendly_name": "Years of Experience", "description": "Number of years in the league", "dtype": "Int32", "type_group": "Identifier", "selectable": False},

    {"name": "height", "friendly_name": "Height", "description": "The height of the player.", "dtype": "Float32", "type_group": "Attribute", "selectable": True},
    {"name": "weight", "friendly_name": "Weight", "description": "The weight of the player.", "dtype": "Float32", "type_group": "Attribute", "selectable": True},

    {"name": "fantasy_points", "friendly_name": "Fantasy Points", "description": "Standard fantasy points.", "dtype": "Float32", "type_group": "Target", "selectable": False},
    {"name": "fantasy_points_ppr", "friendly_name": "Fantasy Points PPR", "description": "PPR fantasy points.", "dtype": "Float32", "type_group": "Target", "selectable": False},
    {"name": "position_rank", "friendly_name": "Position Rank", "description": "Fantasy Rank by Position Group", "dtype": "Int32", "type_group": "Target", "selectable": False},
    {"name": "ppr_position_rank", "friendly_name": "PPR Position Rank", "description": "PPR Fantasy Rank by Position Group", "dtype": "Int32", "type_group": "Target", "selectable": False},
]

EXPERIMENT_CLASSES = ['fantasy_football','event_quarters']
YEARS = [2022, 2023, 2024]
POSITIONS = ['QB', 'RB', 'TE', 'WR']
TARGET = ['fantasy_points']