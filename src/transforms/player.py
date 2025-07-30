import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.consts import POSITION_MAPPER, HIGH_POSITION_MAPPER, IMPUTE_FEATURES_DICT
from src.extracts.player_stats import collect_players, collect_combine, collect_roster, get_approximate_value, get_madden_ratings, collect_weekly_espn_player_stats
from src.formatters.general import df_rename_fold
from src.formatters.reformat_team_name import team_id_repl
from src.transforms.averages import dynamic_window_rolling_average
from src.transforms.player_groups.qbs import make_qb_career

MADDEN_FEATURES = [
    'overallrating',
    # Pace
    'agility',
    'acceleration',
    'speed',
    'stamina',
    # Strength / Fitness
    'strength',
    'toughness',
    'injury',
    'awareness',
    'jumping',
    'trucking',
    # Passing
    'throwpower',
    'throwaccuracyshort',
    'throwaccuracymid',
    'throwaccuracydeep',
    'playaction',
    'throwonrun',
    # Rushing
    'carrying',
    'ballcarriervision',
    'stiffarm',
    'spinmove',
    'jukemove',
    # Receiving
    'catching',
    'shortrouterunning',
    'midrouterunning',
    'deeprouterunning',
    'spectacularcatch',
    'catchintraffic',
    'release',
    # Blocking
    'runblocking',
    'passblocking',
    'impactblocking',
    # Coverage / Defense
    'mancoverage',
    'zonecoverage',
    'tackle',
    'hitpower',
    'press',
    'pursuit',
    # Special Teams
    'kickaccuracy',
    'kickpower',
    'return',
]

def get_static_players():
    """
    Pull from player and combine data
    """
    df = collect_players()

    ### Combine Extractor (First Come)
    combine_df = collect_combine()
    #combine_df = combine_df[combine_df.position_group == position_group].copy()
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
        'common_first_name',
        'first_name',
        'last_name',
        'short_name',
        'football_name',
        'suffix',
        'esb_id',
        'nfl_id',
        'pfr_id',
        'pff_id',
        'otc_id',
        'espn_id',
        'smart_id',
        'birth_date',
        #'high_pos_group',
        #'position_group',
        #'position',
        'height',
        'weight',
        'headshot',
        'college_name',
        'college_conference',
        #'jersey_number',
        'rookie_season',
        #'last_season',
        #'latest_team',
        #'status',
        #'status_abbr',
        #'ngs_status',
        #'ngs_status_short_description',
        #'years_of_experience',
        #'pff_position',
        #'pff_status',
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
    df['last_updated'] = pd.to_datetime('now')
    return df

def apply_rookie_av(df):
    if df['draft_pick'] == 1:
        df['last_season_av'] = 12
    elif df['draft_pick'] == 2:
        df['last_season_av'] = 11
    elif df['draft_pick'] == 3:
        df['last_season_av'] = 10.5
    elif df['draft_pick'] == 4:
        df['last_season_av'] = 9
    elif df['draft_pick'] == 5:
        df['last_season_av'] = 8.5
    else:
        df['last_season_av'] = (9 - df['draft_round']) * 0.5
    return df

## Preseason Player
### Has Preseason Player Data Ratings
def get_preseason_players(season):
    """
    Using info from the previous season create a preseason player. Rating information to be used
    as a base rating for the upcoming season
    :return:
    """
    df = collect_roster(season)
    df = df[df.week == 1].copy()
    df = pd.merge(df, get_static_players(), on='player_id', how='left')
    df['is_rookie'] = (df['rookie_season'] == season) & (df.years_exp == 0)

    ### AV Extractor (Previous Season)
    av_df = get_approximate_value(season - 1)[[
        'player_id',
        'approximate_value'
    ]].rename(columns={'player_id': 'pfr_id', 'approximate_value': 'last_season_av'})
    df = pd.merge(df, av_df, on='pfr_id', how='left')

    processed_madden_df = pd.concat([
        get_madden_ratings(season),
        get_madden_ratings(season-1),
    ]).drop_duplicates(subset=['player_id'], keep=('first' if season != 2002 else 'last'))

    processed_madden_df['season'] = season
    df = pd.merge(df, processed_madden_df.drop(columns=['position_group']), on=['player_id','season'], how='left')

    rookie_approx_value_df = df[df['is_rookie']==True].copy()
    rookie_approx_value_df.draft_round = rookie_approx_value_df.draft_round.fillna(8)
    rookie_approx_value_df.draft_pick = rookie_approx_value_df.draft_pick.fillna(rookie_approx_value_df.draft_pick.max() + 1)
    rookie_approx_value_df = rookie_approx_value_df.apply(apply_rookie_av, axis=1)

    df = df[df['is_rookie']==False].copy()
    df = pd.concat([df, rookie_approx_value_df], ignore_index=True)

    df = df[df.position_group == 'quarterback'].copy()
    df = df.drop_duplicates(subset=['player_id'], keep='first').drop(columns=[
        'team',
        'week',
        'position',
        'jersey_number',
        'status_abbr',
        'position_group',
        'high_pos_group',
    ])
    df = df[[
        'season',
        'player_id',
        'madden_id',
        'years_exp',
         'is_rookie', 'last_season_av',  'overallrating', 'agility',
         'acceleration', 'speed', 'stamina', 'strength', 'toughness', 'injury',
         'awareness', 'jumping', 'trucking', 'throwpower', 'throwaccuracyshort',
         'throwaccuracymid', 'throwaccuracydeep', 'playaction', 'throwonrun',
         'carrying', 'ballcarriervision', 'stiffarm', 'spinmove', 'jukemove',
         'catching', 'shortrouterunning', 'midrouterunning', 'deeprouterunning',
         'spectacularcatch', 'catchintraffic', 'release', 'runblocking',
         'passblocking', 'impactblocking', 'mancoverage', 'zonecoverage',
         'tackle', 'hitpower', 'press', 'pursuit', 'kickaccuracy', 'kickpower',
         'return']]
    return df

def make_player_avg_group_features(data, group_features_dict, mode='season_avg'):
    """
    Calculate dynamic window avg for multiple attributes (like epa, rushing_yards, etc.) for both offense and defense.

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data. (Filter data prior to calling this function)
        group_features: List of attributes to calculate dynamic window avg

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive avg values for each attribute.
    """

    features = pd.DataFrame()

    for attr, agg_method in group_features_dict.items():

        # Separate attribute values
        offense = data.groupby(['player_id','season','week'], as_index=False).agg({attr: agg_method})

        # Lag attribute one period back
        offense[f'{attr}_shifted'] = offense.groupby('player_id')[attr].shift()

        # Calculate dynamic window MA for the attribute
        offense[f'{mode}_{attr}'] = offense.groupby('player_id').apply(dynamic_window_rolling_average, attr, mode).values
        offense = offense[[f'{mode}_{attr}'] + ['player_id','season','week']]
        # Collect features for this attribute
        if features.shape[0] == 0:
            features = offense
        else:
            features = pd.merge(features, offense, on=['player_id','season','week'])

    return features.drop_duplicates(subset=['player_id','season','week'])

def impute_base_player_ratings(df):
    general_features = [
        'forty',
        'bench',
        'vertical',
        'broad_jump',
        'cone',
        'shuttle',
        'last_season_av',
    ]

    general_helper_features = [
        'height',
        'weight',
        'age',
        'years_exp',
        'draft_year',
        'draft_pick',
        'is_rookie'
    ]

    dfs = []
    #for pos_group in list(HIGH_POSITION_MAPPER.keys()):
    for pos_group in ['quarterback']:
        pos_group_features = []
        '''
        for feature in IMPUTE_FEATURES_DICT[pos_group]:
            pos_group_features.append(f"season_avg_{feature}")
            pos_group_features.append(f"season_total_{feature}")
            pos_group_features.append(f"form_{feature}")
        '''
        allowed_impute_cols = general_features+MADDEN_FEATURES+pos_group_features
        group_df = df[df['position_group']==pos_group].copy()

        impute_df = group_df[allowed_impute_cols + general_helper_features]


        cols_with_missing = ((impute_df[allowed_impute_cols].isnull().sum() > 0) & (impute_df[allowed_impute_cols].isnull().sum() != df.shape[0])).reset_index().rename(columns={0: 'missing', 'index': 'col'})
        cols_with_missing = list(cols_with_missing[cols_with_missing['missing'] == True].col.values)

        group_df = group_df.drop(columns=cols_with_missing)

        impute_df = impute_df[cols_with_missing+ general_helper_features].reset_index(drop=True)
        impute_df = impute_df.astype(float)

        imputer = IterativeImputer()
        out = imputer.fit_transform(impute_df)
        dfs.append(pd.concat([group_df.reset_index().drop(columns='index'), pd.DataFrame(data=out, columns=cols_with_missing+general_helper_features).drop(columns=general_helper_features)], axis=1))
    df = pd.concat(dfs).reset_index().drop(columns='index')
    return df

def _simple_adjust_preseason_ratings(df):
    dfs = []
    #for pos_group in list(HIGH_POSITION_MAPPER.keys()):
    for pos_group in ['quarterback']:

        allowed_adjust_preseason_cols = ['last_season_av']
        group_df = df[df['position_group'] == pos_group].copy()

        rookies_df = group_df[group_df['is_rookie'] == True].copy()
        rookies_df['madden_adjustment'] = 0
        unfit_for_adjustment_df = group_df[((group_df['is_rookie'] == False) & (group_df["last_season_av"].isnull()))].copy()
        unfit_for_adjustment_df['madden_adjustment'] = -0.5

        fit_for_adjustment_df = group_df[((group_df['is_rookie'] == False) & (group_df["last_season_av"].notnull()))].copy()
        # Standardize the allowed adjustment features to range [-3, 3]
        adj_df = fit_for_adjustment_df[allowed_adjust_preseason_cols]
        standardized_features = 6 * (adj_df - adj_df.min()) / (adj_df.max() - adj_df.min()) - 3

        # Madden rating data
        madden_rating_df = fit_for_adjustment_df[['overallrating']].copy()
        standardized_madden = 6 * (madden_rating_df - madden_rating_df.min()) / (madden_rating_df.max() - madden_rating_df.min()) - 3

        # Compute adjustment factor (bounded between -3 and 3)
        standardized_diff = (standardized_features.mean(axis=1) - standardized_madden['overallrating'])
        madden_adjustment = standardized_diff.clip(-3, 3)

        # Apply adjustment to Madden ratings
        fit_for_adjustment_df['madden_adjustment'] = madden_adjustment

        group_df = pd.concat([
            rookies_df,
            unfit_for_adjustment_df,
            fit_for_adjustment_df
        ], ignore_index=True)
        for madden_feature in MADDEN_FEATURES:
            group_df[madden_feature] = (group_df['madden_adjustment'] + group_df[madden_feature]).clip(lower=5, upper=99)  # Bound ratings between 5 and 99
        group_df = group_df.drop(columns=['madden_adjustment'])
        dfs.append(group_df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def adjust_preseason_ratings(df):
    """
    After we impute ratings we need to adjust the default madden ratings to be fit more to
    what the game data says from last year and their last av
    :param df:
    :return:
    """
    ## madden_rating overall
    ## av
    ## last season VALUE
    ## last season fantasy_points
    ADJUST_PRESEASON_RATING_FEATURES_DICT = {
        "quarterback": ["VALUE_ELO","passing_epa"]
    }

    return _simple_adjust_preseason_ratings(df)

    dfs = []
    #for pos_group in list(HIGH_POSITION_MAPPER.keys()):
    for pos_group in ['quarterback']:
        pos_group_features = []
        for feature in ADJUST_PRESEASON_RATING_FEATURES_DICT[pos_group]:
            pos_group_features.append(f"season_avg_{feature}")
        allowed_adjust_preseason_cols = pos_group_features + ['last_season_av']
        group_df = df[df['position_group'] == pos_group].copy()

        rookies_df = group_df[group_df['is_rookie'] == True].copy()
        rookies_df['madden_adjustment'] = 0
        if pos_group == 'quarterback':
            unfit_for_adjustment_df = group_df[group_df["season_avg_attempts"] < 3].copy()
            unfit_for_adjustment_df['madden_adjustment'] = -0.5
        else:
            unfit_for_adjustment_df = pd.DataFrame()

        fit_for_adjustment_df = group_df[((group_df['is_rookie'] == False) & (group_df["season_avg_attempts"]>=3))].copy()
        # Standardize the allowed adjustment features to range [-3, 3]
        adj_df = fit_for_adjustment_df[allowed_adjust_preseason_cols]
        standardized_features = 6 * (adj_df - adj_df.min()) / (adj_df.max() - adj_df.min()) - 3

        # Madden rating data
        madden_rating_df = fit_for_adjustment_df[['overallrating']].copy()
        standardized_madden = 6 * (madden_rating_df - madden_rating_df.min()) / (madden_rating_df.max() - madden_rating_df.min()) - 3

        # Compute adjustment factor (bounded between -3 and 3)
        standardized_diff = (standardized_features.mean(axis=1) - standardized_madden['overallrating'])
        madden_adjustment = standardized_diff.clip(-3, 3)

        # Apply adjustment to Madden ratings
        fit_for_adjustment_df = fit_for_adjustment_df.reset_index(drop=True)
        fit_for_adjustment_df['madden_adjustment'] = madden_adjustment

        group_df = pd.concat([
            rookies_df,
            unfit_for_adjustment_df,
            fit_for_adjustment_df
        ], ignore_index=True)
        group_df['adjusted_overallrating'] = (fit_for_adjustment_df['madden_adjustment'] * fit_for_adjustment_df['overallrating']).clip(lower=35, upper=99)  # Bound ratings between 35 and 99

        dfs.append(group_df)
    df = pd.concat(dfs,ignore_index=True)
    return df


def make_player_stats(season, week = None, season_type=None, position_group='quarterback'):
    """
    Mode can be
    :param season:
    :param week:
    :param season_type:
    :param position_group:
    :return:
    """
    if position_group in ['d_field','d_line']:
        group = 'def'
    elif position_group == ['kick','special_teams']:
        group = 'st'
    else:
        group = ''
    df = collect_weekly_espn_player_stats(season, week=week, season_type=season_type,  group=group)

    ## Add position specific extras, column selection and data transforms
    if position_group == 'quarterback':
        df = make_qb_career(df[df.position_group == position_group].copy())
    else:
        pass

    return df

### Has Last Season Player Stats (avgs)