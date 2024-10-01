from src.transforms.general import make_avg_group_features

def make_normal_play_group_features(data):
    ## Create General features for rolling avgs
    general_features_dict = {
        'first_down': 'sum',
        'third_down_converted': 'sum',
        'third_down_failed': 'sum',
        'fourth_down_converted': 'sum',
        'fourth_down_failed': 'sum',

        # Penalty features
        'first_down_penalty': 'sum',
        # 'penalty_yards': 'sum',
        # 'penalty': 'sum',
        # 'defensive_penalty': 'sum',

        # Add advanced general features here

        'shotgun': 'sum',
        'no_huddle': 'sum',
        'qb_dropback': 'sum',
        'qb_scramble': 'sum',
        'goal_to_go': 'sum',
        'is_redzone': 'sum',
        #'successful_two_point_conversion': 'sum',
        #'drive': 'nunique',  # Number of unique drives in the quarter
        #'series': 'nunique',  # Number of unique series in the quarter
    }
    general_features = data[
        (~data['down'].isna()) &
        (data['play_type'].isin(['pass', 'qb_kneel', 'qb_spike', 'run']))
        ].copy()
    general_features = make_avg_group_features(general_features, general_features_dict)
    ## make down percentages
    general_features['avg_third_down_percentage_offense'] = general_features.apply(
        lambda row: row['avg_third_down_converted_offense'] / (row['avg_third_down_converted_offense'] + row['avg_third_down_failed_offense'])
        if (row['avg_third_down_converted_offense'] + row['avg_third_down_failed_offense']) > 0 else 0,
        axis=1
    )
    general_features['avg_third_down_percentage_defense'] = general_features.apply(
        lambda row: row['avg_third_down_converted_defense'] / (row['avg_third_down_converted_defense'] + row['avg_third_down_failed_defense'])
        if (row['avg_third_down_converted_defense'] + row['avg_third_down_failed_defense']) > 0 else 0,
        axis=1
    )

    general_features['avg_fourth_down_percentage_offense'] = general_features.apply(
        lambda row: row['avg_fourth_down_converted_offense'] / (row['avg_fourth_down_converted_offense'] + row['avg_fourth_down_failed_offense'])
        if (row['avg_fourth_down_converted_offense'] + row['avg_fourth_down_failed_offense']) > 0 else 0,
        axis=1
    )
    general_features['avg_fourth_down_percentage_defense'] = general_features.apply(
        lambda row: row['avg_fourth_down_converted_defense'] / (row['avg_fourth_down_converted_defense'] + row['avg_fourth_down_failed_defense'])
        if (row['avg_fourth_down_converted_defense'] + row['avg_fourth_down_failed_defense']) > 0 else 0,
        axis=1
    )

    ## rename columns
    general_features = general_features.rename(columns={
        'avg_posteam_score_post_offense': 'avg_points_offense',
        'avg_posteam_score_post_defense': 'avg_points_defense',
        'avg_score_differential_post_offense': 'avg_point_differential_offense',
        'avg_score_differential_post_defense': 'avg_point_differential_defense',
    })
    return general_features

def make_general_group_features(data):
    """
    Unfiltered play by play data features for offense and defense

    Parameters:
        data (DataFrame): Play-by-play dataframe containing the relevant play data.

    Returns:
        DataFrame: Combined dataframe containing offensive and defensive avg values.
    """

    #### Handles time of possession for offense and defense

    group_features_dict = {
        'posteam_score_post': 'last',
        'score_differential_post': 'last',
        'epa': 'sum',
        'wpa': 'sum',
        'time_of_possession': 'sum',
        'field_goal_made': 'sum',
        'field_goal_attempt':'sum',
        'field_goal_distance': 'mean',
        'extra_point_made': 'sum',
        'extra_point_attempt': 'sum',
        'turnover':'sum',
    }
    data['turnover'] = data['fumble_lost'] + data['interception']
    data['game_seconds_remaining'] = data['game_seconds_remaining'].fillna(0)
    # For each play in the game calculate the difference in the clock
    data['time_of_possession'] = data.groupby('game_id')['game_seconds_remaining'].diff(-1).abs()
    data['field_goal_made'] = data['field_goal_result'] == 'made'
    data['extra_point_made'] = data['extra_point_result'] == 'made'
    data['field_goal_distance'] = None
    data.loc[data['field_goal_attempt']==1, 'field_goal_distance'] = data.loc[data['field_goal_attempt']==1, 'kick_distance']

    general_features = make_avg_group_features(data, group_features_dict)
    general_features['avg_field_goal_percentage_offense'] = general_features.apply(
        lambda row: row['avg_field_goal_made_offense'] / (row['avg_field_goal_attempt_offense'])
        if (row['avg_field_goal_attempt_offense']) > 0 else 0,
        axis=1
    )
    general_features['avg_field_goal_percentage_defense'] = general_features.apply(
        lambda row: row['avg_field_goal_made_defense'] / (row['avg_field_goal_attempt_defense'])
        if (row['avg_field_goal_attempt_defense']) > 0 else 0,
        axis=1
    )
    general_features['avg_extra_point_percentage_offense'] = general_features.apply(
        lambda row: row['avg_extra_point_made_offense'] / (row['avg_extra_point_attempt_offense'])
        if (row['avg_extra_point_attempt_offense']) > 0 else 0,
        axis=1
    )
    general_features['avg_extra_point_percentage_defense'] = general_features.apply(
        lambda row: row['avg_extra_point_made_defense'] / (row['avg_extra_point_attempt_defense'])
        if (row['avg_extra_point_attempt_defense']) > 0 else 0,
        axis=1
    )
    ## rename columns
    general_features = general_features.rename(columns={
        'avg_posteam_score_post_offense': 'avg_points_offense',
        'avg_posteam_score_post_defense': 'avg_points_defense',
        'avg_score_differential_post_offense': 'avg_point_differential_offense',
        'avg_score_differential_post_defense': 'avg_point_differential_defense',
    })

    return general_features.drop_duplicates(subset=['team', 'season', 'week'])