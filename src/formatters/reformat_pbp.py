import numpy as np
import pandas as pd
import numpy

def plays_formatting(data):
    data.fillna(-1000000, inplace=True)
    data.replace(-1000000, None, inplace=True)
    ## Fixes
    data['quarter_seconds_remaining'] = data['quarter_seconds_remaining'].ffill()
    data['game_seconds_remaining'] = data['game_seconds_remaining'].ffill()

    #############################################################################################
    ## Attrs
    #############################################################################################
    data['is_redzone'] = data['yardline_100'] <= 20
    data['is_middle_8'] = ((data['qtr'] == 2) & (data['quarter_seconds_remaining'] <= 60 * 4)) | (data['qtr'] == 3) & (data['quarter_seconds_remaining'] >= (60 * 15) - (60 * 4))
    data['is_third_and_short'] = (data['down'] == 3) & (data['ydstogo'] < 3)
    data['is_third_and_medium'] = (data['down'] == 3) & (data['ydstogo'] >= 3) & (data['ydstogo'] < 7)
    data['is_third_and_long'] = (data['down'] == 3) & (data['ydstogo'] >= 7)
    data['successful_two_point_conversion'] = np.where(
        data['two_point_conv_result'] == 'success', 1,
        np.where(
            data['two_point_conv_result'].isna() & data['desc'].str.contains('ATTEMPT SUCCEEDS'), 1, 0
        )
    )
    data['sack_yards'] = None
    data.loc[data.sack == 1, 'sack_yards'] = data.loc[data.sack == 1, 'yards_gained']
    data = data.drop(columns=['two_point_conv_result'])
    ### Pass attrs
    data['short_left_pass'] = (data['pass_length'] == 'short') & (data['pass_location'] == 'left')
    data['short_middle_pass'] = (data['pass_length'] == 'short') & (data['pass_location'] == 'middle')
    data['short_right_pass'] = (data['pass_length'] == 'short') & (data['pass_location'] == 'right')
    data['deep_left_pass'] = (data['pass_length'] == 'deep') & (data['pass_location'] == 'left')
    data['deep_middle_pass'] = (data['pass_length'] == 'deep') & (data['pass_location'] == 'middle')
    data['deep_right_pass'] = (data['pass_length'] == 'deep') & (data['pass_location'] == 'right')
    data = data.drop(columns=['pass_length', 'pass_location'])

    ### Rush attrs
    data['left_end_rush'] = (data['run_location'] == 'left') & (data['run_gap'] == 'end')
    data['left_guard_rush'] = (data['run_location'] == 'left') & (data['run_gap'] == 'guard')
    data['left_tackle_rush'] = (data['run_location'] == 'left') & (data['run_gap'] == 'tackle')
    data['right_end_rush'] = (data['run_location'] == 'right') & (data['run_gap'] == 'end')
    data['right_guard_rush'] = (data['run_location'] == 'right') & (data['run_gap'] == 'guard')
    data['right_tackle_rush'] = (data['run_location'] == 'right') & (data['run_gap'] == 'tackle')
    data = data.drop(columns=['run_location', 'run_gap'])
    return data

def penalty_formatting(df):
    """
    Adds additional context to penalties
    """
    ## offensive and defensive penalties ##
    df['off_penalty'] = numpy.where(
        df['penalty_team'] == df['posteam'],
        1,
        0
    )
    df['def_penalty'] = numpy.where(
        df['penalty_team'] == df['defteam'],
        1,
        0
    )
    ## remove nans from penalties to enable groupings ##
    df['penalty_type'] = df['penalty_type'].fillna('No Penalty')
    return df