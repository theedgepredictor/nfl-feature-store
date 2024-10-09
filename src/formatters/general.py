import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype


def df_rename_pivot(df, all_cols, pivot_cols, t1_prefix, t2_prefix, sub_merge_df=None):
    '''
    The reverse of a df_rename_fold
    Pivot one generic type into two prefixed column types
    Ex: team_id -> away_team_id and home_team_id
    '''
    try:
        df = df[all_cols]
        t1_cols = [t1_prefix + i for i in all_cols if i not in pivot_cols]
        t2_cols = [t2_prefix + i for i in all_cols if i not in pivot_cols]
        original_cols = [i for i in all_cols if i not in pivot_cols]

        t1_renamed_pivot_df = df.rename(columns=dict(zip(original_cols, t1_cols)))
        t2_renamed_pivot_df = df.rename(columns=dict(zip(original_cols, t2_cols)))

        if sub_merge_df is None:
            df_out = pd.merge(t1_renamed_pivot_df, t2_renamed_pivot_df, on=pivot_cols).reset_index().drop(columns='index')
        else:
            sub_merge_cols = sub_merge_df.columns.values
            t1_sub_df = pd.merge(sub_merge_df, t1_renamed_pivot_df, how='inner', left_on=[t1_prefix + i for i in pivot_cols], right_on=pivot_cols).drop(columns=pivot_cols)
            t2_sub_df = pd.merge(sub_merge_df, t2_renamed_pivot_df, how='inner', left_on=[t2_prefix + i for i in pivot_cols], right_on=pivot_cols).drop(columns=pivot_cols)
            df_out = pd.merge(t1_sub_df, t2_sub_df, on=list(sub_merge_cols))
        return df_out
    except Exception as e:
        print("--df_rename_pivot-- " + str(e))
        print(f"columns in: {df.columns}")
        print(f"shape: {df.shape}")
        return df


def df_rename_fold(df, t1_prefix, t2_prefix):
    '''
    The reverse of a df_rename_pivot
    Fold two prefixed column types into one generic type
    Ex: away_team_id and home_team_id -> team_id
    '''
    try:
        t1_all_cols = [i for i in df.columns if t2_prefix not in i]
        t2_all_cols = [i for i in df.columns if t1_prefix not in i]

        t1_cols = [i for i in df.columns if t1_prefix in i]
        t2_cols = [i for i in df.columns if t2_prefix in i]
        t1_new_cols = [i.replace(t1_prefix, '') for i in df.columns if t1_prefix in i]
        t2_new_cols = [i.replace(t2_prefix, '') for i in df.columns if t2_prefix in i]

        t1_df = df[t1_all_cols].rename(columns=dict(zip(t1_cols, t1_new_cols)))
        t2_df = df[t2_all_cols].rename(columns=dict(zip(t2_cols, t2_new_cols)))

        df_out = pd.concat([t1_df, t2_df]).reset_index().drop(columns='index')
        return df_out
    except Exception as e:
        print("--df_rename_fold-- " + str(e))
        print(f"columns in: {df.columns}")
        print(f"shape: {df.shape}")
        return df


def df_rename_dif(df, t1_prefix=None, t2_prefix=None, t1_cols=None, t2_cols=None, sub_prefix=''):
    '''
    An extension of the df_rename_pivot
    Take the difference of two prefixed column types
    Ex: away_team_turnovers - home_team_turnovers -> team_turnovers_dif
    Note: This method applies the difference to the columns and removes the two prefixed column types
    '''
    if t1_cols is None and t2_cols is None:
        if t1_prefix is None or t2_prefix is None:
            raise Exception('You must specify either prefix or cols')
        t1_cols = [i for i in df.columns if t1_prefix in i]
        t2_cols = [i for i in df.columns if t2_prefix in i]
    for t1_col, t2_col in zip(t1_cols, t2_cols):
        if is_numeric_dtype(df[t1_col]) and is_numeric_dtype(df[t2_col]):
            df[f"dif_{t1_col.replace(t1_prefix, sub_prefix)}"] = df[t1_col] - df[t2_col]
    df_out = df.drop(columns=t1_cols + t2_cols)
    return df_out

def df_rename_exavg(df, t1_prefix=None, t2_prefix=None, t1_cols=None, t2_cols=None, sub_prefix=''):
    '''
    An extension of the df_rename_pivot
    Take the average of two prefixed column types to get the exavg (expected average)
    Ex: (away_team_turnovers + home_team_turnovers)/2 -> team_turnovers_df_rename_exavg
    Note: This method applies the exavg to the columns and removes the two prefixed column types
    '''
    if t1_cols is None and t2_cols is None:
        if t1_prefix is None or t2_prefix is None:
            raise Exception('You must specify either prefix or cols')
        t1_cols = [i for i in df.columns if t1_prefix in i]
        t2_cols = [i for i in df.columns if t2_prefix in i]
    for t1_col, t2_col in zip(t1_cols, t2_cols):
        if is_numeric_dtype(df[t1_col]) and is_numeric_dtype(df[t2_col]):
            df[f"exavg_{t1_col.replace(t1_prefix, sub_prefix)}"] = (df[t1_col] + df[t2_col]) / 2
    df_out = df.drop(columns=t1_cols + t2_cols)
    return df_out

def df_rename_shift(df, drop_cols=None):
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)

    root_cols = [col for col in df.columns if '_offense' not in col and '_defense' not in col and 'away_' not in col and 'home_' not in col]

    away_cols = [col for col in df.columns if '_offense' not in col and '_defense' not in col and 'away_' in col and 'home_' not in col]
    away_rename_dict = {col: col.replace('away_', '') for col in away_cols}
    home_cols = [col for col in df.columns if '_offense' not in col and '_defense' not in col and 'away_' not in col and 'home_' in col]
    home_rename_dict = {col: col.replace('home_', '') for col in home_cols}

    off_away_cols = [col for col in df.columns if '_offense' in col and 'away_' in col]
    off_away_rename_dict = {col: col.replace('away_', '') for col in off_away_cols}
    def_away_cols = [col for col in df.columns if '_defense' in col and 'away_' in col]
    def_away_rename_dict = {col: col.replace('away_', '') for col in def_away_cols}

    off_home_cols = [col for col in df.columns if '_offense' in col and 'home_' in col]
    off_home_rename_dict = {col: col.replace('home_', '') for col in off_home_cols}
    def_home_cols = [col for col in df.columns if '_defense' in col and 'home_' in col]
    def_home_rename_dict = {col: col.replace('home_', '') for col in def_home_cols}

    away_df = df[root_cols + away_cols + off_away_cols + def_home_cols].rename(columns={**away_rename_dict, **off_away_rename_dict, **def_home_rename_dict})
    away_df['is_home'] = 0
    home_df = df[root_cols + home_cols + off_home_cols + def_away_cols].rename(columns={**home_rename_dict, **off_home_rename_dict, **def_away_rename_dict})
    home_df['is_home'] = 1
    del df
    out_df = pd.concat([away_df, home_df])
    return out_df

def suffix_to_prefix(df, suffix='_home', prefix='home_'):
    df.columns = [
        prefix + col.replace(suffix, '') if suffix in col else col
        for col in df.columns
    ]
    return df
