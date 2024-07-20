import re

import numpy as np
import pandas as pd
import datetime
import os
from typing import List
import pyarrow as pa

def get_dataframe(path: str, columns: List = None):
    """
    Read a DataFrame from a parquet file.

    Args:
        path (str): Path to the parquet file.
        columns (List): List of columns to select (default is None).

    Returns:
        pd.DataFrame: Read DataFrame.
    """
    try:
        return pd.read_parquet(path, engine='pyarrow', dtype_backend='numpy_nullable', columns=columns)
    except Exception as e:
        print(e)
        return pd.DataFrame()


def put_dataframe(df: pd.DataFrame, path: str):
    """
    Write a DataFrame to a parquet file.

    Args:
        df (pd.DataFrame): DataFrame to write.
        path (str): Path to the parquet file.
        schema (dict): Schema dictionary.

    Returns:
        None
    """
    key, file_name = path.rsplit('/', 1)
    if file_name.split('.')[1] != 'parquet':
        raise Exception("Invalid Filetype for Storage (Supported: 'parquet')")
    os.makedirs(key, exist_ok=True)
    df.to_parquet(f"{key}/{file_name}",engine='pyarrow', schema=pa.Schema.from_pandas(df))


def create_dataframe(obj, schema: dict):
    """
    Create a DataFrame from an object with a specified schema.

    Args:
        obj: Object to convert to a DataFrame.
        schema (dict): Schema dictionary.

    Returns:
        pd.DataFrame: Created DataFrame.
    """
    df = pd.DataFrame(obj)
    for column, dtype in schema.items():
        df[column] = df[column].astype(dtype)
    return df

def get_meta_cols(target):
    return [
    'player_id',
    'season',
    target,
    'display_name',
    'position_group'
]

def train_test_split(feature_store_df, position, year, target, holdout=False):
    train_years = list(range(2010, year - 1))
    holdout_year = year - 1
    test_year = year
    meta_cols = [
        'player_id',
        'season',
        'fantasy_points',
        'display_name',
        'college_name',
        'birth_date',
        'entry_year',
        'position',
        'position_group'
    ]

    # Filter for the specific position
    pos_df = feature_store_df[feature_store_df['position_group'] == position].copy()

    if position == 'QB':
        filt = (pos_df['total_last_year_completions'] >= 0) | (pos_df['total_2_years_ago_completions'] >= 0) | (pos_df['years_of_experience'] == 0)
        print("Records Dropped during train-test split: ",pos_df.shape[0] - sum(filt))
        pos_df = pos_df[filt].copy()
    elif position == 'RB':
        filt = (pos_df['total_last_year_carries'] >= 0) | (pos_df['total_2_years_ago_carries'] >= 0) | (pos_df['years_of_experience'] == 0)
        print("Records Dropped during train-test split: ",pos_df.shape[0] - sum(filt))
        pos_df = pos_df[filt].copy()
    elif position == 'WR':
        filt = (pos_df['total_last_year_receptions'] >= 0) | (pos_df['total_2_years_ago_receptions'] >= 0) | (pos_df['years_of_experience'] == 0)
        print("Records Dropped during train-test split: ",pos_df.shape[0] - sum(filt))
        pos_df = pos_df[filt].copy()
    elif position == 'TE':
        filt = (pos_df['total_last_year_receptions'] >= 0) | (pos_df['total_2_years_ago_receptions'] >= 0) | (pos_df['years_of_experience'] == 0)
        print("Records Dropped during train-test split: ",pos_df.shape[0] - sum(filt))
        pos_df = pos_df[filt].copy()

    # Split the data
    train_df = pos_df[pos_df['season'].isin(train_years)].copy()
    holdout_df = pos_df[pos_df['season'] == holdout_year].copy()
    test_df = pos_df[pos_df['season'] == test_year].copy()

    cols_to_drop = list(set(meta_cols + list(train_df.select_dtypes(exclude=[np.number]).columns)))
    if target == 'fantasy_points':
        cols_to_drop.extend([col for col in train_df.columns if 'ppr' in col] + ['position_rank', 'avg_last_year_fantasy_points'])
    elif target == 'fantasy_points_ppr':
        cols_to_drop.extend([col for col in train_df.columns if 'fantasy_points' in col and 'ppr' not in col]  + ['ppr_position_rank', 'avg_last_year_ppr_fantasy_points'])
    else:
        Exception('Invalid target variable: Accepted values are "fantasy_points" or "fantasy_points_ppr"')
    #print(cols_to_drop)
    # Define features and target variable
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df[target]

    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df[target]

    if holdout:
        X_holdout = holdout_df.drop(columns=cols_to_drop)
        y_holdout = holdout_df[target]
    else:
        X_holdout = None
        y_holdout = None
        X_train = pd.concat([X_train, holdout_df.drop(columns=cols_to_drop)])
        y_train = pd.concat([y_train, holdout_df[target]])

    return X_train, y_train, X_holdout, y_holdout, X_test, y_test, test_df[get_meta_cols(target)]