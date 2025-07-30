import re
import numpy as np
import pandas as pd
import datetime
import os
from typing import List
import pyarrow as pa
from bs4 import BeautifulSoup
from pandas.core.dtypes.common import is_numeric_dtype


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

def get_seasons_to_update(root_path, feature_store_name):
    """
    Get a list of seasons to update based on the root path and sport.

    Args:
        root_path (str): Root path for the sport data.
        sport (ESPNSportTypes): Type of sport.

    Returns:
        List: List of seasons to update.
    """
    current_season = find_year_for_season()
    if os.path.exists(f'{root_path}/{feature_store_name}'):
        seasons = os.listdir(f'{root_path}/{feature_store_name}')
        fs_season = -1
        for season in seasons:
            temp = int(season.split('.')[0])
            if temp > fs_season:
                fs_season = temp
    else:
        fs_season = 2002
    if fs_season == -1:
        fs_season = 2002
    return list(range(fs_season, current_season + 1))


def find_year_for_season( date: datetime.datetime = None):
    """
    Find the year for a specific season based on the league and date.

    Args:
        league (ESPNSportTypes): Type of sport.
        date (datetime.datetime): Date for the sport (default is None).

    Returns:
        int: Year for the season.
    """
    SEASON_START_MONTH = {

        "NFL": {'start': 6, 'wrap': False},
    }
    if date is None:
        today = datetime.datetime.utcnow()
    else:
        today = date
    start = SEASON_START_MONTH["NFL"]['start']
    wrap = SEASON_START_MONTH["NFL"]['wrap']
    if wrap and start - 1 <= today.month <= 12:
        return today.year + 1
    elif not wrap and start == 1 and today.month == 12:
        return today.year + 1
    elif not wrap and not start - 1 <= today.month <= 12:
        return today.year - 1
    else:
        return today.year

def get_webpage_soup(html, html_attr=None, attr_key_val=None):
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        if html_attr:
            soup = soup.find(html_attr, attr_key_val)
        return soup
    return None

def clean_string(s):
    if isinstance(s, str):
        return re.sub("[\W_]+",'',s)
    else:
        return s

def re_alphanumspace(s):
    if isinstance(s, str):
        return re.sub("^[a-zA-Z0-9 ]*$",'',s)
    else:
        return s

def re_braces(s):
    if isinstance(s, str):
        return re.sub("[\(\[].*?[\)\]]", "", s)
    else:
        return s

def re_numbers(s):
    if isinstance(s, str):
        n = ''.join(re.findall(r'\d+', s))
        return int(n) if n != '' else n
    else:
        return s


def name_filter(s):
    s = clean_string(s)
    s = re_braces(s)
    s = str(s)
    s = s.replace(' ', '').lower()
    return s