import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Any
import datetime

from src.consts import FEATURE_STORE_METADATA


def get_fs_meta_dict():
    return {i["name"]:i["dtype"] for i in FEATURE_STORE_METADATA}

def _cast2date(value: Any) -> Any:
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, datetime.date):
        return value
    return pd.to_datetime(value).date()

def _cast_pandas_column(df: pd.DataFrame, col: str, desired_type: str, memory_saver=True) -> pd.DataFrame:
    if desired_type == "datetime64":
        df[col] = pd.to_datetime(df[col])
    elif desired_type == "date":
        df[col] = df[col].apply(lambda x: _cast2date(value=x)).replace(to_replace={pd.NaT: None})
    elif desired_type == "bytes":
        df[col] = df[col].astype("string").str.encode(encoding="utf-8").replace(to_replace={pd.NA: None})
    elif desired_type == "decimal":
        # First cast to string
        df = _cast_pandas_column(df=df, col=col, desired_type="string")
        # Then cast to decimal
        df[col] = df[col].apply(lambda x: Decimal(str(x)) if str(x) not in ("", "none", "None", " ", "<NA>") else None)
    else:
        try:
            df[col] = df[col].astype(desired_type)
            if memory_saver:
                if '64' in str(df[col].dtype):
                    df[col] = df[col].astype('Int32') if pd.api.types.is_integer_dtype(df[col]) else df[col].astype('Float32')
        except TypeError as ex:
            if "object cannot be converted to an IntegerDtype" not in str(ex):
                raise ex
            df[col] = df[col].apply(lambda x: int(x) if str(x) not in ("", "none", "None", " ", "<NA>") else None).astype(desired_type)
        except pd.errors.IntCastingNaNError as ex:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
        except Exception as ex:
            print(f"Issue Converting: {col}")
            raise ex
    return df

def fs_apply_type(df, method='equals'):
    fs_meta_dict = get_fs_meta_dict()
    fs_meta_keys = fs_meta_dict.keys()
    for column in df.columns:
        if method == ' equals':
            if column in fs_meta_keys:
                df = _cast_pandas_column(df, column, fs_meta_dict[column])
        elif method == 'contains':
            for fs_key in fs_meta_keys:
                if fs_key in column:
                    desired_dtype = "Float32" if 'avg_' in column else fs_meta_dict[fs_key]
                    df = _cast_pandas_column(df, column, desired_dtype)
    return df