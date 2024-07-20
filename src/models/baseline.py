import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from src.utils import train_test_split


def train_baseline_model(feature_store_df, position, year, target):
    X_train, y_train, X_holdout, y_holdout, X_test, y_test, meta_test = train_test_split(feature_store_df, position, year, target, holdout=True)

    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
    }
    model = xgb.XGBRegressor(**param, early_stopping_rounds=50, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_holdout, y_holdout)], verbose=False)
    preds = model.predict(X_test)
    meta_test['predictions'] = preds
    mae = mean_absolute_error(y_test, preds)
    return mae, model, meta_test