from typing import List

import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


class ModelTuner:
    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_val: pd.DataFrame,
            y_val: pd.DataFrame,
            X_test: pd.DataFrame,
            y_test: pd.DataFrame,
            n_trials: int = 50
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.hyperparameters = None
        self.model = None

    def objective(self, trial):
        param = {
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
        }
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        # Use validation set as trial test set
        dtest = xgb.DMatrix(self.X_val, label=self.y_val)

        for train_index, test_index in kfold.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
            y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[test_index]

            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
            bst = xgb.train(param, dtrain, num_boost_round=param['n_estimators'], evals=[(dval, 'validation')], early_stopping_rounds=50)
            y_pred_test = bst.predict(dtest)

            score = mean_absolute_error(self.y_val, y_pred_test.round())
            scores.append(score)

        return float(np.mean(scores))

    def run_optimization(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial), n_trials=self.n_trials)
        self.hyperparameters = study.best_params
        return self.hyperparameters

    def fit_predict(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.hyperparameters['objective'] = 'reg:squarederror'
        self.hyperparameters['eval_metric'] = 'mae'

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        self.model = xgb.train(self.hyperparameters, dtrain, num_boost_round=self.hyperparameters['n_estimators'], evals=[(dval, 'validation')], early_stopping_rounds=50)

        y_pred_val = self.model.predict(dval)
        y_pred_test = self.model.predict(dtest)

        mae_val = mean_absolute_error(self.y_val, y_pred_val.round())
        mae_test = mean_absolute_error(self.y_test, y_pred_test.round())

        print(f'Validation MAE: {mae_val:.4f}')
        print(f'Test MAE: {mae_test:.4f}')

        return self.model