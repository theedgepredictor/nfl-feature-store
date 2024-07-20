import os

import pandas as pd
import pyarrow as pa
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.experiment import ExperimentData
from src.feature_stores.player_season import make_season_feature_store
from src.models.baseline import train_baseline_model
from src.models.feature_selection import FeatureSelection
from src.models.hpo import ModelTuner
from src.utils import put_dataframe, get_dataframe, train_test_split


def main():
    root_path = './data/feature_store'
    fs_type_path = 'player/season'
    fs_file_name = 'fs.parquet'

    experiment_class = "fantasy_football"
    year = 2022

    position = 'TE'
    target = 'fantasy_points'
    n_features = 25
    n_trials = 40

    path = f"{root_path}/{fs_type_path}/{fs_file_name}"
    experiment_obj = ExperimentData(experiment_class=experiment_class, year=year, position=position, target=target)
    ###################################################################
    ## 1. Load Experiment
    ###################################################################
    print("## 1. Load Experiment")
    experiment = experiment_obj.load_experiment()
    if experiment is None:
        experiment_obj.set_feature_store(path)
    fs_df = get_dataframe(path)

    ###################################################################
    ## 2. Load Baseline Model Performance
    ###################################################################
    print("## 2. Load Baseline Model Performance")
    if experiment is None or experiment['initial_baseline_performance'] is None:
        baseline_mae, baseline_model, baseline_meta_test = train_baseline_model(fs_df, position, year, target)
        experiment_obj.set_initial_baseline_performance(baseline_mae)
    else:
        baseline_mae = experiment_obj.initial_baseline_performance
    print(f'Mean Absolute Error for {year} {position}s on test set: {baseline_mae}')

    ###################################################################
    ## 3. Load Feature Selection
    ###################################################################
    print("## 3. Load Feature Selection")
    X_train, y_train, X_holdout, y_holdout, X_test, y_test, meta_test = train_test_split(fs_df, position, year, target, holdout=True)

    X = pd.concat([X_train, X_holdout, X_test])
    y = pd.concat([y_train, y_holdout, y_test])

    if experiment is None or experiment.get('feature_selection') is None:
        feature_selector = FeatureSelection(X, y, n_features=n_features, visualize=False)
        # selected_features = feature_selector.pipeline_feature_selection(method='random_forest')

        feature_selector.random_forest_feature_importance()

        selected_features = feature_selector.get_selected_features_names('random_forest')
        experiment_obj.set_selected_features(selected_features)
    else:
        selected_features = experiment.feature_selection
    X_train = X_train[selected_features]
    X_holdout = X_holdout[selected_features]
    X_test = X_test[selected_features]

    ###################################################################
    ## 3. Load Hyperparameter Tuning
    ###################################################################
    print("## 3. Load Hyperparameter Tuning")
    model_tuner = ModelTuner(X_train, y_train, X_holdout, y_holdout, X_test, y_test, n_trials=n_trials)

    if experiment is None or experiment.get('hyperparameters') is None:
        best_hyperparameters = model_tuner.run_optimization()
        experiment_obj.set_hyperparameters(best_hyperparameters)
    else:
        best_hyperparameters = experiment.get('hyperparameters')

    model = model_tuner.fit_predict(best_hyperparameters)

    ###################################################################
    ## 4. Load Tuned Model Performance
    ###################################################################
    print("## 4. Load Tuned Model Performance")
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred_test = model.predict(dtest)

    if experiment is None or experiment.get('tuned_model_performance') is None:
        tuned_model_mae = mean_absolute_error(y_test, y_pred_test)
        experiment_obj.set_tuned_model_performance(tuned_model_mae)
    else:
        tuned_model_mae = experiment.get('tuned_model_performance')

    ###################################################################
    ## 5. Compare Baseline to Tuned
    ###################################################################
    print("## 5. Compare Baseline to Tuned")
    print(f'Baseline MAE: {baseline_mae}')
    print(f'Mean Absolute Error for {year} {position}s on test set: {tuned_model_mae}')
    print(f"Difference in MAE: {baseline_mae - tuned_model_mae}")

    meta_test['predictions'] = y_pred_test
    meta_test['predictions'] = meta_test['predictions'].astype('Float32')
    meta_test['created_at'] = pd.Timestamp.now()

    ###################################################################
    ## 6. Save Inference
    ###################################################################
    print("## 6. Save Inference")
    put_dataframe(meta_test, experiment_obj.inference_path)

    ###################################################################
    ## 7. Save Experiment
    ###################################################################
    print("## 7. Save Experiment")
    experiment_obj.save_experiment()

if __name__ == '__main__':
    main()