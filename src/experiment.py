import json
import os


class ExperimentData:
    def __init__(self, experiment_class, year, position, target):
        """
        Crud operations for collecting information for an experiment

        1. Define what feature store the experiment used
        2. Define what experiment class this is part of
        3. Define year the experiment is ran for
        4. Define the target variable used
        5. Define the position the model was built for
        6. Define initial baseline model performance
        7. Define features selected from feature selection process
        8. Define hyperparameters selected from hpo process
        9. Define tuned model performance
        10. Define inference path
        """
        self.experiment_class = experiment_class
        self.year = year
        self.position = position
        self.target = target

        self.feature_store = None
        self.initial_baseline_performance = None
        self.selected_features = None
        self.hyperparameters = None
        self.tuned_model_performance = None
        self.experiment_path = f"./data/experiments/{self.experiment_class}/{self.year}/{self.position}/{self.target}.json"
        self.inference_path = f"./data/experiments/{self.experiment_class}/{self.year}/{self.position}/{self.target}.parquet"

        self._ensure_directory_exists(os.path.dirname(self.experiment_path))


    def _ensure_directory_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_experiment(self):
        experiment_data = {
            "feature_store": self.feature_store,
            "initial_baseline_performance": self.initial_baseline_performance,
            "selected_features": self.selected_features,
            "hyperparameters": self.hyperparameters,
            "tuned_model_performance": self.tuned_model_performance,
            "inference_path": self.inference_path
        }

        with open(self.experiment_path, 'w') as f:
            json.dump(experiment_data, f, indent=4)

    def load_experiment(self):
        if os.path.exists(self.experiment_path):
            with open(self.experiment_path, 'r') as f:
                experiment_data = json.load(f)
            self.feature_store = experiment_data.get("feature_store")
            self.initial_baseline_performance = experiment_data.get("initial_baseline_performance")
            self.selected_features = experiment_data.get("selected_features")
            self.hyperparameters = experiment_data.get("hyperparameters")
            self.tuned_model_performance = experiment_data.get("tuned_model_performance")
            self.inference_path = experiment_data.get("inference_path")
            return self.__dict__
        else:
            print(f"No experiment data found at {self.experiment_path}")
            return None

    def set_feature_store(self, feature_store):
        self.feature_store = feature_store

    def set_initial_baseline_performance(self, performance):
        self.initial_baseline_performance = performance

    def set_selected_features(self, features):
        self.selected_features = features

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def set_tuned_model_performance(self, performance):
        self.tuned_model_performance = performance