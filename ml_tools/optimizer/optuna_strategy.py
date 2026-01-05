from typing import Dict, Any
from math import log, ceil, floor

import optuna
from optuna.trial import FixedTrial
import numpy as np
from sklearn.model_selection import KFold

from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model import build_prediction_strategy
from ml_tools.optimizer.search_strategy import SearchStrategy
from ml_tools.optimizer.search_space import (SearchSpace,
                                             IntDimension,
                                             FloatDimension,
                                             CategoricalDimension,
                                             BoolDimension,
                                             StructDimension,
                                             ChoiceDimension,
                                             ListDimension)

class OptunaStrategy(SearchStrategy):
    """ A class implementing the Optuna hyperparameter optimization strategy
    """

    def search(self,
               search_space:      SearchSpace,
               series_collection: SeriesCollection,
               num_trials:        int,
               number_of_folds:   int,
               output_file:       str,
               num_procs:         int) -> PredictionStrategy:

        super().search(search_space,
                       series_collection,
                       num_trials,
                       number_of_folds,
                       output_file,
                       num_procs)

        def log_progress(_study, trial):
            with open(output_file, "a") as output:
                output.write(f"Trial {trial.number}: Params: {trial.params}, Value: {trial.value}\n")
            print(f"Trial {trial.number} completed: Value: {trial.value}")

        print("Starting NN Optimization")
        with open(output_file, 'w') as output:
            output.write("RESULTS\n---------\n")

        study     = optuna.create_study(direction='minimize')
        objective = self._setup_objective(search_space,
                                          series_collection,
                                          number_of_folds,
                                          num_procs)

        study.optimize(objective, n_trials=num_trials, callbacks=[log_progress])

        with open(output_file, 'a') as output:
            output.write("\nBEST PARAMETERS\n----------------\n")
            output.write(f"{study.best_params}\n")

        best_params = self._get_sample(FixedTrial(study.best_params), search_space.dimensions)
        best_model  = build_prediction_strategy(strategy_type     = search_space.prediction_strategy_type,
                                                params            = best_params,
                                                input_features    = search_space.input_features,
                                                predicted_features = search_space.predicted_features,
                                                biasing_model     = search_space.biasing_model)

        best_model.train(series_collection, num_procs=num_procs)

        return best_model

    def _setup_objective(self,
                         search_space:      SearchSpace,
                         series_collection: SeriesCollection,
                         number_of_folds:   int,
                         num_procs:         int) -> callable:
        """ Method to setup the objective function for the Optuna optimization

        Parameters
        ----------
        search_space : SearchSpace
            The hyperparameter search space to explore
        series_collection : SeriesCollection
            The collection of series to use for training and validation
        number_of_folds : int
            The number of folds to use in cross-validation
        num_procs : int
            The number of processes to use for parallel model training
        """

        def objective(trial: optuna.trial.Trial) -> float:
            model = build_prediction_strategy(strategy_type     = search_space.prediction_strategy_type,
                                              params            = self._get_sample(trial, search_space.dimensions),
                                              input_features    = search_space.input_features,
                                              predicted_features = search_space.predicted_features,
                                              biasing_model     = search_space.biasing_model)

            rms = []
            kf = KFold(n_splits=number_of_folds, shuffle=True)
            for fold, (train_idx, val_idx) in enumerate(kf.split(series_collection)):
                print(f"Starting fold {fold + 1}/{number_of_folds}...")

                training_set   = SeriesCollection([series_collection[i] for i in train_idx])
                validation_set = SeriesCollection([series_collection[i] for i in val_idx])

                print("Training started...")
                model.train(training_set, num_procs=num_procs)
                print("Training completed.")

                print("Validating model...")
                feature_order = list(search_space.predicted_features)
                measured_rows = []
                for series in validation_set:
                    parts = []
                    for name in feature_order:
                        v = np.asarray(series[0][name], dtype=float)
                        v = np.atleast_1d(v).reshape(-1)
                        parts.append(v)
                    measured_rows.append(np.concatenate(parts, axis=0))
                measured = np.vstack(measured_rows)

                predicted_rows = []
                for series in model.predict(validation_set):
                    parts = []
                    for name in feature_order:
                        v = np.asarray(series[0][name], dtype=float)
                        v = np.atleast_1d(v).reshape(-1)
                        parts.append(v)
                    predicted_rows.append(np.concatenate(parts, axis=0))
                predicted = np.vstack(predicted_rows)

                diff = measured - predicted
                fold_rms = np.sqrt(np.mean(np.square(diff, dtype=float)))
                print(f"Fold {fold + 1} RMS: {fold_rms}")

                rms.append(fold_rms)

            average_rms = sum(rms) / len(rms)
            print(f"Optimization step completed. Average RMS across folds: {average_rms}")
            return average_rms

        return objective


    def _get_sample(self, trial: optuna.trial.Trial, dimensions: StructDimension) -> Dict:
        """ Method for using an Optuna trial to sample hyperparameters from the search space

        Parameters
        ----------
        trial : optuna.trial.Trial
            An object representing a specific hyperparameter configuration
        dimensions : StructDimension
            The root hyperparameter search space to explore

        Returns
        -------
        Dict
            A dictionary of model parameters extracted from the trial
        """

        def join(parent: str, child: str) -> str:
            return child if not parent else f"{parent}.{child}"

        def index(parent: str, i: int) -> str:
            return f"{parent}_{i}" if parent else f"[{i}]"

        def sample(name: str, dim) -> Any:
            result: Any = None
            if isinstance(dim, IntDimension):
                if dim.log is None:
                    result = trial.suggest_int(name or 'int', dim.low, dim.high)
                else:
                    base = int(dim.log)
                    exp_name = (name + f".log{base}.exp") if name else f'int.log{base}.exp'
                    e = trial.suggest_int(exp_name, ceil(log(dim.low, base)), floor(log(dim.high, base)))
                    result = int(base ** e)
            elif isinstance(dim, FloatDimension):
                if dim.log is None:
                    result = trial.suggest_float(name or 'float', dim.low, dim.high)
                else:
                    base = int(dim.log)
                    exp_name = (name + f".log{base}.exp") if name else f'float.log{base}.exp'
                    e = trial.suggest_float(exp_name, log(dim.low, base), log(dim.high, base))
                    result = float(base ** e)
            elif isinstance(dim, BoolDimension):
                result = trial.suggest_categorical(name or 'bool', dim.choices)
            elif isinstance(dim, CategoricalDimension):
                choices = dim.choices
                def _is_simple(v):
                    return v is None or isinstance(v, (bool, int, float, str))
                if all(_is_simple(c) for c in choices):
                    result = trial.suggest_categorical(name or 'categorical', choices)
                else:
                    # Present human-readable string representations to Optuna for storage/output,
                    # but map back to the original Python objects for downstream model building.
                    labels = [repr(c) for c in choices]
                    picked = trial.suggest_categorical(name or 'categorical', labels)
                    idx = labels.index(picked)
                    result = choices[idx]
            elif isinstance(dim, StructDimension):
                d = {k: sample(join(name, k), child) for k, child in dim.fields.items()}
                if dim.struct_type:
                    d['type'] = dim.struct_type
                result = d
            elif isinstance(dim, ChoiceDimension):
                labels = list(dim.options.keys())
                label = trial.suggest_categorical(join(name, 'choice') if name else 'choice', labels)
                result = sample(join(name, label), dim.options[label])
            elif isinstance(dim, ListDimension):
                items = getattr(dim, 'items', None) or []
                d = {}
                for i, item in enumerate(items):
                    label = f"{dim.label}_{i}" if dim.label else f"item_{i}"
                    d[label] = sample(index(name, label), item)
                result = d
            else:
                raise TypeError(f"Unsupported dimension type at {name or '<root>'}: {type(dim)}")

            return result

        result = sample('', dimensions)
        if not isinstance(result, dict):
            raise TypeError("Root StructDimension must sample to a dict of parameters")
        return result
