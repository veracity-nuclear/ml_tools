from typing import Dict, Any, Optional
from math import log, ceil, floor
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
               checkpoint_dir:    Optional[str] = None,
               resume:            bool = False,
               save_every_n_trials: int = 0,
               study_storage:     Optional[str] = None,
               num_procs:         int = 1,
               num_fold_workers:  int = 1,
               num_jobs:          int = 1) -> PredictionStrategy:

        super().search(search_space,
                       series_collection,
                       num_trials,
                       number_of_folds,
                       output_file,
                       checkpoint_dir,
                       resume,
                       save_every_n_trials,
                       study_storage,
                       num_procs,
                       num_fold_workers,
                       num_jobs)

        checkpoint_path = None
        storage_uri = study_storage
        if checkpoint_dir:
            checkpoint_root = Path(checkpoint_dir)
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_root / "optuna_checkpoint.json"
            if storage_uri is None:
                storage_uri = f"sqlite:///{checkpoint_root / 'optuna_study.db'}"
        load_if_exists = resume or storage_uri is not None
        if load_if_exists and storage_uri is None:
            # Resume only works with persistent storage; fall back to in-memory study.
            load_if_exists = False

        def _dump_checkpoint(study_obj: optuna.Study) -> None:
            if not checkpoint_path:
                return
            try:
                best_params = study_obj.best_params
                best_value = study_obj.best_value
            except ValueError:
                best_params = {}
                best_value = None

            payload = {
                "n_trials": len(study_obj.trials),
                "best_params": best_params,
                "best_value": best_value,
            }
            checkpoint_path.write_text(json.dumps(payload, indent=2))

        def log_progress(_study, trial):
            with open(output_file, "a") as output:
                output.write(f"Trial {trial.number}: Params: {trial.params}, Value: {trial.value}\n")
            print(f"Trial {trial.number} completed: Value: {trial.value}")
            if save_every_n_trials and ((trial.number + 1) % save_every_n_trials == 0):
                _dump_checkpoint(_study)

        print("Starting NN Optimization")
        with open(output_file, 'w') as output:
            output.write("RESULTS\n---------\n")

        # Avoid nested parallelism - priority: num_jobs > num_fold_workers > num_procs
        # Only one level of parallelism should be active at a time
        effective_fold_workers = num_fold_workers
        effective_num_procs = num_procs

        if num_jobs and num_jobs > 1:
            # Trial-level parallelism takes priority - disable fold and training parallelism
            if num_fold_workers > 1:
                print(f"num_jobs={num_jobs} > 1, forcing num_fold_workers=1 to avoid nested parallelism")
                effective_fold_workers = 1
            if num_procs > 1:
                print(f"num_jobs={num_jobs} > 1, forcing num_procs=1 to avoid nested parallelism")
                effective_num_procs = 1
        elif num_fold_workers > 1:
            # Fold-level parallelism takes priority - disable training parallelism
            if num_procs > 1:
                print(f"num_fold_workers={num_fold_workers} > 1, forcing num_procs=1 to avoid nested parallelism")
                effective_num_procs = 1

        study     = optuna.create_study(direction='minimize',
                                        storage=storage_uri,
                                        study_name="optuna_study",
                                        load_if_exists=load_if_exists)
        objective = self._setup_objective(search_space,
                                          series_collection,
                                          number_of_folds,
                                          effective_fold_workers,
                                          effective_num_procs)

        study.optimize(objective,
                   n_trials=num_trials,
                   callbacks=[log_progress],
                   n_jobs=num_jobs)

        with open(output_file, 'a') as output:
            output.write("\nBEST PARAMETERS\n----------------\n")
            output.write(f"{study.best_params}\n")

        _dump_checkpoint(study)

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
                         num_fold_workers:  int,
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
        num_fold_workers : int
            Max workers for evaluating CV folds in parallel; 1 keeps sequential.
        num_procs : int
            The number of processes to use for parallel model training
        """

        def objective(trial: optuna.trial.Trial) -> float:
            params = self._get_sample(trial, search_space.dimensions)
            # Use num_procs passed to this function (already accounts for parallelism priority)
            train_procs = num_procs

            def evaluate_fold(fold_split):
                fold, (train_idx, val_idx) = fold_split
                print(f"Starting fold {fold}/{number_of_folds}...")

                fold_training_set   = SeriesCollection([series_collection[i] for i in train_idx])
                fold_validation_set = SeriesCollection([series_collection[i] for i in val_idx])

                fold_model = build_prediction_strategy(strategy_type      = search_space.prediction_strategy_type,
                                                       params             = params,
                                                       input_features     = search_space.input_features,
                                                       predicted_features = search_space.predicted_features,
                                                       biasing_model      = search_space.biasing_model)

                print(f"Fold {fold}: training start (num_procs={train_procs})")
                fold_model.train(fold_training_set, num_procs=train_procs)
                print(f"Fold {fold}: training complete")

                feature_order = list(search_space.predicted_features)
                measured_rows = []
                for series in fold_validation_set:
                    parts = []
                    for name in feature_order:
                        v = np.asarray(series[0][name], dtype=float)
                        v = np.atleast_1d(v).reshape(-1)
                        parts.append(v)
                    measured_rows.append(np.concatenate(parts, axis=0))
                measured = np.vstack(measured_rows)

                predicted_rows = []
                print(f"Fold {fold}: predicting start")
                for series in fold_model.predict(fold_validation_set):
                    parts = []
                    for name in feature_order:
                        v = np.asarray(series[0][name], dtype=float)
                        v = np.atleast_1d(v).reshape(-1)
                        parts.append(v)
                    predicted_rows.append(np.concatenate(parts, axis=0))
                print(f"Fold {fold}: predicting complete")
                predicted = np.vstack(predicted_rows)

                diff = measured - predicted
                fold_rms = np.sqrt(np.mean(np.square(diff, dtype=float)))
                print(f"Fold {fold}: rms={fold_rms}")
                return fold_rms

            splits = list(enumerate(KFold(n_splits=number_of_folds, shuffle=True).split(series_collection), start=1))
            if num_fold_workers > 1:
                with ThreadPoolExecutor(max_workers=num_fold_workers) as executor:
                    rms = list(executor.map(evaluate_fold, splits))
            else:
                rms = [evaluate_fold(split) for split in splits]

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
