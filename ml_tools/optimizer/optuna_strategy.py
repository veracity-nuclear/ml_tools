from typing import Dict, Optional

import optuna
import numpy as np
from sklearn.model_selection import KFold

from ml_tools.data.series_collection import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.optimizer.search_strategy import SearchStrategy
from ml_tools.optimizer.search_space import SearchSpace, \
    Int         as IntDimension, \
    Float       as FloatDimension, \
    Categorical as CategoricalDimension, \
    Bool        as BoolDimension, \
    Struct      as StructDimension, \
    Choice      as ChoiceDimension, \
    ListDim     as ListDimDimension

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

        def log_progress(study, trial):
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

        print(f"Best parameters: {study.best_params}")

        best_model = None

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
            model = PredictionStrategy.from_dict(strategy_type     = search_space.prediction_strategy_type,
                                                 dict              = self._get_parameters(trial, search_space.dimensions),
                                                 input_features    = search_space.input_features,
                                                 predicted_feature = search_space.predicted_feature,
                                                 biasing_model     = search_space.biasing_model)

            rms = []
            kf = KFold(n_splits=number_of_folds, shuffle=True)
            for fold, (train_idx, val_idx) in enumerate(kf.split(series_collection)):
                print(f"Starting fold {fold + 1}/{number_of_folds}...")

                training_set   = [series_collection[i] for i in train_idx]
                validation_set = [series_collection[i] for i in val_idx]

                print("Training started...")
                model.train(training_set, num_procs=num_procs)
                print("Training completed.")

                print("Validating model...")
                measured  = np.asarray([[series[0][search_space.predicted_feature]] for series in validation_set])
                predicted = model.predict(validation_set)
                diff      = measured - predicted
                fold_rms  = np.sqrt(np.dot(diff.flatten(), diff.flatten())) / float(len(diff))
                print(f"Fold {fold + 1} RMS: {fold_rms}")

                rms.append(fold_rms)

            average_rms = sum(rms) / len(rms)
            print(f"Optimization step completed. Average RMS across folds: {average_rms}")
            return average_rms

        return objective


    def _get_parameters(self, trial: optuna.trial.Trial, dimensions: StructDimension) -> Dict:
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
            return f"{parent}[{i}]" if parent else f"[{i}]"

        def sample(name: str, dim) -> Dict:
            if isinstance(dim, IntDimension):
                log = bool(getattr(dim, 'log', False))
                return {name: trial.suggest_int(name or 'value', dim.low, dim.high, log=log)}
            if isinstance(dim, FloatDimension):
                log = bool(getattr(dim, 'log', False))
                return {name: trial.suggest_float(name or 'value', dim.low, dim.high, log=log)}
            if isinstance(dim, BoolDimension):
                return {name: trial.suggest_categorical(name or 'value', [False, True])}
            if isinstance(dim, CategoricalDimension):
                return {name: trial.suggest_categorical(name or 'value', dim.choices)}
            if isinstance(dim, StructDimension):
                return {k: sample(join(name, k), child) for k, child in dim.fields.items()}
            if isinstance(dim, ChoiceDimension):
                labels = list(dim.options.keys())
                label = trial.suggest_categorical(join(name, 'type') if name else 'type', labels)
                value = sample(join(name, label), dim.options[label])
                return {label: value}
            if isinstance(dim, ListDimDimension):
                items = getattr(dim, 'items', None)
                return {index(name, i): sample('', d) for i, d in enumerate(items)}

            raise TypeError(f"Unsupported dimension type at {name or '<root>'}: {type(dim)}")

        result = sample('', dimensions)
        return result if isinstance(result, dict) else {"value": result}
