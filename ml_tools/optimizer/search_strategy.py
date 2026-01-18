from abc import ABC, abstractmethod
from typing import Optional

from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.optimizer.search_space import SearchSpace

class SearchStrategy(ABC):
    """ Abstract base class for search strategies used in hyperparameter optimization.
    """

    @abstractmethod
    def search(self,
               search_space:      SearchSpace,
               series_collection: SeriesCollection,
               num_trials:        int,
               number_of_folds:   int,
               output_file:       str,
               num_procs:         int,
               checkpoint_dir:    Optional[str] = None,
               resume:            bool = False,
               save_every_n_trials: int = 0,
               num_fold_workers:  int = 1,
               study_storage:     Optional[str] = None,
               num_jobs:          int = 1) -> PredictionStrategy:
        """ Method for performing model hyperparameter optimization

        Parameters
        ----------
        search_space : SearchSpace
            The hyperparameter search space to explore
        series_collection : SeriesCollection
            The collection of series to use for training and validation
        num_trials : int
            The number of hyperparameter trials to perform
        number_of_folds : int
            The number of folds to use in cross-validation
        output_file : str
            The file to which optimization results are written
        num_procs : int
            The number of processes to use for parallel model training
        checkpoint_dir : Optional[str]
            Directory to write checkpoint artifacts (study DB, JSON snapshots).
        resume : bool
            Whether to resume from an existing study/checkpoint when available.
        save_every_n_trials : int
            Frequency (in trials) to dump lightweight checkpoints; 0 disables.
        num_fold_workers : int
            Max workers for evaluating CV folds in parallel; 1 keeps sequential.
        study_storage : Optional[str]
            Optuna storage URI (e.g., sqlite:///optuna.db); inferred when checkpoint_dir is set.
        num_jobs : int
            Number of parallel workers per process when the backend supports it (Optuna's num_jobs).

        Returns
        -------
        PredictionStrategy
            The best model found during optimization
        """

        assert num_trials > 0, f"num_trials = {num_trials}"
        assert number_of_folds > 1, f"number_of_folds = {number_of_folds}"
        assert num_procs > 0, f"num_procs = {num_procs}"
        assert save_every_n_trials >= 0, f"save_every_n_trials = {save_every_n_trials}"
        assert num_fold_workers > 0, f"num_fold_workers = {num_fold_workers}"
        assert num_jobs >= 1, f"num_jobs = {num_jobs}"
