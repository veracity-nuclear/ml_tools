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
               checkpoint_dir:    Optional[str] = None,
               resume:            bool = False,
               save_every_n_trials: int = 0,
               num_procs:         int = 1) -> PredictionStrategy:
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
        checkpoint_dir : Optional[str]
            Directory to write checkpoint artifacts (study DB, JSON snapshots).
        resume : bool
            Whether to resume from an existing study/checkpoint when available.
        save_every_n_trials : int
            Frequency (in trials) to dump lightweight checkpoints; 0 disables.
        num_procs : int
            The number of processes to use for parallel model training

        Returns
        -------
        PredictionStrategy
            The best model configuration found during optimization. Search
            strategies do not train the final model on the full dataset.
        """

        assert num_trials > 0, f"num_trials = {num_trials}"
        assert number_of_folds > 1, f"number_of_folds = {number_of_folds}"
        assert save_every_n_trials >= 0, f"save_every_n_trials = {save_every_n_trials}"
        assert num_procs > 0, f"num_procs = {num_procs}"
