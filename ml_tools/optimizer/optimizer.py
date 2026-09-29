from typing import Optional

from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.optimizer.search_space import SearchSpace
from ml_tools.optimizer.search_strategy import SearchStrategy

class Optimizer():
    """ A class for performing model hyperparameter optimization

    Parameters
    ----------
    search_space : SearchSpace
        The hyperparameter search space to explore
    search_strategy : SearchStrategy
        The search strategy to use when exploring the search space

    Attributes
    ----------
    search_space : SearchSpace
        The hyperparameter search space to explore
    search_strategy : SearchStrategy
        The search strategy to use when exploring the search space
    """

    @property
    def search_space(self) -> SearchSpace:
        return self._search_space

    @search_space.setter
    def search_space(self, value: SearchSpace) -> None:
        self._search_space = value

    @property
    def search_strategy(self) -> SearchStrategy:
        return self._search_strategy

    @search_strategy.setter
    def search_strategy(self, value: SearchStrategy) -> None:
        self._search_strategy = value


    def __init__(self,
                 search_space:      SearchSpace,
                 search_strategy:   SearchStrategy,) -> None:
        self.search_space      = search_space
        self.search_strategy   = search_strategy


    def optimize(self,
                 train_data:         SeriesCollection,
                 num_trials:        int = 10,
                 number_of_folds:   int = 5,
                 output_file:       str = "optimization_results.out",
                 checkpoint_dir:    Optional[str] = None,
                 resume:            bool = False,
                 save_every_n_trials: int = 0,
                 num_procs:         int = 1,
                 train_best_model:  bool = True,
                 *,
                 validation_data:   Optional[SeriesCollection] = None,
                 validation_split:  float = 0.2,
                 validation_seed:   Optional[int] = 42) -> PredictionStrategy:
        """ Method for performing model hyperparameter optimization

        Parameters
        ----------
        train_data : SeriesCollection
            Collection used for cross-validation training and scoring.
        num_trials : int
            The number of hyperparameter trials to perform (Default is 10)
        number_of_folds : int
            The number of folds to use in cross-validation (Default is 5)
        output_file : str
            The file to which optimization results are written (Default is "optimization_results.out")
        checkpoint_dir : Optional[str]
            Directory to write checkpoint artifacts (study DB, JSON snapshots).
        resume : bool
            Whether to resume from an existing study/checkpoint when available.
        save_every_n_trials : int
            Frequency (in trials) to dump lightweight checkpoints; 0 disables.
        num_procs : int
            The number of processes to use for parallel model training (Default is 1)
        train_best_model : bool
            Whether to train the best model using train_data and the configured
            validation policy before returning it. Default is True.
        validation_data : SeriesCollection, optional
            Explicit validation collection supplied to models during training.
            This collection is not used for outer-fold scoring.
        validation_split : float
            Fraction of each training collection reserved for validation when
            validation_data is omitted and the strategy requires validation.
        validation_seed : int, optional
            Random seed used for automatic validation splits.

        Returns
        -------
        PredictionStrategy
            The best model found during optimization. The model is trained only
            when ``train_best_model`` is True.
        """

        best_model = self.search_strategy.search(search_space      = self.search_space,
                                                 train_data         = train_data,
                                                 num_trials        = num_trials,
                                                 number_of_folds   = number_of_folds,
                                                 output_file       = output_file,
                                                 checkpoint_dir    = checkpoint_dir,
                                                 resume            = resume,
                                                 save_every_n_trials = save_every_n_trials,
                                                 num_procs         = num_procs,
                                                 validation_data   = validation_data,
                                                 validation_split  = validation_split,
                                                 validation_seed   = validation_seed)

        if train_best_model:
            best_model.train(train_data,
                             validation_data=validation_data,
                             num_procs=num_procs,
                             validation_split=validation_split,
                             validation_seed=validation_seed)

        return best_model
