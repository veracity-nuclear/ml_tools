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
                 series_collection: SeriesCollection,
                 num_trials:        int = 10,
                 number_of_folds:   int = 5,
                 output_file:       str = "optimization_results.out",
                 num_procs:         int = 1) -> PredictionStrategy:
        """ Method for performing model hyperparameter optimization

        Parameters
        ----------
        series_collection : SeriesCollection
            The collection of series to use for training and validation
        num_trials : int
            The number of hyperparameter trials to perform (Default is 10)
        number_of_folds : int
            The number of folds to use in cross-validation (Default is 5)
        output_file : str
            The file to which optimization results are written (Default is "optimization_results.out")
        num_procs : int
            The number of processes to use for parallel model training (Default is 1)

        Returns
        -------
        PredictionStrategy
            The best model found during optimization
        """

        return self.search_strategy.search(search_space      = self.search_space,
                                           series_collection = series_collection,
                                           num_trials        = num_trials,
                                           number_of_folds   = number_of_folds,
                                           output_file       = output_file,
                                           num_procs         = num_procs)
