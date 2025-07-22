from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from ml_tools import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureProcessor
from ml_tools.model.nn_strategy import Activation


class Optimizer(ABC):
    """ An abstract class for model optimizers

    Attributes
    ----------
    dimensions : Dimensions
        The degrees of freedom and bounds for the optimization search
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding feature processing strategy
    predicted_feature : str
        The string specifying the feature to be predicted
    series_collection : SeriesCollection
        The input state series which to predict outputs for
    num_procs : int
        The number of parallel processors to use when reading data from the HDF5
    test_fraction : float
        The fraction of training data to withold for testing
    number_of_folds : int
        The number of folds to use for cross-fold validation
    epoch_limit : int
        The limit on the number of training epochs conducted during training
    convergence_criteria : float
        The convergence criteria for training
    convergence_patience : int
        Number of epochs with no improvement (i.e. error improves by greater than the convergence_criteria)
        after which training will be stopped
    biasing_model : Optional[PredictionStrategy]
        A model that is used to provide an initial prediction of the predicted output, acting ultimately as an initial bias
    """

    @dataclass
    class Dimensions():
        """ A data class for the dimensions and bounds of the model hyperparameter search space
        """

        @dataclass
        class Layer():
            """ A data class for dimensions and bounds for the dense layer's hyperparameter search space
            """

            neurons:    Tuple[int, int]      # Min and Max number neurons for this layer
            activation: List[Activation]     # List of activiation functions to consider for this dense layer
            dropout:    Tuple[float, float]  # Min and Max dropout rates for this layer

            def __post_init__(self):
                assert all(value > 0 for value in self.neurons), f"'neurons = {neurons}'"
                assert all(0.0 <= value <= 1.0 for value in self.dropout), f"dropout = {dropout}"
                assert self.neurons[0] <= self.neurons[1], f"'neurons = {neurons}'"
                assert self.dropout[0] <= self.dropout[1], f"'dropout = {dropout}'"

        initial_learning_rate: Tuple[float, float]  # Min and Max initial learning rates
        learning_decay_rate:   Tuple[float, float]  # Min and Max learning decay rates
        batch_size_log2:       Tuple[int, int]      # Min and Max training batch sizes
        num_dens_layers:       Tuple[int, int]      # Min and Max number of dense layers
        dens_layers:           List[Layer]          # List of dense layers to consider

        def __post_init__(self):
            assert all(value > 0. for value in self.initial_learning_rate), \
                f"initial_learning_rate = {self.initial_learning_rate}"
            assert all(value >= 0. for value in self.learning_decay_rate), \
                f"learning_decay_rate = {self.learning_decay_rate}"
            assert all(value >= 0 for value in self.batch_size_log2), \
                f"batch_size_log2 = {self.batch_size_log2}"
            assert len(self.dens_layers) > 0, \
                f"len(dens_layers) = {len(self.dens_layers)}"
            assert all(len(self.dens_layers) >= value >= 1 for value in self.num_dens_layers), \
                f"num_dens_layers = {self.num_dens_layers}"

            assert self.initial_learning_rate[0] <= self.initial_learning_rate[1], \
                f"initial_learning_rate = {self.initial_learning_rate}"
            assert self.learning_decay_rate[0] <= self.learning_decay_rate[1], \
                f"learning_decay_rate = {self.learning_decay_rate}"
            assert self.batch_size_log2[0] <= self.batch_size_log2[1], \
                f"batch_size_log2 = {self.batch_size_log2}"
            assert self.num_dens_layers[0] <= self.num_dens_layers[1], \
                f"num_dens_layers = {self.num_dens_layers}"


    @property
    def dimensions(self) -> Dimensions:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: Dimensions) -> None:
        self._dimensions = dimensions

    @property
    def input_features(self) -> Dict[str, FeatureProcessor]:
        return self._input_features

    @input_features.setter
    def input_features(self, input_features: Dict[str, FeatureProcessor]) -> None:
        self._input_features = input_features

    @property
    def predicted_feature(self) -> str:
        return self._predicted_feature

    @predicted_feature.setter
    def predicted_feature(self, predicted_feature: str) -> None:
        assert predicted_feature not in self.input_features, f"'{predicted_feature}' is also an input feature"
        self._predicted_feature = predicted_feature

    @property
    def series_collection(self) -> SeriesCollection:
        return self._series_collection

    @series_collection.setter
    def series_collection(self, series_collection: SeriesCollection) -> None:
        assert len(series_collection) > 0, f"len(series_collection) = {len(series_collection)}"
        self._series_collection = series_collection

    @property
    def num_procs(self) -> int:
        return self._num_procs

    @num_procs.setter
    def num_procs(self, num_procs: int) -> None:
        assert num_procs > 0, f"num_procs = {num_procs}"
        self._num_procs = num_procs

    @property
    def test_fraction(self) -> float:
        return self._test_fraction

    @test_fraction.setter
    def test_fraction(self, test_fraction: float) -> None:
        assert test_fraction >= 0.0
        self._test_fraction = test_fraction

    @property
    def number_of_folds(self) -> int:
        return self._number_of_folds

    @number_of_folds.setter
    def number_of_folds(self, number_of_folds: int) -> None:
        assert number_of_folds > 0
        self._number_of_folds = number_of_folds

    @property
    def epoch_limit(self) -> int:
        return self._epoch_limit

    @epoch_limit.setter
    def epoch_limit(self, epoch_limit: int) -> None:
        assert epoch_limit > 0
        self._epoch_limit = epoch_limit

    @property
    def convergence_criteria(self) -> float:
        return self._convergence_criteria

    @convergence_criteria.setter
    def convergence_criteria(self, convergence_criteria: float) -> None:
        assert convergence_criteria >= 0.0
        self._convergence_criteria = convergence_criteria

    @property
    def convergence_patience(self) -> int:
        return self._convergence_patience

    @convergence_patience.setter
    def convergence_patience(self, convergence_patience: int) -> None:
        assert convergence_patience > 0
        self._convergence_patience = convergence_patience

    @property
    def biasing_model(self) -> Optional[PredictionStrategy]:
        return self._biasing_model

    @biasing_model.setter
    def biasing_model(self, biasing_model: Optional[PredictionStrategy]) -> None:
        self._biasing_model = biasing_model


    @abstractmethod
    def optimize(self, num_trials: int, output_file: str) -> PredictionStrategy:
        """ Method for performing model hyperparameter optimization

        Parameters
        ----------
        num_trials : int
            Number of optimization trials to conduct
        output_file : str
            The output file to write optimization outputs to

        Returns
        -------
        PredictionStrategy
            The optimized model
        """
