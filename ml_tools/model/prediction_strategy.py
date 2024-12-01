from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Type, TypeVar
import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor
from ml_tools.model.state import State, StateSeries
from ml_tools.model.feature_processor import FeatureProcessor, write_feature_processor, read_feature_processor


class PredictionStrategy(ABC):
    """ An abstract class for prediction strategies

    Attributes
    ----------
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding feature processing strategy
    predicted_feature : str
        The string specifying the feature to be predicted
    biasing_model : PredictionStrategy
        A model that it is used to provide an initial prediction of the predicted output, acting ultimately as an initial bias
    hasBiasingModel : bool
        Whether or not this model has a bias model
    isTrained : bool
        Whether or not this model has been trained
    """

    @property
    def input_features(self) -> Dict[str, FeatureProcessor]:
        return self._input_features

    @input_features.setter
    def input_features(self, input_features: Dict[str, FeatureProcessor]) -> None:
        self._input_features = {}
        for feature, processor in input_features.items():
            assert feature is not self.predicted_feature, f"'{feature}' is also the predicted feature"
            self._input_features[feature] = processor
        self._input_features = dict(sorted(self._input_features.items()))  # Ensure input features are in alphabetical order

    @property
    def predicted_feature(self) -> str:
        return self._predicted_feature

    @predicted_feature.setter
    def predicted_feature(self, predicted_feature: str) -> None:
        assert predicted_feature not in self.input_features.keys(), f"'{feature}' is also an input feature"
        self._predicted_feature = predicted_feature

    @property
    def biasing_model(self) -> PredictionStrategy:
        return self._biasing_model

    @biasing_model.setter
    def biasing_model(self, bias: PredictionStrategy) -> None:
        assert bias.isTrained
        self._biasing_model = bias

    @property
    def hasBiasingModel(self) -> bool:
        return self._biasing_model is not None

    @property
    @abstractmethod
    def isTrained(self) -> bool:
        pass


    def __init__(self):
        self._predicted_feature = None
        self._biasing_model     = None
        self._input_features    = {}


    @abstractmethod
    def train(self, train_data: List[StateSeries], test_data: List[StateSeries] = [], num_procs: int = 1) -> None:
        """ The method that trains the prediction strategy given a set of training data and testing data

        Parameters
        ----------
        train_data : List[StateSeries]
            The state series to use for training
        test_data : List[StateSeries]
            The state series to use for testing the trained model
            (NOTE: not all prediction strategies will require providing training / testing data as part of training)
        num_procs : int
            The number of parallel processors to use when training
        """
        pass


    def preprocess_inputs(self, state_series: List[StateSeries], num_procs: int = 1) -> np.ndarray:
        """ Preprocesses all the input state series into the series of processed input features of the model

        Parameters
        ----------
        state_series : List[StateSeries]
            The list of state series to be pre-processed into a collection of model input series
        num_procs : int
            The number of parallel processors to use when processing the data

        Returns
        -------
        np.ndarray
            The preprocessed collection of state series input features
        """

        processed_inputs = []

        for series in state_series:
            series_inputs = []
            for feature, processor in self.input_features.items():
                feature_data = [state.feature(feature) for state in series]
                feature_data = np.array_split(feature_data, num_procs)
                with ProcessPoolExecutor(max_workers=num_procs) as executor:
                    processed_data = list(executor.map(processor.preprocess, feature_data))
                series_inputs.append(np.vstack(processed_data))
            processed_inputs.append(np.hstack(series_inputs).tolist())

        return np.asarray(processed_inputs)


    def base_save_model(self, h5_file: h5py.File) -> None:
        """ A method for saving base-class data for a trained model

        Parameters
        ----------
        h5_file : h5py.File
            An opened, writeable HDF5 file handle
        """
        #TODO: need to handle biasing
        if self._biasing_model is not None:
            raise AttributeError('Cannot save model with bias model attached')

        h5_file.create_dataset('predicted_feature',     data=self.predicted_feature)
        input_features_group = h5_file.create_group('input_features')
        for name, feature in self.input_features.items():
            write_feature_processor(input_features_group.create_group(name), feature)


    def base_load_model(self, h5_file: h5py.File) -> None:
        """ A method for loading base-class data for a trained model

        Parameters
        ----------
        h5_file : h5py.File
            An opened HDF5 file handle
        """
        self.predicted_feature = h5_file['predicted_feature'][()].decode('utf-8')
        input_features = {}
        for name, feature in h5_file['input_features'].items():
            input_features[name] = read_feature_processor(feature)
        self.input_features = input_features


    def predict(self, state_series: List[StateSeries]) -> List[List[float]]:
        """ The method that approximates the predicted features corresponding to a list of state series

        Parameters
        ----------
        state_series : List[StateSeries]
            The input state series which to predict outputs for

        Returns
        -------
        List[List[float]]
            The predicted features for each state in each series
        """
        assert self.isTrained

        y = np.asarray(self._predict_all(state_series))
        if self.hasBiasingModel:
            y += np.asarray(self.biasing_model.predict(state_series))
        return y.tolist()


    def _predict_all(self, state_series: List[StateSeries]) -> np.ndarray:
        """ The method that predicts the target values corresponding to the given state series

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Parameters
        ----------
        state_series : List[StateSeries]
            The input state series at which to predict the target value

        Returns
        -------
        np.ndarray
            The predicted target values for each state in each series
        """
        return [self._predict_one(series) for series in state_series]


    def _get_targets(self, state_series: List[StateSeries]) -> np.ndarray:
        """ The method that extracts the target values for each state for training

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Parameters
        ----------
        state_series : List[StateSeries]
            The state series to extract / derive the target values from

        Returns
        -------
        np.ndarray
            The target values of each state of each series to use in training
        """

        targets = np.array([[state.feature(self.predicted_feature) for state in series] for series in state_series])
        if self.hasBiasingModel:
            bias = np.asarray(self.biasing_model.predict(state_series))
            targets -= bias
        return targets


    @abstractmethod
    def _predict_one(self, state_series: StateSeries) -> np.ndarray:
        """ The method that predicts the target values corresponding to the given state series

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Parameters
        ----------
        state_series : StateSeries
            The input state series for which to predict the target values

        Returns
        -------
        np.ndarray
            The predicted target values for each state of the series
        """
        pass