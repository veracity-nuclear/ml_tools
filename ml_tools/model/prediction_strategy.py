from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Type
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py

from ml_tools.model.state import StateSeries, SeriesCollection
from ml_tools.model.feature_processor import FeatureProcessor, write_feature_processor, read_feature_processor


class PredictionStrategy(ABC):
    """ An abstract class for prediction strategies. Not meant to be instantiated directly.

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

    @property
    def predicted_feature(self) -> str:
        return self._predicted_feature

    @predicted_feature.setter
    def predicted_feature(self, predicted_feature: str) -> None:
        assert predicted_feature not in self.input_features, f"'{predicted_feature}' is also an input feature"
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
    def train(self, train_data: SeriesCollection, test_data: Optional[SeriesCollection] = None, num_procs: int = 1) -> None:
        """ The method that trains the prediction strategy given a set of training data and testing data

        Parameters
        ----------
        train_data : SeriesCollection
            The state series to use for training
        test_data : SeriesCollection
            The state series to use for testing the trained model
            (NOTE: not all prediction strategies will require providing training / testing data as part of training)
        num_procs : int
            The number of parallel processors to use when training
        """


    def preprocess_inputs(self, state_series: SeriesCollection, num_procs: int = 1) -> np.ndarray:
        """ Preprocesses all the input state series into the series of processed input features of the model

        Parameters
        ----------
        state_series : SeriesCollection
            The list of state series to be pre-processed into a collection of model input series
        num_procs : int
            The number of parallel processors to use when processing the data

        Returns
        -------
        np.ndarray
            The preprocessed collection of state series input features. All series are padded
            with 0.0 to match the length of the longest series.
        """
        input_features = [self.input_features] * len(state_series) # input_features for each worker

        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            processed_inputs = list(executor.map(self._process_single_series, state_series, input_features))

        return self._pad_series(processed_inputs)


    @staticmethod
    def _process_single_series(series: StateSeries, input_features: Dict[str, FeatureProcessor]) -> np.ndarray:
        """ Helper method to support parallelizing preprocess_inputs

        This is required to be a separate method so it can be pickled by ProcessPoolExecutor
        """

        processed_inputs = []
        for feature, processor in input_features.items():
            feature_data = np.array([state[feature] for state in series])
            processed_data = processor.preprocess(feature_data)
            processed_inputs.append(np.vstack(processed_data))
        return np.hstack(processed_inputs)

    @staticmethod
    def _pad_series(state_series: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
        """ Pads state series data to have the same number of timesteps

        Parameters
        ----------
        sequences : List[np.ndarray]
            List of state series arrays (timesteps_i, num_features)
        pad_value : float
            The value used for padding (Default: 0.0)

        Returns
        -------
        np.ndarray
            A padded array of shape (num_series, max_timesteps, num_features)
        """

        max_len = max(series.shape[0] for series in state_series)
        num_features = state_series[0].shape[1]

        padded = np.full((len(state_series), max_len, num_features), pad_value, dtype=np.float32)
        for i, series in enumerate(state_series):
            padded[i, :series.shape[0], :] = series

        return padded

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

        h5_file.create_dataset('predicted_feature', data=self.predicted_feature)
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


    @classmethod
    @abstractmethod
    def read_from_file(cls, file_name: str) -> Type[PredictionStrategy]:
        """ A method for loading a trained model from a file

        Parameters
        ----------
        file_name : str
            The name of the file to load the model from
        """
        raise NotImplementedError


    def predict(self, state_series: SeriesCollection) -> np.ndarray:
        """ The method that approximates the predicted features corresponding to a list of state series

        Parameters
        ----------
        state_series : SeriesCollection
            The input state series which to predict outputs for

        Returns
        -------
        np.ndarray
            Predicted feature values for each state in each series, including any
            padded timesteps. Users should take care to mask or ignore predictions
            at padded positions, as they do not correspond to real input data.
        """
        assert self.isTrained

        y = self._predict_all(self.preprocess_inputs(state_series))

        if self.hasBiasingModel:
            y += self.biasing_model.predict(state_series)
        return y


    def _predict_all(self, state_series: np.ndarray) -> np.ndarray:
        """ The method that predicts the target values corresponding to the given state series

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Parameters
        ----------
        state_series : np.ndarray
            The input state series at which to predict the target value

        Returns
        -------
        np.ndarray
            The predicted target values for each state in each series
        """
        return np.asarray([self._predict_one(series) for series in state_series])


    def _get_targets(self, state_series: SeriesCollection) -> np.ndarray:
        """ The method that extracts the target values for each state for training

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Parameters
        ----------
        state_series : SeriesCollection
            The state series to extract / derive the target values from

        Returns
        -------
        np.ndarray
            The target values of each state of each series to use in training
        """

        targets = np.array([[state[self.predicted_feature] for state in series] for series in state_series])
        if self.hasBiasingModel:
            bias = np.asarray(self.biasing_model.predict(state_series))
            targets -= bias
        return targets


    @abstractmethod
    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        """ The method that predicts the target values corresponding to the given state series

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Parameters
        ----------
        state_series : np.ndarray
            The input state series for which to predict the target values

        Returns
        -------
        np.ndarray
            The predicted target values for each state of the series
        """
