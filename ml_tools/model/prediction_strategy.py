from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Type, Union, Sequence
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py

from ml_tools.model.state import State, StateSeries, SeriesCollection
from ml_tools.model.feature_processor import (
    FeatureProcessor,
    NoProcessing,
    read_feature_processor,
    write_feature_processor,
)

FeatureSpec = Union[Dict[str, FeatureProcessor], str, Sequence[str]]


class PredictionStrategy(ABC):
    """ An abstract class for prediction strategies. Not meant to be instantiated directly.

    Attributes
    ----------
    input_features : Dict[str, FeatureProcessor]
        Input feature and processor pairs, keyed by feature name.
    predicted_features : Dict[str, FeatureProcessor]
        Output feature and processor pairs, keyed by feature name.
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
    def input_features(self, input_features: FeatureSpec) -> None:
        input_features = self.create_feature_processor_map(input_features)
        self._input_features = {}
        for feature, processor in input_features.items():
            assert feature not in self._predicted_features, f"'{feature}' is also a predicted feature"
            self._input_features[feature] = processor

    @property
    def predicted_features(self) -> Dict[str, FeatureProcessor]:
        return self._predicted_features

    @predicted_features.setter
    def predicted_features(self, predicted_features: FeatureSpec) -> None:
        predicted_features = self.create_feature_processor_map(predicted_features)
        assert len(predicted_features) > 0, "predicted_features must be non-empty"
        self._predicted_features = {}
        for feature, processor in predicted_features.items():
            assert isinstance(feature, str) and feature, "predicted feature keys must be non-empty strings"
            assert isinstance(processor, FeatureProcessor), f"Invalid processor for '{feature}'"
            assert feature not in self.input_features, f"'{feature}' is also an input feature"
            self._predicted_features[feature] = processor
        self._predicted_feature_order = list(predicted_features.keys())
        self._predicted_feature_sizes = None

    @property
    def predicted_feature_names(self) -> List[str]:
        return list(self._predicted_feature_order)

    @property
    def biasing_model(self) -> PredictionStrategy:
        return self._biasing_model

    @biasing_model.setter
    def biasing_model(self, bias: PredictionStrategy) -> None:
        assert bias.isTrained
        if self._predicted_features:
            assert set(bias.predicted_features) == set(self.predicted_features), \
                "Biasing model outputs must match predicted features"
            for feature, processor in self.predicted_features.items():
                assert processor == bias.predicted_features[feature], \
                    f"Processor mismatch for predicted feature '{feature}'"
        self._biasing_model = bias

    @property
    def hasBiasingModel(self) -> bool:
        return self._biasing_model is not None

    @property
    @abstractmethod
    def isTrained(self) -> bool:
        pass


    def __init__(self):
        self._predicted_features = {}
        self._predicted_feature_order = []
        self._predicted_feature_sizes = None
        self._biasing_model = None
        self._input_features = {}

    @staticmethod
    def create_feature_processor_map(features: FeatureSpec) -> Dict[str, FeatureProcessor]:
        """Normalize feature specs into a name -> processor mapping.

        Parameters
        ----------
        features : FeatureSpec
            Feature specification in one of three forms:
            - Dict[str, FeatureProcessor]: explicit feature/processor mapping
            - str: a single feature name (mapped to NoProcessing)
            - Sequence[str]: feature names (mapped to NoProcessing)
        Returns
        -------
        Dict[str, FeatureProcessor]
            Normalized feature/processor mapping.

        Raises
        ------
        TypeError
            If features is not a dict, string, or sequence of strings.
        AssertionError
            If entries are invalid.
        """
        if isinstance(features, dict):
            feature_map = features
        elif isinstance(features, str):
            feature_map = {features: NoProcessing()}
        elif isinstance(features, (list, tuple)):
            feature_map = {name: NoProcessing() for name in features}
        else:
            raise TypeError("features must be a dict, string, or list/tuple of strings")

        for name, processor in feature_map.items():
            assert isinstance(name, str) and name, "feature names must be non-empty strings"
            assert isinstance(processor, FeatureProcessor), f"Invalid processor for '{name}'"
        return feature_map


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


    @staticmethod
    def preprocess_features(series_collection: SeriesCollection,
                            features: Dict[str, FeatureProcessor],
                            num_procs: int = 1) -> np.ndarray:
        """ Preprocesses all the input state series into the series of processed features of the model

        Parameters
        ----------
        series_collection : SeriesCollection
            The list of state series to be pre-processed into a collection of model input series
        features : Dict[str, FeatureProcessor]
            Feature/processor pairs specifying what to preprocess and how to do so
        num_procs : int
            The number of parallel processors to use when processing the data

        Returns
        -------
        np.ndarray
            The preprocessed collection of state series input features. All series are padded
            with 0.0 to match the length of the longest series.
        """
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        if num_procs <= 1:
            processed_inputs = [
                PredictionStrategy._preprocess_single_series(series, features)
                for series in series_collection
            ]
        else:
            features = [features] * len(series_collection)  # input_features for each worker
            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                processed_inputs = list(executor.map(PredictionStrategy._preprocess_single_series,
                                                     series_collection,
                                                     features))

        return PredictionStrategy._pad_series(processed_inputs)

    @staticmethod
    def _preprocess_single_series(series: StateSeries, input_features: Dict[str, FeatureProcessor]) -> np.ndarray:
        """ Helper method to support parallelizing preprocess_features

        This is required to be a separate method so it can be pickled by ProcessPoolExecutor
        """

        processed_inputs = []
        for feature, processor in input_features.items():
            feature_data = [state.features[feature] for state in series]
            processed_data = np.asarray(processor.preprocess(feature_data))
            if processed_data.ndim == 1:
                processed_data = processed_data[:, np.newaxis]
            elif processed_data.ndim > 2:
                processed_data = np.vstack(processed_data)
            processed_inputs.append(processed_data)
        return np.hstack(processed_inputs)


    @staticmethod
    def postprocess_features(data_array:     np.ndarray,
                             series_lengths: Optional[List[int]],
                             feature_order:  List[str],
                             feature_sizes:  Dict[str, int],
                             features:       Dict[str, FeatureProcessor],
                             num_procs:      int = 1) -> SeriesCollection:
        """Convert data arrays into post-processed series collections.

        Parameters
        ----------
        data_array : np.ndarray
            Data array of shape (num_series, max_timesteps, output_dim).
        series_lengths : Optional[List[int]]
            True lengths for each series. If not provided, assumes all series
            are full length.
        feature_order : List[str]
            Ordered list of output feature names corresponding to the vectorized array.
        feature_sizes : Dict[str, int]
            Mapping from feature name to output width in the vectorized array.
        features : Dict[str, FeatureProcessor]
            Feature/processor pairs used for post-processing each output feature.
        num_procs : int
            The number of parallel processors to use when post-processing the data.

        Returns
        -------
        SeriesCollection
            Series collection of predicted states, trimmed to each series length.
        """
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        if series_lengths is None:
            series_lengths = [data_array.shape[1]] * data_array.shape[0]

        total_size = sum(feature_sizes[name] for name in feature_order)
        assert data_array.shape[2] == total_size, \
            f"Predicted output dim {data_array.shape[2]} does not match expected {total_size}"

        if num_procs <= 1:
            state_series_list = [
                PredictionStrategy._postprocess_single_series(padded,
                                                              series_len,
                                                              feature_order,
                                                              feature_sizes,
                                                              features)
                for padded, series_len in zip(data_array, series_lengths)
            ]
        else:
            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                state_series_list = list(executor.map(PredictionStrategy._postprocess_single_series,
                                                      data_array,
                                                      series_lengths,
                                                      [feature_order] * len(series_lengths),
                                                      [feature_sizes] * len(series_lengths),
                                                      [features] * len(series_lengths)))

        return SeriesCollection(state_series_list)

    @staticmethod
    def _postprocess_single_series(series_data: np.ndarray,
                                   series_len: int,
                                   feature_order: List[str],
                                   feature_sizes: Dict[str, int],
                                   feature_processors: Dict[str, FeatureProcessor]) -> StateSeries:
        """Helper method to support parallelizing postprocess_features."""
        series_data = series_data[:series_len]
        processed_by_feature: Dict[str, np.ndarray] = {}
        start = 0
        for name in feature_order:
            size = feature_sizes[name]
            chunk = series_data[:, start:start + size]
            processed_by_feature[name] = np.asarray(feature_processors[name].postprocess(chunk))
            start += size

        series_states: List[State] = []
        for i in range(series_len):
            series_states.append(State({name: processed_by_feature[name][i]
                                        for name in feature_order}))
        return StateSeries(series_states)


    @staticmethod
    def _pad_series(series_collection: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
        """ Pads state series data to have the same number of timesteps

        Parameters
        ----------
        series_collection : List[np.ndarray]
            The state series collection to be padded
        pad_value : float
            The value used for padding (Default: 0.0)

        Returns
        -------
        np.ndarray
            A padded array of shape (num_series, max_timesteps, num_features)
        """

        max_len = max(series.shape[0] for series in series_collection)
        num_features = series_collection[0].shape[1]

        padded = np.full((len(series_collection), max_len, num_features), pad_value, dtype=np.float32)
        for i, series in enumerate(series_collection):
            padded[i, :series.shape[0], :] = series

        return padded

    def __eq__(self, other: object) -> bool:
        """Structural equality for prediction strategies.

        Compares class/type, predicted feature, input feature mapping (keys and
        processors), and biasing model (if present). Subclasses should call
        super().__eq__(other) and then compare their own configuration fields.
        """
        if self is other:
            return True
        if type(self) is not type(other):
            return False

        assert isinstance(other, PredictionStrategy)

        same_pred = (set(self.predicted_features) == set(other.predicted_features) and
                     all(proc == other.predicted_features[key]
                         for key, proc in self.predicted_features.items()))
        same_inputs = (set(self.input_features) == set(other.input_features) and
                       all(proc == other.input_features[key]
                           for key, proc in self.input_features.items()))
        same_bias = (self.hasBiasingModel == other.hasBiasingModel and
                    (not self.hasBiasingModel or self.biasing_model == other.biasing_model))

        return same_pred and same_inputs and same_bias

    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        with h5py.File(file_name, 'w') as h5_file:
            self.write_model_to_hdf5(h5_file)

    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        """ A method for saving base-class data for a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            An opened, writeable HDF5 group or file handle
        """
        if self._biasing_model is not None:
            biasing_model_group = h5_group.create_group('biasing_model')
            self._biasing_model.write_model_to_hdf5(biasing_model_group)

        pred_group = h5_group.create_group('predicted_features')
        pred_order = self.predicted_feature_names
        for name in pred_order:
            write_feature_processor(pred_group.create_group(name), self.predicted_features[name])
        if self._predicted_feature_sizes is not None:
            sizes = [self._predicted_feature_sizes[name] for name in pred_order]
            h5_group.create_dataset('predicted_feature_sizes', data=sizes)
        input_features_group = h5_group.create_group('input_features')
        for name, feature in self.input_features.items():
            write_feature_processor(input_features_group.create_group(name), feature)

    def load_model(self, h5_group: h5py.Group) -> None:
        """ A method for loading base-class data for a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            An opened HDF5 group or file handle
        """
        pred_group = h5_group['predicted_features']
        pred_order = list(pred_group.keys())
        predicted_features = {}
        for name in pred_order:
            predicted_features[name] = read_feature_processor(pred_group[name])
        self.predicted_features = predicted_features
        if 'predicted_feature_sizes' in h5_group:
            sizes = [int(v) for v in h5_group['predicted_feature_sizes'][()]]
            self._predicted_feature_sizes = dict(zip(pred_order, sizes))
        input_features = {}
        for name, feature in h5_group['input_features'].items():
            input_features[name] = read_feature_processor(feature)
        self.input_features = input_features

        if 'biasing_model' in h5_group:
            self._biasing_model.load_model(h5_group['biasing_model'])

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


    def predict(self,
                series_collection: SeriesCollection,
                num_procs:         int = 1,
    ) -> SeriesCollection:
        """Approximate predicted features for each state in each series.

        Parameters
        ----------
        series_collection : SeriesCollection
            The input state series collection which to predict outputs for
        num_procs : int, optional
            The number of processes to use for parallel processing, by default 1

        Returns
        -------
        SeriesCollection
            Series collection of predicted states. Each state contains only
            the predicted features. Padding is removed, so only real timesteps
            from the input are returned.
        """
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        processed_inputs = PredictionStrategy.preprocess_features(series_collection,
                                                                  self.input_features,
                                                                  num_procs=num_procs)
        raw_predictions = self.predict_processed_inputs(processed_inputs, num_procs=num_procs)
        series_lengths = [len(series) for series in series_collection]
        return PredictionStrategy.postprocess_features(raw_predictions,
                                                       series_lengths,
                                                       self._predicted_feature_order,
                                                       self._predicted_feature_sizes,
                                                       self.predicted_features,
                                                       num_procs=num_procs)


    def predict_processed_inputs(self, processed_inputs: np.ndarray, num_procs: int = 1) -> np.ndarray:
        """Predict target values from preprocessed, padded inputs.

        Parameters
        ----------
        processed_inputs : np.ndarray
            Preprocessed and padded input features.

        Returns
        -------
        np.ndarray
            Array of shape (num_series, max_timesteps, output_dim) containing
            predictions for each series. No NaN padding adjustments are made.
        """
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        assert self.isTrained

        y = np.asarray(self._predict_all(processed_inputs, num_procs=num_procs), dtype=np.float32)

        if self.hasBiasingModel:
            # pylint: disable=protected-access
            y += self.biasing_model._predict_all(processed_inputs, num_procs=num_procs)

        return y


    def _predict_all(self, series_collection: np.ndarray, num_procs: int = 1) -> np.ndarray:
        """ The method that predicts the target values corresponding to the given state series collection

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Parameters
        ----------
        series_collection : np.ndarray
            The preprocessed and padded state series collection in np.array format for which
            to predict the target value
        num_procs : int
            The number of parallel processors to use when predicting the data

        Returns
        -------
        np.ndarray
            The predicted target values for each state in each series
        """
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        if num_procs <= 1:
            return np.asarray([self._predict_one(series) for series in series_collection])

        series_list = list(series_collection)
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            return np.asarray(list(executor.map(self._predict_one, series_list)))


    def _get_targets(self, series_collection: SeriesCollection, num_procs: int = 1) -> np.ndarray:
        """ The method that extracts the target values for each state for training

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Target values are returned as a padded array if the series are of varying
        lengths, with 0.0 used for padding.

        Parameters
        ----------
        series_collection : SeriesCollection
            The state series collection to extract / derive the target values from

        Returns
        -------
        np.ndarray
            The target values of each state of each series to use in training
        """

        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        state = series_collection[0][0]
        self._predicted_feature_sizes = {name: int(len(state[name]))
                                         for name in self._predicted_feature_order}

        targets = PredictionStrategy.preprocess_features(series_collection,
                                                         self.predicted_features,
                                                         num_procs=num_procs)

        if self.hasBiasingModel:
            bias_series = self.biasing_model.predict(series_collection, num_procs=num_procs)
            bias = PredictionStrategy.preprocess_features(bias_series,
                                                         self.predicted_features,
                                                         num_procs=num_procs)
            targets = targets - bias
        return targets

    @classmethod
    def from_dict(cls,
                  params:            Dict,
                  input_features:    FeatureSpec,
                  predicted_features: FeatureSpec,
                  biasing_model:     Optional[PredictionStrategy] = None) -> PredictionStrategy:
        """Construct a concrete PredictionStrategy from a parameter dict.

        Parameters
        ----------
        params : Dict
            Model parameters and/or architecture description. The expected
            schema is strategy‑specific. For example, NNStrategy expects either
            a 'neural_network' key or a top‑level 'layers' key.
        input_features : FeatureSpec
            Input feature/processor pairs (Dict) or feature name(s) (str/List[str], automatically mapped to NoProcessing).
        predicted_features : FeatureSpec
            Output feature/processor pairs (Dict) or feature name(s) (str/List[str], automatically mapped to NoProcessing).
        biasing_model : Optional[PredictionStrategy], optional
            Optional prior model used to bias predictions.

        Returns
        -------
        PredictionStrategy
            A configured, untrained strategy instance.
        """

        instance = cls(input_features     = input_features,
                       predicted_features = predicted_features,
                       **params)
        if biasing_model is not None:
            instance.biasing_model = biasing_model
        return instance

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
