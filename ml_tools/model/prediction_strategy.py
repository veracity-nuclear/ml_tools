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


    def preprocess_inputs(self, series_collection: SeriesCollection, num_procs: int = 1) -> np.ndarray:
        """ Preprocesses all the input state series into the series of processed input features of the model

        Parameters
        ----------
        series_collection : SeriesCollection
            The list of state series to be pre-processed into a collection of model input series
        num_procs : int
            The number of parallel processors to use when processing the data

        Returns
        -------
        np.ndarray
            The preprocessed collection of state series input features. All series are padded
            with 0.0 to match the length of the longest series.
        """
        if num_procs <= 1:
            processed_inputs = [
                self._process_single_series(series, self.input_features)
                for series in series_collection
            ]
        else:
            input_features = [self.input_features] * len(series_collection)  # input_features for each worker
            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                processed_inputs = list(executor.map(self._process_single_series, series_collection, input_features))

        return self._pad_series(processed_inputs)


    @staticmethod
    def _process_single_series(series: StateSeries, input_features: Dict[str, FeatureProcessor]) -> np.ndarray:
        """ Helper method to support parallelizing preprocess_inputs

        This is required to be a separate method so it can be pickled by ProcessPoolExecutor
        """

        processed_inputs = []
        for feature, processor in input_features.items():
            feature_data = np.array([state.features[feature] for state in series])
            processed_data = np.asarray(processor.preprocess(feature_data))
            if processed_data.ndim == 1:
                processed_data = processed_data[:, np.newaxis]
            elif processed_data.ndim > 2:
                processed_data = np.vstack(processed_data)
            processed_inputs.append(processed_data)
        return np.hstack(processed_inputs)

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

        same_pred = self.predicted_feature == other.predicted_feature
        same_inputs = (set(self.input_features) == set(other.input_features) and
                       all(proc == other.input_features[key]
                           for key, proc in self.input_features.items()))
        same_bias = (self.hasBiasingModel == other.hasBiasingModel and
                    (not self.hasBiasingModel or self.biasing_model == other.biasing_model))

        return same_pred and same_inputs and same_bias

    def base_save_model(self, h5_group: h5py.Group) -> None:
        """ A method for saving base-class data for a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            An opened, writeable HDF5 group or file handle
        """
        if self._biasing_model is not None:
            biasing_model_group = h5_group.create_group('biasing_model')
            self._biasing_model.write_model_to_hdf5(biasing_model_group)

        h5_group.create_dataset('predicted_feature', data=self.predicted_feature)
        input_features_group = h5_group.create_group('input_features')
        for name, feature in self.input_features.items():
            write_feature_processor(input_features_group.create_group(name), feature)

    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        with h5py.File(file_name, 'w') as h5_file:
            self.base_save_model(h5_file)
            self.write_model_to_hdf5(h5_file)

    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        """ A method for writing the model to an already opened HDF5 file

        Parameters
        ----------
        h5_group : h5py.Group
            The opened HDF5 file or group to which the model should be written
        """
        pass

    def load_model(self, h5_group: h5py.Group) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            The opened HDF5 file or group from which the model should be loaded
        """
        self.base_load_model(h5_group)

    def base_load_model(self, h5_group: h5py.Group) -> None:
        """ A method for loading base-class data for a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            An opened HDF5 group or file handle
        """
        self.predicted_feature = h5_group['predicted_feature'][()].decode('utf-8')
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
    ) -> List[List[np.ndarray]]:
        """Approximate predicted features for each state in each series.

        Parameters
        ----------
        series_collection : SeriesCollection
            The input state series collection which to predict outputs for
        num_procs : int, optional
            The number of processes to use for parallel processing, by default 1

        Returns
        -------
        List[List[np.ndarray]]
            Ragged list of predictions. The first list is over the series, the
            second list is over the states within that series, and each element
            is the predicted output for that state. Padding is removed, so only
            real timesteps from the input are returned.
        """
        processed_inputs = self.preprocess_inputs(series_collection, num_procs=num_procs)
        padded_predictions = self.predict_processed_inputs(processed_inputs)
        series_lengths = [len(series) for series in series_collection]
        return self.post_process_outputs(padded_predictions, series_lengths)


    def predict_processed_inputs(self, processed_inputs: np.ndarray) -> np.ndarray:
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
        assert self.isTrained

        y = np.asarray(self._predict_all(processed_inputs), dtype=np.float32)

        if self.hasBiasingModel:
            # pylint: disable=protected-access
            y += self.biasing_model._predict_all(processed_inputs)

        return y


    def post_process_outputs(
        self,
        padded_predictions: np.ndarray,
        series_lengths: Optional[List[int]] = None,
    ) -> List[List[np.ndarray]]:
        """Convert padded predictions into ragged per-series outputs.

        Parameters
        ----------
        padded_predictions : np.ndarray
            Padded prediction array of shape (num_series, max_timesteps, output_dim).
        series_lengths : Optional[List[int]]
            True lengths for each series. If not provided, assumes all series
            are full length.

        Returns
        -------
        List[List[np.ndarray]]
            Ragged list of predictions, trimmed to each series length.
        """
        if series_lengths is None:
            series_lengths = [padded_predictions.shape[1]] * padded_predictions.shape[0]

        ragged_predictions: List[List[np.ndarray]] = []
        for padded, series_len in zip(padded_predictions, series_lengths):
            ragged_predictions.append(list(padded[:series_len]))

        return ragged_predictions


    def _predict_all(self, series_collection: np.ndarray) -> np.ndarray:
        """ The method that predicts the target values corresponding to the given state series collection

        Target value in this case refers either directly to the predicted feature if
        no bias model is present, or the error of the bias model, in the case that a
        bias model is present.

        Parameters
        ----------
        series_collection : np.ndarray
            The preprocessed and padded state series collection in np.array format for which
            to predict the target value

        Returns
        -------
        np.ndarray
            The predicted target values for each state in each series
        """
        return np.asarray([self._predict_one(series) for series in series_collection])


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

        seq_targets: List[np.ndarray] = []
        for series in series_collection:
            vals: List[np.ndarray] = []
            for state in series:
                v = np.asarray(state[self.predicted_feature])
                v = np.atleast_1d(v).astype(np.float32)
                vals.append(v)
            if not vals:
                continue
            seq_targets.append(np.vstack(vals))
        targets = self._pad_series(seq_targets, pad_value=0.0)

        if self.hasBiasingModel:
            bias_ragged = self.biasing_model.predict(series_collection, num_procs=num_procs)
            bias_series: List[np.ndarray] = []
            for series in bias_ragged:
                bias_series.append(np.vstack(series))
            bias = self._pad_series(bias_series, pad_value=0.0)
            targets = targets - bias
        return targets

    @classmethod
    def from_dict(cls,
                  params:            Dict,
                  input_features:    Dict[str, FeatureProcessor],
                  predicted_feature: str,
                  biasing_model:     Optional[PredictionStrategy] = None) -> PredictionStrategy:
        """Construct a concrete PredictionStrategy from a parameter dict.

        Parameters
        ----------
        params : Dict
            Model parameters and/or architecture description. The expected
            schema is strategy‑specific. For example, NNStrategy expects either
            a 'neural_network' key or a top‑level 'layers' key.
        input_features : Dict[str, FeatureProcessor]
            Feature processors keyed by feature name.
        predicted_feature : str
            Target feature name to predict.
        biasing_model : Optional[PredictionStrategy], optional
            Optional prior model used to bias predictions.

        Returns
        -------
        PredictionStrategy
            A configured, untrained strategy instance.
        """

        instance = cls(input_features    = input_features,
                       predicted_feature = predicted_feature,
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
