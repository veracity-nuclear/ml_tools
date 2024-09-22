from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Type, TypeVar
import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor
from ml_tools.model.state import State
from ml_tools.model.feature_processor import FeatureProcessor, write_feature_processor, read_feature_processor

T = TypeVar('T', bound='PredictionStrategy')

class PredictionStrategy(ABC):
    """ An abstract class for prediction strategies

    Attributes
    ----------
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding feature processing strategy
    """
    _biasing = None
    _predicted_feature = None
    _input_features = {}

    @property
    def input_features(self) -> Dict[str, FeatureProcessor]:
        return self._input_features

    @input_features.setter
    def input_features(self, input_features: Dict[str, FeatureProcessor]) -> None:
        self._input_features = {}
        for feature, processor in input_features.items():
            assert(feature is not self.predicted_feature)
            self._input_features[feature] = processor
        self._input_features = dict(sorted(self._input_features.items()))  # Ensure input features are in alphabetical order

    @property
    def predicted_feature(self) -> str:
        return self._predicted_feature

    @predicted_feature.setter
    def predicted_feature(self, predicted_feature: str) -> None:
        assert(predicted_feature not in self.input_features.keys())
        self._predicted_feature = predicted_feature

    @property
    def biasing_model(self) -> Type[T]:
        return self._biasing

    @biasing_model.setter
    def biasing_model(self, bias: Type[T]) -> None:
        assert(bias.isTrained)
        self._biasing = bias

    @property
    @abstractmethod
    def isTrained(self) -> bool:
        pass

    @property
    def hasBiasingModel(self) -> bool:
        return self._biasing is not None

    @abstractmethod
    def train(self, states: List[State], num_procs: int = 1) -> None:
        """ The method that trains the CIPS Index prediction strategy given a set of training data (i.e. List of States)

        Parameters
        ----------
        states : List[State]
            The states to use for training
        num_procs : int
            The number of parallel processors to use when training
        """
        pass

    def _biased_prediction(self, states: List[State]) -> np.ndarray:
        y = np.array([[state.feature(self.predicted_feature)] for state in states])
        if self.hasBiasingModel:
            x = np.asarray(self.biasing_model.predict(states)).reshape(-1, 1)
            y -= x
        return y

    @abstractmethod
    def _predict_one(self, state: State) -> float:
        """ The method that predicts biased CIPS index

        Parameters
        ----------
        state : State
            The input state at which to predict the biased CIPS index

        Returns
        -------
        float
            The biased CIPS index
        """
        pass

    def _predict_all(self, states: List[State]) -> List[float]:
        """ The method that approximates the biased CIPS Index corresponding to a list of states

        Parameters
        ----------
        states : List[State]
            The input states at which to predict the biased CIPS index

        Returns
        -------
        List[float]
            The biased CIPS index for each state
        """
        return [self._predict_one(state) for state in states]

    def predict(self, states: List[State]) -> List[float]:
        """ The method that approximates the predicted feature corresponding to a list of states

        Parameters
        ----------
        states : List[State]
            The input states at which to predict

        Returns
        -------
        List[float]
            The predicted feature for each state
        """
        y = np.asarray(self._predict_all(states))
        if self.hasBiasingModel:
            y += np.asarray(self.biasing_model.predict(states))
        return y.tolist()

    def preprocess_inputs(self, states: List[State], num_procs: int = 1) -> np.ndarray:
        """ Preprocesses all the input states into the processed input features of the model

        Parameters
        ----------
        states : List[State]
            The list of states to be pre-processed into a collection of model inputs
        num_procs : int
            The number of parallel processors to use when processing the data

        Returns
        -------
        np.ndarray
            The preprocessed collection of input features
        """

        inputs = []

        for feature, processor in self.input_features.items():
            feature_data = [state.feature(feature) for state in states]
            feature_data = np.array_split(feature_data, num_procs)
            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                processed_data = list(executor.map(processor.preprocess, feature_data))
            inputs.append(np.vstack(processed_data))

        return np.hstack(inputs)

    def base_save_model(self, h5_file: h5py.File):
        """ A method for saving base-class data for a trained model

        Parameters
        ----------
        h5_file : h5py.File
            An opened, writeable HDF5 file handle
        """
        #TODO: need to handle biasing
        if self._biasing is not None:
            raise AttributeError('Cannot save model with bias model attached')

        h5_file.create_dataset('predicted_feature',     data=self.predicted_feature)
        input_features_group = h5_file.create_group('input_features')
        for name, feature in self.input_features.items():
            write_feature_processor(input_features_group.create_group(name), feature)

    def base_load_model(self, h5_file: h5py.File):
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