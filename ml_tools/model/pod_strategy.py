from typing import List, Dict
import numpy as np
from math import isclose

from ml_tools.model.state import State
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import FeatureProcessor
from ml_tools.model.feature_processor import NoProcessing

class PODStrategy(PredictionStrategy):
    """ A concrete class for a POD-based prediction strategy

    Attributes
    ----------
    input_feature : str
        The feature to use as input for this model.  Note: This strategy only allows one input feature and
        this feature is expected to be a vector of floats
    fine_to_coarse_map : np.ndarray
        The mapping that specifies the weights of the predicted feature "fine-mesh" signals to the
        input feature "coarse-mesh".  This should be an M-by-N matrix where M is the number of input feature
        values and N is the number of predicted feature values.  Each row of this matrix should sum to 1.0.
    """

    @property
    def input_feature(self) -> str:
        return self._input_feature

    @property
    def fine_to_coarse_map(self) -> np.ndarray:
        return self._fine_to_coarse_map

    @property
    def isTrained(self) -> bool:
        return self._pod_mat is not None


    def __init__(self, input_feature: str, predicted_feature: str, fine_to_coarse_map: np.ndarray) -> None:

        super().__init__()
        self.input_features      = {input_feature: NoProcessing()}
        self.predicted_feature   = predicted_feature
        self._input_feature      = input_feature
        self._fine_to_coarse_map = fine_to_coarse_map
        self._pod_mat            = None


    def train(self, train_states: List[State], test_states: List[State] = [], num_procs: int = 1) -> None:

        input_feature  = self.input_feature
        output_feature = self.predicted_feature

        state = train_states[0]

        assert self.fine_to_coarse_map.shape[0] == len(state.feature(input_feature))
        assert all(len(row) == len(state.feature(output_feature)) for row in self.fine_to_coarse_map)
        assert all(isclose(row.sum(), 1.) for row in self.fine_to_coarse_map)

        A = np.zeros((len(train_states[0].feature(output_feature)), len(train_states)))
        for i, state in enumerate(train_states):
            A[:,i] = state.feature(self._predicted_feature)

        u, S, v = np.linalg.svd(A)
        C = self.fine_to_coarse_map

        nvec = self._fine_to_coarse_map.shape[0]
        theta = np.matmul(C, u[:,:nvec])
        self._pod_mat = np.matmul(u[:,:nvec],np.linalg.inv(theta))


    def _predict_one(self, state: State) -> float:

        assert(self.isTrained)
        assert(not self.hasBiasingModel)

        return np.matmul(self._pod_mat, self.preprocess_inputs([state])[0])


    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """
        np.save(open(file_name, 'wb'), self._pod_mat)


    def load_model(self, file_name: str) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to load the model from
        """
        self._pod_mat = np.load(open(file_name, 'rb'))