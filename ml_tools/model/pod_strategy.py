from typing import List, Dict
import numpy as np
from math import isclose

from ml_tools.model.state import State
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import FeatureProcessor
from ml_tools.model.feature_processor import NoProcessing

class PODStrategy(PredictionStrategy):
    @property
    def isTrained(self) -> bool:
        return self._pod_mat is not None

    """ A concrete class for a POD-based prediction strategy

    Parameters
    ----------
    input_feature : str
        The state feature to use as the input feature
    predicted_feature : str
        The state feature to be predicted
    input_to_pred_map : np.ndarray
        The mapping that specifies the weights of the predicted feature "fine-mesh" signals to the
        input feature "coarse-mesh".  This should be an M-by-N matrix where M is the number of input feature
        values and N is the number of predicted feature values.  Each row of this matrix should sum to 1.0.
    """
    def __init__(self, input_feature: str, predicted_feature: str, input_to_pred_map: np.ndarray) -> None:

        assert len(input_feature) == 1

        super().__init__()
        self.input_feature      = {input_feature: NoProcessing()}
        self.predicted_feature  = predicted_feature
        self._learning_fraction = learning_fraction
        self._map               = input_to_pred_map
        self._pod_mat           = None


    def train(self, states: List[State], num_procs: int = 1) -> None:

        input_feature  = self.input_feature.keys()[0]
        output_feature = self.predicted_feature

        state = states[0]

        assert self._map.shape[0] == len(state.feature(input_feature))
        assert all(len(row) == len(state.feature(output_feature)) for row in self._map)
        assert all(isclose(row.sum(), 1.) for row in self._map)

        A = np.zeros((len(states[0].feature(output_feature)), len(states)))
        for i, state in enumerate(states):
            A[:,i] = state.feature(self._predicted_feature)

        u, S, v = np.linalg.svd(A)
        C = self._map

        nvec = self._map.shape[0]
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