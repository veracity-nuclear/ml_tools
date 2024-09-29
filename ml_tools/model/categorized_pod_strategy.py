from typing import List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from math import isclose
from pyvera.utils.logger import log

from ml_tools.model.state import State
from ml_tools.model.feature_processor import NoProcessing
from ml_tools.model.prediction_strategy import PredictionStrategy

#TODO: should this inherit from POD?
class CategorizedPODStrategy(PredictionStrategy):
    @property
    def isTrained(self) -> bool:
        return all(mat is not None for mat in self._pod_mat)

    """ A concrete class for a categorized POD-based prediction strategy

    Attributes
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
    def __init__(self, input_feature: str, predicted_feature: str, input_to_pred_map: np.ndarray, ndim: int, ncluster: int, max_svd_size: int = None) -> None:

        super().__init__()
        self.input_feature      = {input_feature: NoProcessing()}
        self.predicted_feature  = predicted_feature
        self._map               = input_to_pred_map
        self._ndim              = ndim
        self._ncluster          = ncluster
        self._max_svd_size      = max_svd_size

        self._pod_mat = [None]*ncluster

        if ncluster > 1:
            self._pca    = PCA(n_components=ndim)
            self._kmeans = KMeans(n_clusters=ncluster)


    def train(self, train_states: List[State], test_states: List[State] = []) -> None:

        input_feature  = self.input_feature.keys()[0]
        output_feature = self.predicted_feature

        state = train_states[0]

        assert self._map.shape[0] == len(state.feature(input_feature))
        assert all(len(row) == len(state.feature(output_feature)) for row in self._map)
        assert all(isclose(row.sum(), 1.) for row in self._map)

        if self._ncluster > 1:
            X      = self.preprocess_inputs(train_states)
            X_pca  = self._pca.fit_transform(X)
            labels = self._kmeans.fit_predict(X_pca)
            nlabel = np.bincount(labels)
        else:
            labels = np.zeros(len(train_states), dtype=int)
            nlabel = np.asarray([len(train_states)])

        C = self._map
        nvec = self._map.shape[0]

        #loop over clusters and
        for k in range(self._ncluster):
            klabels = np.where(labels == k)[0]
            if self._max_svd_size is not None and nlabel[k] > self._max_svd_size:
                klabels = np.random.choice(klabels, size=self._max_svd_size, replace=False)

            A = np.zeros((len(train_states[0].feature(output_feature)), len(klabels)))
            for i, id in enumerate(klabels):
                A[:,i] = train_states[id].feature(output_feature)

            u, S, v = np.linalg.svd(A)

            theta = np.matmul(C, u[:,:nvec])
            self._pod_mat[k] = np.matmul(u[:,:nvec],np.linalg.inv(theta))


    def _predict_one(self, state: State) -> float:
        return self._predict_all([state])[0]


    def _predict_all(self, states: List[State]) -> List[float]:

        assert(self.isTrained)
        assert(not self.hasBiasingModel)
        if self._ncluster > 1:
            X      = self.preprocess_inputs(states)
            X_pca  = self._pca.transform(X)
            labels = self._kmeans.predict(X_pca)
        else:
            labels = [0]*len(states)

        return [np.matmul(self._pod_mat[labels[i]], states[i].feature(self.input_feature)) for i in range(len(states))]


    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """
        raise NotImplementedError

    def load_model(self, file_name: str) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to load the model from
        """
        raise NotImplementedError