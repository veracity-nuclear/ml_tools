from typing import List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from math import isclose

from ml_tools.model.state import State
from ml_tools.model.feature_processor import NoProcessing
from ml_tools.model.prediction_strategy import PredictionStrategy

class PODStrategy(PredictionStrategy):
    """ A concrete class for a categorized POD-based prediction strategy

    This prediction strategy is a POD-based strategy that may use K-means clustering to
    improve prediction accuracy by creating separate POD prediction models for the different
    clusters of inputs.  The K-means clustering may further be enhanced with PCA, which will
    project input feature vectors to a reduced dimensionality prior to clustering.  Simple
    POD may be achieved by simply not providing either k-means clustering / PCA specifications.

    Attributes
    ----------
    input_feature : str
        The feature to use as input for this model.  Note: This strategy only allows one input feature and
        this feature is expected to be a vector of floats
    fine_to_coarse_map : np.ndarray
        The mapping that specifies the weights of the predicted feature "fine-mesh" signals to the
        input feature "coarse-mesh".  This should be an M-by-N matrix where M is the number of input feature
        values and N is the number of predicted feature values.  Each row of this matrix should sum to 1.0.
    nclusters : int
        The number of K-means clusters to create separate POD models for
    max_svd_size : int
        The maximum allowed number of training samples to use for the SVD of a cluster POD model
    ndim : int
        The number of dimensions to use for the input feature PCA projection
    """

    @property
    def input_feature(self) -> str:
        return self._input_feature

    @property
    def fine_to_coarse_map(self) -> np.ndarray:
        return self._fine_to_coarse_map

    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def nclusters(self) -> int:
        return self._nclusters

    @property
    def max_svd_size(self) -> int:
        return self._max_svd_size

    @property
    def isTrained(self) -> bool:
        return all(mat is not None for mat in self._pod_mat)

    def __init__(self, input_feature: str, predicted_feature: str, fine_to_coarse_map: np.ndarray, nclusters: int = 1, max_svd_size: int = None, ndims: int = None) -> None:

        super().__init__()
        self.input_features      = {input_feature: NoProcessing()}
        self.predicted_feature   = predicted_feature
        self._input_feature      = input_feature
        self._fine_to_coarse_map = fine_to_coarse_map
        self._ndims              = ndims
        self._nclusters          = nclusters
        self._max_svd_size       = max_svd_size

        self._pod_mat = None


    def train(self, train_states: List[State], test_states: List[State] = [], num_procs: int = 1) -> None:

        self._pod_mat  = [None]*self.nclusters
        input_feature  = self.input_feature
        output_feature = self.predicted_feature
        state          = train_states[0]

        assert self.fine_to_coarse_map.shape[0] == len(state.feature(input_feature))
        assert all(len(row) == len(state.feature(output_feature)) for row in self.fine_to_coarse_map)
        assert all(isclose(row.sum(), 1.) for row in self.fine_to_coarse_map)

        # Setup of the PCA project and K-means clustering of the input feature based on the training samples
        if self.nclusters > 1:
            self._kmeans = KMeans(n_clusters=self.nclusters)
            X            = self.preprocess_inputs(train_states)

            if not(self.ndims is None):
                self._pca = PCA(n_components=self.ndims)
                X         = self._pca.fit_transform(X)

            labels = self._kmeans.fit_predict(X)
            nlabel = np.bincount(labels)

        else:
            labels = np.zeros(len(train_states), dtype=int)
            nlabel = np.asarray([len(train_states)])

        C = self.fine_to_coarse_map
        nvec = self.fine_to_coarse_map.shape[0]

        # Create separate POD models for each cluster
        for k in range(self.nclusters):
            klabels = np.where(labels == k)[0]
            if self.max_svd_size is not None and nlabel[k] > self.max_svd_size:
                klabels = np.random.choice(klabels, size=self.max_svd_size, replace=False)

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

        if self.nclusters > 1:
            X      = self.preprocess_inputs(states) if self.ndims is None else self._pca.transform(self.preprocess_inputs(states))
            labels = self._kmeans.predict(X)
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