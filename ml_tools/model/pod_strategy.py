from __future__ import annotations
from typing import Optional, Type, Dict
from math import isclose
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from ml_tools.model.state import SeriesCollection
from ml_tools.model.feature_processor import NoProcessing
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureProcessor
from ml_tools.model import register_prediction_strategy

@register_prediction_strategy()  # registers under 'PODStrategy'
class PODStrategy(PredictionStrategy):
    """ A concrete class for a categorized POD-based prediction strategy

    This prediction strategy is a POD-based strategy that may use K-means clustering to
    improve prediction accuracy by creating separate POD prediction models for the different
    clusters of inputs.  The K-means clustering may further be enhanced with PCA, which will
    project input feature vectors to a reduced dimensionality prior to clustering.  Simple
    POD may be achieved by simply not providing either k-means clustering / PCA specifications.

    This prediction strategy is only intended for use with static State-Points, meaning
    non-temporal series, or said another way, State Series with series lengths of one.

    Parameters
    ----------
    input_feature : str
        The feature to use as input for this model.  Note: This strategy only allows one input feature and
        this feature is expected to be a vector of floats
    predicted_feature : str
        The string specifying the feature to be predicted
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
    def ndims(self) -> Optional[int]:
        return self._ndims

    @property
    def nclusters(self) -> int:
        return self._nclusters

    @property
    def max_svd_size(self) -> Optional[int]:
        return self._max_svd_size

    @property
    def isTrained(self) -> bool:
        return all(mat is not None for mat in self._pod_mat)

    def __init__(self,
                 input_feature:      str,
                 predicted_feature:  str,
                 fine_to_coarse_map: np.ndarray,
                 nclusters:          int = 1,
                 max_svd_size:       Optional[int] = None,
                 ndims:              Optional[int] = None) -> None:

        super().__init__()
        self.input_features      = {input_feature: NoProcessing()}
        self.predicted_feature   = predicted_feature
        self._input_feature      = input_feature
        self._fine_to_coarse_map = fine_to_coarse_map
        self._ndims              = ndims
        self._nclusters          = nclusters
        self._max_svd_size       = max_svd_size

        self._pod_mat = None
        self._pca     = None
        self._kmeans  = None


    def train(self, train_data: SeriesCollection, test_data: Optional[SeriesCollection] = None, num_procs: int = 1) -> None:

        assert test_data is None, "The POD Prediction Strategy does not use test data"

        self._pod_mat  = [None]*self.nclusters
        input_feature  = self.input_feature
        output_feature = self.predicted_feature
        state          = train_data[0][0]

        assert self.fine_to_coarse_map.shape[0] == len(state[input_feature]), \
            f"Fine-to-coarse mapping entry length is {self.fine_to_coarse_map.shape[0]}, \
                length of {input_feature} is {len(state.feature(input_feature))}"
        assert all(len(row) == len(state[output_feature]) for row in self.fine_to_coarse_map)
        assert all(isclose(row.sum(), 1.) for row in self.fine_to_coarse_map)

        # Setup of the PCA project and K-means clustering of the input feature based on the training samples
        targets = np.vstack([np.array(series) for series in self._get_targets(train_data, num_procs=num_procs)])
        if self.nclusters > 1:
            self._kmeans = KMeans(n_clusters=self.nclusters)
            X            = self.preprocess_inputs(train_data)
            X            = X.reshape(-1, X.shape[-1])

            if not self.ndims is None:
                self._pca = PCA(n_components=self.ndims)
                X         = self._pca.fit_transform(X)

            labels = self._kmeans.fit_predict(X)
            nlabel = np.bincount(labels)

        else:
            labels = np.zeros(len(targets), dtype=int)
            nlabel = np.asarray([len(targets)])

        C = self.fine_to_coarse_map
        nvec = self.fine_to_coarse_map.shape[0]

        # Create separate POD models for each cluster
        for k in range(self.nclusters):
            klabels = np.where(labels == k)[0]
            if self.max_svd_size is not None and nlabel[k] > self.max_svd_size:
                klabels = np.random.choice(klabels, size=self.max_svd_size, replace=False)

            A = np.zeros((len(state[output_feature]), len(klabels)))
            for i, label in enumerate(klabels):
                A[:,i] = targets[label]

            u, S, v = np.linalg.svd(A)

            theta = np.matmul(C, u[:,:nvec])
            self._pod_mat[k] = np.matmul(u[:,:nvec],np.linalg.inv(theta))


    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        return self._predict_all([state_series])[0]


    def _predict_all(self, series_collection: np.ndarray) -> np.ndarray:

        assert self.isTrained
        assert not self.hasBiasingModel
        assert series_collection.shape[1] == 1, \
            "All State Series must be static statepoints (i.e. len(series) == 1)"

        X = series_collection[:, 0, :]

        if self.nclusters > 1:
            X_reduced = X if self.ndims is None else self._pca.transform(X)
            labels    = self._kmeans.predict(X_reduced)
        else:
            labels = np.zeros(X.shape[0], dtype=int)

        results = np.array([np.matmul(self._pod_mat[label], vec)
                            for label, vec in zip(labels, X)])

        return results[:, np.newaxis]

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        assert isinstance(other, PODStrategy)
        return (self.input_feature == other.input_feature and
                self.nclusters     == other.nclusters     and
                self.max_svd_size  == other.max_svd_size  and
                self.ndims         == other.ndims         and
                np.allclose(self.fine_to_coarse_map, other.fine_to_coarse_map))


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

    @classmethod
    def read_from_file(cls, file_name: str) -> Type[PODStrategy]:
        """ A method for loading a trained model from a file

        Parameters
        ----------
        file_name : str
            The name of the file to load the model from
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls,
                  params:            Dict,
                  input_features:    Dict[str, FeatureProcessor],
                  predicted_feature: str,
                  biasing_model:     Optional[PredictionStrategy] = None) -> PODStrategy:

        assert input_features is not None and len(input_features) == 1, \
            "PODStrategy requires exactly one input feature"
        input_feature = list(input_features.keys())[0]
        kwargs = {
            "fine_to_coarse_map": (np.asarray(params.get("fine_to_coarse_map"), dtype=float)
                                    if params.get("fine_to_coarse_map") is not None else None),
            "nclusters": params.get("nclusters", 1),
            "max_svd_size": params.get("max_svd_size", None),
            "ndims": params.get("ndims", None),
        }
        instance = cls(input_feature     = input_feature,
                       predicted_feature = predicted_feature,
                       **kwargs)
        if biasing_model is not None:
            instance.biasing_model = biasing_model
        return instance

    def to_dict(self) -> Dict:
        return {"input_feature":      self.input_feature,
                "fine_to_coarse_map": (self.fine_to_coarse_map.tolist()),
                "nclusters":          self.nclusters,
                "max_svd_size":       self.max_svd_size,
                "ndims":              self.ndims}
