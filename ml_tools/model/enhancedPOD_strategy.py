from __future__ import annotations

import os
from typing import Optional, Dict
import numpy as np
import h5py

from ml_tools.model.state import SeriesCollection
from ml_tools.model.feature_processor import FeatureProcessor
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model import register_prediction_strategy
from ml_tools.model.gbm_strategy import GBMStrategy
from ml_tools.model.nn_strategy import NNStrategy

@register_prediction_strategy()  # registers under 'EnhancedPODStrategy' by default
class EnhancedPODStrategy(PredictionStrategy):
    """ A concrete class for an enhanced POD-based prediction strategy

    This prediction strategy is a POD-based strategy that may use a GBM or NN strategy
    to improve prediction accuracy by creating separate POD prediction models.

    This prediction strategy is only intended for use with static State-Points, meaning
    non-temporal series, or said another way, State Series with series lengths of one.

    Parameters
    ----------
    input_features : Dict[str, FeatureProcessor]
        The feature to use as input for this model.
    predicted_feature : str
        The string specifying the feature to be predicted
    max_svd_size : int
        The maximum allowed number of training samples to use for the SVD of a cluster POD model
    num_moments : int
        The number of total moments to use in the POD analysis
    constraints: List[Tuple[float, np.ndarray]]
        pairs of (gamma, W) constraints for computing theta.

    Attributes
    ----------
    max_svd_size : int
        The maximum allowed number of training samples to use for the SVD of a cluster POD model
    num_moments : int
        The number of total moments to use in the POD analysis
    """

    @property
    def num_moments(self) -> Optional[int]:
        return self._num_moments

    @property
    def max_svd_size(self) -> Optional[int]:
        return self._max_svd_size

    @property
    def isTrained(self) -> bool:
        return self._pod_matrix is not None

    def __init__(self,
                 input_features:        Dict[str, FeatureProcessor],
                 predicted_feature:     str,
                 theta_model:           Optional[str] = 'GBM',
                 max_svd_size:          Optional[int] = None,
                 num_moments:           Optional[int] = 1,
                 constraints:           Optional[list] = [],
                 theta_model_settings:  Optional[dict] = {}) -> None:

        super().__init__()
        self.input_features    = input_features
        self.predicted_feature = predicted_feature
        self._num_moments      = num_moments
        self._max_svd_size     = max_svd_size
        self._constraints      = constraints
        self._pod_matrix       = None
        if theta_model == "GBM":
            self._theta_model      = [GBMStrategy(input_features, f'theta-{i+1}', **theta_model_settings)
                                        for i in range(num_moments)]
        elif theta_model == "NN":
            # NN can predict all moments at once
            self._theta_model      = [NNStrategy(input_features, 'theta', **theta_model_settings)]
        else:
            raise ValueError(f"Unsupported theta model type: {theta_model}")

    def _solve_theta_star(self, predicted):
        """
        theta = U^T * predicted_feature
        theta_star = argmin_theta ||U theta - predicted_feature||^2_2
                                  + gamma_1 ||W_1*(U theta - predicted_feature) ||^2_2
                                  + gamma_2 ||W_2*(U theta - predicted_feature) ||^2_2
        """
        if len(self._constraints) == 0:
            return self._pod_matrix.T @ predicted

        A = [self._pod_matrix]
        b = [predicted]

        for gamma, W in self._constraints:
            A.append(gamma * W @ self._pod_matrix)
            b.append(gamma * W @ predicted)

        # Least-squares solve
        theta_star, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
        return theta_star

    def _add_theta_to_collection(self, collection: SeriesCollection) -> None:
        for series in collection:
            for state in series:
                predicted   = state[self.predicted_feature]
                theta_star = self._solve_theta_star(predicted)
                state.features['theta'] = theta_star

                for i, theta_i in enumerate(theta_star):
                    state.features[f'theta-{i+1}'] = [theta_i]

    def _add_theta(self,
                   train_data: SeriesCollection,
                   test_data: Optional[SeriesCollection] = None) -> None:
        self._add_theta_to_collection(train_data)
        if test_data is not None:
            self._add_theta_to_collection(test_data)

    def train(self,
              train_data: SeriesCollection,
              test_data: Optional[SeriesCollection] = None, num_procs: int = 1) -> None:

        if self._max_svd_size is None:
            self._max_svd_size = len(train_data)

        # setup matrix and perform SVD
        A = np.vstack([np.array(series) for series in self._get_targets(train_data)])
        if len(train_data) > self.max_svd_size:
            A = A[np.random.choice(A.shape[0], self.max_svd_size, replace=False), :]
        u, _, _ = np.linalg.svd(A.T, full_matrices=False)

        # store intermediate theta_d and pod matrix
        self._pod_matrix = u[:, :self.num_moments]

        # add theta_xi to collections for training
        self._add_theta(train_data, test_data)

        # train models
        for model in self._theta_model:
            model.train(train_data, test_data, num_procs)

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        assert self.isTrained

        n_timesteps = state_series.shape[0]

        pred = []
        for i in range(n_timesteps):
            X = state_series[i,:].reshape(1, -1)
            theta = np.asarray([self._theta_model[r].predict_processed_inputs(X[np.newaxis, :])[0]
                                    for r in range(self.num_moments)])
            pred.append(self._pod_matrix @ theta.flatten())

        y = np.asarray(pred)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        return y.reshape(n_timesteps, -1)

    def save_model(self, file_name: str, clean_files: bool = True) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"

        with h5py.File(file_name, 'a') as h5_file:
            self.base_save_model(h5_file)
            h5_file.create_dataset('num_moments', data=self.num_moments)
            h5_file.create_dataset('pod_mat', data=self._pod_matrix)
            h5_file.create_dataset('num_constraints', data=len(self._constraints))

            for i, (gamma, W) in enumerate(self._constraints):
                h5_file.create_dataset(f'constraint_{i+1}_gamma', data=gamma)
                h5_file.create_dataset(f'constraint_{i+1}_W', data=W)

            for i in range(self.num_moments):
                self._theta_model[i].write_model_to_hdf5(h5_file, group=f'theta-{i+1}', clean_files=clean_files)

    def load_model(self, h5_file: h5py.File) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        h5_file : h5py.File
            An open HDF5 file object from which to load the model
        """
        file_name = h5_file.filename
        self.base_load_model(h5_file)

        self.num_moments  = int(h5_file['num_moments'][()])
        self._pod_matrix     = h5_file['pod_mat'][()]
        #TODO fix for NN
        self._theta_model    = [GBMStrategy(self.input_features, f'theta-{i+1}') for i in range(self.num_moments)]

        num_constraints = int(h5_file['num_constraints'][()])
        self._constraints = [(h5_file[f'constraint_{i+1}_gamma'][()],
                                  h5_file[f'constraint_{i+1}_W'][()]) for i in range(num_constraints)]

        for i in range(self.num_moments):
            self._theta_model[i].load_model(h5_file, group=f'theta-{i+1}')

    @classmethod
    def read_from_file(cls, file_name: str) -> EnhancedPODStrategy:
        """ A basic factory method for building a MIP-DT Strategy from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the file from which to read the model

        Returns
        -------
        EnhancedPODStrategy:
            The model from the hdf5 file
        """
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        assert os.path.exists(file_name), f"file name = {file_name}"

        with h5py.File(file_name, "r") as h5_file:
            r = int(h5_file['num_moments'][()])
        new_pod = cls({}, None, r)
        new_pod.load_model(h5py.File(file_name, "r"))

        return new_pod
