from __future__ import annotations

import os
from typing import Optional, Dict, Sequence
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py

from ml_tools.model.state import SeriesCollection
from ml_tools.model.feature_processor import FeatureProcessor
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model import register_prediction_strategy
from ml_tools.model.gbm_strategy import GBMStrategy
from ml_tools.model.nn_strategy import NNStrategy

class LOGProcessing(FeatureProcessor):
    def __init__(self):
        pass

    def preprocess(self, orig_data: Sequence) -> np.ndarray:
        dat = np.array(orig_data, copy=True)
        return np.sign(dat) * np.log10(np.abs(dat) + 1)

    def postprocess(self, processed_data: np.ndarray) -> np.ndarray:
        processed = np.array(processed_data, copy=True)
        return np.sign(processed) * (10**np.abs(processed) - 1)

    def __eq__(self, other: FeatureProcessor) -> bool:
        return isinstance(other, LOGProcessing)

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
    predicted_features : str
        The string specifying the feature to be predicted
    theta_model_type : Optional[str]
        The type of model to use for predicting theta.  Supported types are 'GBM' and 'NN'.  Default is 'GBM'.
    theta_scaling : Optional[str]
        The type of scaling to use for theta normalization. Supported types are 'singular_values', 'sqrt_singular_values',
        'max_singular_value', 'max_sqrt_singular_value', and 'ones'. Default is 'singular_values'.
    max_svd_size : int
        The maximum allowed number of training samples to use for the SVD of a cluster POD model
    num_moments : int
        The number of total moments to use in the POD analysis
    constraints: List[Tuple[float, np.ndarray]]
        pairs of (gamma, W) constraints for computing theta.
    theta_model_settings: Optional[Dict[str, any]]
        Additional settings to pass to the theta model constructor

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
    def theta_model_type(self) -> Optional[str]:
        return self._theta_model_type

    @property
    def isTrained(self) -> bool:
        return self._pod_matrix is not None

    def __init__(self,
                 input_features:        Dict[str, FeatureProcessor],
                 predicted_features:    str,
                 theta_model_type:      Optional[str] = 'GBM',
                 max_svd_size:          Optional[int] = None,
                 num_moments:           Optional[int] = 1,
                 constraints:           Optional[list] = None,
                 theta_scaling:         Optional[str] = 'singular_values',
                 theta_model_settings:  Optional[Dict[str, any]] = None) -> None:

        super().__init__()
        self.input_features    = input_features
        self.predicted_features = predicted_features
        self._num_moments      = num_moments
        self._max_svd_size     = max_svd_size
        self._constraints      = constraints if constraints is not None else []
        self._pod_matrix       = None
        self._theta_model_type = theta_model_type
        if theta_scaling in ['singular_values', 'sqrt_singular_values',
                             'max_singular_value', 'max_sqrt_singular_value',
                             'ones']: 
            self._theta_scaling    = theta_scaling
        else:
            raise ValueError(f"Unsupported theta scaling type: {theta_scaling}")
        theta_model_settings   = theta_model_settings if theta_model_settings is not None else {}
        if theta_model_type == "GBM":
            self._theta_model      = [GBMStrategy(input_features, f'theta-{i+1}', **theta_model_settings)
                                        for i in range(num_moments)]
        elif theta_model_type == "NN":
            # NN can predict all moments at once
            self._theta_model      = [NNStrategy(input_features, 'theta', **theta_model_settings)]
        else:
            raise ValueError(f"Unsupported theta model type: {theta_model_type}")

    def _solve_theta_star(self, predicted):
        """
        theta = U^T * predicted_features
        theta_star = argmin_theta ||U theta - predicted_features||^2_2
                                  + gamma_1 ||W_1*(U theta - predicted_features) ||^2_2
                                  + gamma_2 ||W_2*(U theta - predicted_features) ||^2_2
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

    @staticmethod
    def _compute_theta_for_state(args):
        """Helper function for parallel theta computation."""
        predicted, pod_matrix, constraints, scaling_vector = args
        
        # Solve theta
        if len(constraints) == 0:
            theta_star = pod_matrix.T @ predicted
        else:
            A = [pod_matrix]
            b = [predicted]
            for gamma, W in constraints:
                A.append(gamma * W @ pod_matrix)
                b.append(gamma * W @ predicted)
            theta_star, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
        
        # Normalize by scaling vector
        theta_star = theta_star / scaling_vector
        return theta_star

    def _add_theta_to_collection(self, collection: SeriesCollection, scaling_vector: np.ndarray, num_procs: int = 1) -> None:
        assert len(self.predicted_feature_names) == 1, "EnhancedPODStrategy only supports one predicted feature"
        
        # Collect all states that need theta computation
        all_states = []
        for series in collection:
            for state in series:
                all_states.append(state)
        
        # Prepare arguments for parallel processing
        args_list = [
            (state[self.predicted_feature_names[0]], self._pod_matrix, self._constraints, scaling_vector)
            for state in all_states
        ]
        
        # Compute theta in parallel
        if num_procs > 1:
            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                theta_results = list(executor.map(self._compute_theta_for_state, args_list))
        else:
            theta_results = [self._compute_theta_for_state(args) for args in args_list]
        
        # Assign results back to states
        for state, theta_star in zip(all_states, theta_results):
            state.features['theta'] = theta_star
            for i, theta_i in enumerate(theta_star):
                state.features[f'theta-{i+1}'] = [theta_i]

    def _add_theta(self,
                   train_data: SeriesCollection,
                   test_data: Optional[SeriesCollection] = None,
                   scaling_vector: np.ndarray = None,
                   num_procs: int = 1) -> None:
        if scaling_vector is None:
            scaling_vector = np.ones(self.num_moments)
        self._add_theta_to_collection(train_data, scaling_vector, num_procs)
        if test_data is not None:
            self._add_theta_to_collection(test_data, scaling_vector, num_procs)

    def _compute_pod(self, data: SeriesCollection) -> None:
        if self._max_svd_size is None:
            self._max_svd_size = len(data)

        # setup matrix and perform SVD
        A = np.vstack([np.array(series) for series in self._get_targets(data)])
        if len(data) > self.max_svd_size:
            A = A[np.random.choice(A.shape[0], self.max_svd_size, replace=False), :]
        u, L, _ = np.linalg.svd(A.T, full_matrices=False)

        # store intermediate pod matrix
        self._pod_matrix = u[:, :self.num_moments]
        self._singular_values = L[:self.num_moments]

    def _scaling_vector(self) -> np.ndarray:
        if self._theta_scaling == 'singular_values':
            scaling_vector = self._singular_values + 1e-12
        elif self._theta_scaling == 'sqrt_singular_values':
            scaling_vector = np.sqrt(self._singular_values + 1e-12)
        elif self._theta_scaling == 'max_singular_value':
            scaling_vector = self._singular_values[0] * np.ones(self.num_moments)
        elif self._theta_scaling == 'max_sqrt_singular_value':
            scaling_vector = np.sqrt(self._singular_values[0]) * np.ones(self.num_moments)
        elif self._theta_scaling == 'ones':
            scaling_vector = np.ones(self.num_moments)
        else:
            raise ValueError(f"Unsupported theta scaling type: {self._theta_scaling}")
        return scaling_vector
    
    def train(self,
              train_data: SeriesCollection,
              test_data: Optional[SeriesCollection] = None, num_procs: int = 1) -> None:

        # compute pod matrix
        self._compute_pod(train_data)

        # add theta_xi to collections for training
        self._add_theta(train_data, test_data, scaling_vector=self._scaling_vector(), num_procs=num_procs)

        # train models
        for model in self._theta_model:
            if isinstance(model, NNStrategy):
                model.train(train_data + test_data, num_procs=num_procs)
            else:
                model.train(train_data, test_data, num_procs=num_procs)

    def _predict_theta(self, collection: SeriesCollection) -> np.ndarray:
        if self._theta_model_type == "NN":
            theta_pred = self._theta_model[0].predict(collection)
        else:
            theta_pred_tmp = [self._theta_model[r].predict(collection)
                                    for r in range(self.num_moments)]
            theta_pred = theta_pred_tmp[0]
            for i, series in enumerate(theta_pred):
                for j, state in enumerate(series):
                    for r in range(1, self.num_moments):
                        state.features[f'theta-{r+1}'] = theta_pred_tmp[r][i][j][f'theta-{r+1}']
                    state.features['theta'] = np.array([state[f'theta-{r+1}'] for r in range(self.num_moments)])
        return theta_pred

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        assert self.isTrained

        scaling_vector = self._scaling_vector()

        n_timesteps = state_series.shape[0]
        pred = []
        for i in range(n_timesteps):
            X = state_series[i,:].reshape(1, -1)
            theta = np.asarray([self._theta_model[r].predict_processed_inputs(X[np.newaxis, :])[0]
                                    for r in range(self.num_moments)])
            # Denormalize theta by multiplying by singular values
            theta = theta.flatten() * scaling_vector
            pred.append(self._pod_matrix @ theta)

        y = np.asarray(pred)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        return y.reshape(n_timesteps, -1)

    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            The opened HDF5 group or file to which the model should be written
        """
        file_name = h5_group.file.filename

        super().write_model_to_hdf5(h5_group)
        h5_group.create_dataset('num_moments', data=self.num_moments)
        h5_group.create_dataset('pod_mat', data=self._pod_matrix)
        h5_group.create_dataset('singular_values', data=self._singular_values)
        h5_group.create_dataset('num_constraints', data=len(self._constraints))
        h5_group.create_dataset('max_svd_size', data=self._max_svd_size)
        h5_group.create_dataset('theta_model_type', data=self._theta_model_type)
        h5_group.create_dataset('theta_scaling', data=self._theta_scaling)

        for i, (gamma, W) in enumerate(self._constraints):
            h5_group.create_dataset(f'constraint_{i+1}_gamma', data=gamma)
            h5_group.create_dataset(f'constraint_{i+1}_W', data=W)

        if self._theta_model_type == "NN":
            theta_group = h5_group.require_group('theta')
            self._theta_model[0].write_model_to_hdf5(theta_group)
        else:
            for i in range(self.num_moments):
                theta_group = h5_group.require_group(f'theta-{i+1}')
                self._theta_model[i].write_model_to_hdf5(theta_group)

    def load_model(self, h5_group: h5py.Group) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            An open HDF5 group or file object from which to load the model
        """
        file_name = h5_group.file.filename
        super().load_model(h5_group)

        self._num_moments    = int(h5_group['num_moments'][()])
        self._pod_matrix     = h5_group['pod_mat'][()]
        self._singular_values = h5_group['singular_values'][()]
        self._theta_model_type = h5_group['theta_model_type'][()].decode('utf-8')
        # Load theta_scaling with backward compatibility
        if 'theta_scaling' in h5_group:
            self._theta_scaling = h5_group['theta_scaling'][()].decode('utf-8')
        else:
            self._theta_scaling = 'ones'  # default for old models
        if self._theta_model_type == "GBM":
            self._theta_model    = [GBMStrategy(self.input_features, f'theta-{i+1}') for i in range(self.num_moments)]
        elif self._theta_model_type == "NN":
            self._theta_model    = [NNStrategy(self.input_features, 'theta')]
        else:
            raise ValueError(f"Unsupported theta model type: {self._theta_model_type}")

        num_constraints = int(h5_group['num_constraints'][()])
        self._constraints = [(h5_group[f'constraint_{i+1}_gamma'][()],
                                  h5_group[f'constraint_{i+1}_W'][()]) for i in range(num_constraints)]

        if self._theta_model_type == "NN":
            self._theta_model[0].load_model(h5_group['theta'])
        else:
            for i in range(self.num_moments):
                self._theta_model[i].load_model(h5_group[f'theta-{i+1}'])

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
            theta_model_type = str(h5_file['theta_model_type'][()]).decode('utf-8')
        new_pod = cls({}, None, r, theta_model_type)
        new_pod.load_model(h5py.File(file_name, "r"))

        return new_pod
