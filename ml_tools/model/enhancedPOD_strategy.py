from __future__ import annotations

import os
from typing import Optional, Dict, Type, List, Tuple
import numpy as np
import h5py

from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import FeatureSpec
from ml_tools.model.pod_based_strategy import PODBasedStrategy
from ml_tools.model.pod_theta import (
    THETA_SCALINGS,
    ThetaModel,
    compute_pod_basis,
    project_theta,
    theta_scaling_vector,
)
from ml_tools.model import register_prediction_strategy
from ml_tools.model.feature_processor import NoProcessing

@register_prediction_strategy()  # registers under 'EnhancedPODStrategy' by default
class EnhancedPODStrategy(PODBasedStrategy):
    """ A concrete class for an enhanced POD-based prediction strategy

    This prediction strategy is a POD-based strategy that may use a GBM or NN strategy
    to improve prediction accuracy by creating separate POD prediction models.

    This prediction strategy is only intended for use with static State-Points, meaning
    non-temporal series, or said another way, State Series with series lengths of one.

    Parameters
    ----------
    input_features : FeatureSpec
        The features to use as input for this model
    predicted_features : FeatureSpec
        The string or FeatureSpec specifying the feature to be predicted
    theta_model_type : Optional[str]
        The type of model to use for predicting theta.  Supported types are 'GBM', 'NN', and 'sklearn'.  Default is 'GBM'.
    theta_scaling : Optional[str]
        The type of scaling to use for theta normalization. Supported types are 'singular_values', 'sqrt_singular_values',
        'max_singular_value', 'max_sqrt_singular_value', and 'ones'. Default is 'singular_values'.
    max_svd_size : int
        The maximum allowed number of training samples to use for the SVD of a cluster POD model
    num_moments : int
        The number of total moments to use in the POD analysis
    constraints: Optional[List[Tuple[float, np.ndarray]]]
        pairs of (gamma, W) constraints for computing theta.
    theta_model_settings: Optional[Dict[str, any]]
        Additional settings to pass to the theta model constructor

    Attributes
    ----------
    max_svd_size : int
        The maximum allowed number of training samples to use for the SVD of a cluster POD model
    num_moments : int
        The number of total moments to use in the POD analysis
    moments : np.ndarray
        The POD moments of shape (output_dim, num_moments)
    """

    @property
    def num_moments(self) -> Optional[int]:
        return self._num_moments

    @property
    def max_svd_size(self) -> Optional[int]:
        return self._max_svd_size

    @property
    def theta_model_type(self) -> Optional[str]:
        return self._theta_model.model_type

    @property
    def isTrained(self) -> bool:
        return self._pod_matrix is not None

    @property
    def moments(self) -> np.ndarray:
        assert self.isTrained
        return self._pod_matrix

    def __init__(self,
                 input_features:        FeatureSpec,
                 predicted_features:    FeatureSpec,
                 theta_model_type:      Optional[str] = 'GBM',
                 max_svd_size:          Optional[int] = None,
                 num_moments:           Optional[int] = 1,
                 constraints:           Optional[List[Tuple[float, np.ndarray]]] = None,
                 theta_scaling:         Optional[str] = 'sqrt_singular_values',
                 theta_model_settings:  Optional[Dict[str, any]] = None) -> None:

        super().__init__()
        self.input_features    = input_features
        self.predicted_features = predicted_features
        self._num_moments      = num_moments
        self._max_svd_size     = max_svd_size
        self._constraints      = constraints if constraints is not None else []
        self._pod_matrix       = None
        self._singular_values  = None
        if theta_scaling in THETA_SCALINGS:
            self._theta_scaling    = theta_scaling
        else:
            raise ValueError(f"Unsupported theta scaling type: {theta_scaling}")
        self._theta_model      = ThetaModel(input_features, num_moments, theta_model_type, theta_model_settings)

    def _add_theta_to_collection(self, collection: SeriesCollection, scaling_vector: np.ndarray, num_procs: int = 1) -> None:
        targets = [np.concatenate([state[name] for name in self.predicted_feature_names])
                   for series in collection for state in series]
        thetas  = project_theta(targets, self._pod_matrix, self._constraints, scaling_vector, num_procs)
        ThetaModel.set_targets(collection, thetas)

    def _add_theta(self,
                   train_data: SeriesCollection,
                   validation_data: Optional[SeriesCollection] = None,
                   scaling_vector: np.ndarray = None,
                   num_procs: int = 1) -> None:
        if scaling_vector is None:
            scaling_vector = self._scaling_vector()

        self._add_theta_to_collection(train_data, scaling_vector, num_procs)
        if validation_data is not None:
            self._add_theta_to_collection(validation_data, scaling_vector, num_procs)

    def _compute_pod(self, data: SeriesCollection) -> None:
        A = np.vstack([np.array(series) for series in self._get_targets(data)])
        self._max_svd_size = self._max_svd_size or A.shape[0]
        self._pod_matrix, self._singular_values = compute_pod_basis(A, self.num_moments, self.max_svd_size)

    def _scaling_vector(self) -> np.ndarray:
        return theta_scaling_vector(self._theta_scaling, self._singular_values)

    def train(self,
              train_data: SeriesCollection,
              validation_data: Optional[SeriesCollection] = None,
              num_procs: int = 1,
              *,
              validation_split: float = 0.2,
              validation_seed: Optional[int] = 42) -> None:
        """Train the enhanced POD and theta models using the supplied validation policy."""

        # compute pod matrix
        self._compute_pod(train_data)

        # add theta_xi to collections for training
        # TODO: disable training theta in parallel because it collides with lightgbm somehow
        self._add_theta(train_data, validation_data, scaling_vector=self._scaling_vector())

        # train models
        self._theta_model.train(train_data,
                                validation_data,
                                num_procs=num_procs,
                                validation_split=validation_split,
                                validation_seed=validation_seed)

    def _predict_weights_one(self, state_series: np.ndarray) -> np.ndarray:
        assert self.isTrained

        scaling_vector = self._scaling_vector()

        n_timesteps = state_series.shape[0]
        weights = []
        for i in range(n_timesteps):
            # Denormalize theta by multiplying by singular values
            weights.append(self._theta_model.predict(state_series[i,:]) * scaling_vector)

        return np.asarray(weights).reshape(n_timesteps, -1)

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        return self._predict_weights_one(state_series) @ self._pod_matrix.T

    def write_model_to_hdf5(self, h5_group):
        """ A method for saving a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            The opened HDF5 group or file to which the model should be written
        """
        super().write_model_to_hdf5(h5_group)
        h5_group.create_dataset('num_moments', data=self.num_moments)
        h5_group.create_dataset('pod_mat', data=self._pod_matrix)
        h5_group.create_dataset('singular_values', data=self._singular_values)
        h5_group.create_dataset('num_constraints', data=len(self._constraints))
        h5_group.create_dataset('max_svd_size', data=self._max_svd_size)
        h5_group.create_dataset('theta_scaling', data=self._theta_scaling)

        for i, (gamma, W) in enumerate(self._constraints):
            h5_group.create_dataset(f'constraint_{i+1}_gamma', data=gamma)
            h5_group.create_dataset(f'constraint_{i+1}_W', data=W)

        self._theta_model.write_to_hdf5(h5_group)

    def load_model(self, h5_group: h5py.Group) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            An open HDF5 group or file object from which to load the model
        """
        super().load_model(h5_group)

        self._num_moments    = int(h5_group['num_moments'][()])
        self._pod_matrix     = h5_group['pod_mat'][()]
        self._singular_values = h5_group['singular_values'][()]
        # Load theta_scaling with backward compatibility
        if 'theta_scaling' in h5_group:
            theta_scaling_bytes = bytes(h5_group['theta_scaling'][()])
            self._theta_scaling = theta_scaling_bytes.decode('utf-8')
        else:
            self._theta_scaling = 'ones'  # default for old models

        num_constraints = int(h5_group['num_constraints'][()])
        self._constraints = [(h5_group[f'constraint_{i+1}_gamma'][()],
                                  h5_group[f'constraint_{i+1}_W'][()]) for i in range(num_constraints)]

        self._theta_model = ThetaModel.read_from_hdf5(h5_group, self.input_features, self.num_moments)

    @classmethod
    def read_from_file(cls: EnhancedPODStrategy, file_name: str) -> Type[EnhancedPODStrategy]:
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
            theta_model_type = bytes(h5_file['theta_model_type'][()]).decode('utf-8')
        new_pod = cls(
            input_features={'dummy_input': NoProcessing()},
            predicted_features='dummy_output',
            num_moments=r,
            theta_model_type=theta_model_type
        )
        new_pod.load_model(h5py.File(file_name, "r"))

        return new_pod

    @classmethod
    def _from_params_dict(cls,
                          params: Dict,
                          input_features: FeatureSpec,
                          predicted_features: FeatureSpec) -> EnhancedPODStrategy:
        constraints_payload = params.get("constraints", [])
        assert isinstance(constraints_payload, list), "'constraints' must be a list"
        constraints = []
        for item in constraints_payload:
            assert isinstance(item, dict), "constraint entries must be dicts"
            gamma = float(item["gamma"])
            W = np.asarray(item["W"], dtype=float)
            constraints.append((gamma, W))

        instance = cls(input_features=input_features,
                       predicted_features=predicted_features,
                       theta_model_type=params.get("theta_model_type", "GBM"),
                       max_svd_size=params.get("max_svd_size"),
                       num_moments=params.get("num_moments", 1),
                       constraints=constraints,
                       theta_scaling=params.get("theta_scaling", "sqrt_singular_values"),
                       theta_model_settings={})

        instance._theta_model = ThetaModel.from_dict(params.get("theta_models"),
                                                     input_features,
                                                     instance.num_moments,
                                                     instance.theta_model_type)
        return instance

    def _params_to_dict(self):
        constraints_payload = [{"gamma": float(gamma), "W": np.asarray(W).tolist()}
                               for gamma, W in self._constraints]
        return {
            "theta_model_type": self.theta_model_type,
            "theta_scaling": self._theta_scaling,
            "max_svd_size": self._max_svd_size,
            "num_moments": self._num_moments,
            "constraints": constraints_payload,
            "theta_models": self._theta_model.to_dict(),
        }
