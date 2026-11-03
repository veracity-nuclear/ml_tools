"""Shared POD weight (theta) machinery for POD-based strategies.

These utilities compute a POD basis, project target vectors onto it to obtain
(scaled) POD weights ``theta``, and regress ``theta`` from input features via a
:class:`ThetaModel`.  They are shared by :class:`EnhancedPODStrategy` and other
strategies that work with regressed POD weights.
"""
from __future__ import annotations

from itertools import repeat
from typing import Optional, Dict, List, Tuple, Sequence, Any
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
from sklearn.linear_model import LinearRegression

from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
from ml_tools.model.gbm_strategy import GBMStrategy
from ml_tools.model.nn_strategy import NNStrategy
from ml_tools.model.sklearn_strategy import SklearnStrategy

THETA_SCALINGS = ('singular_values', 'sqrt_singular_values',
                  'max_singular_value', 'max_sqrt_singular_value',
                  'ones')


def compute_pod_basis(targets:      np.ndarray,
                      num_moments:  int,
                      max_svd_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute the POD basis of a set of target vectors

    Parameters
    ----------
    targets : np.ndarray
        Target vectors of shape (num_samples, output_dim)
    num_moments : int
        The number of POD moments to keep
    max_svd_size : Optional[int]
        The maximum number of samples to use for the SVD.  If there are more samples,
        a random subset of this size is used.  If None, all samples are used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The POD moments of shape (output_dim, num_moments) and the corresponding
        singular values of shape (num_moments,)
    """
    if max_svd_size is not None and targets.shape[0] > max_svd_size:
        targets = targets[np.random.choice(targets.shape[0], max_svd_size, replace=False), :]
    u, L, _ = np.linalg.svd(targets.T, full_matrices=False)
    return u[:, :num_moments], L[:num_moments]


def theta_scaling_vector(theta_scaling: str, singular_values: np.ndarray) -> np.ndarray:
    """ The vector by which POD weights are divided to obtain normalized theta

    Parameters
    ----------
    theta_scaling : str
        The scaling type.  One of ``THETA_SCALINGS``.
    singular_values : np.ndarray
        The singular values of the POD moments

    Returns
    -------
    np.ndarray
        The scaling vector of shape (num_moments,)
    """
    num_moments = len(singular_values)
    if theta_scaling == 'singular_values':
        return singular_values + 1e-12
    if theta_scaling == 'sqrt_singular_values':
        return np.sqrt(singular_values + 1e-12)
    if theta_scaling == 'max_singular_value':
        return singular_values[0] * np.ones(num_moments)
    if theta_scaling == 'max_sqrt_singular_value':
        return np.sqrt(singular_values[0]) * np.ones(num_moments)
    if theta_scaling == 'ones':
        return np.ones(num_moments)
    raise ValueError(f"Unsupported theta scaling type: {theta_scaling}")


def _project_theta_one(target:         np.ndarray,
                       pod_matrix:     np.ndarray,
                       constraints:    List[Tuple[float, np.ndarray]],
                       scaling_vector: np.ndarray) -> np.ndarray:
    if len(constraints) == 0:
        theta_star = pod_matrix.T @ target
    else:
        A = [pod_matrix]
        b = [target]
        for gamma, W in constraints:
            A.append(gamma * W @ pod_matrix)
            b.append(gamma * W @ target)
        theta_star, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)

    return theta_star / scaling_vector


def project_theta(targets:        Sequence[np.ndarray],
                  pod_matrix:     np.ndarray,
                  constraints:    List[Tuple[float, np.ndarray]],
                  scaling_vector: np.ndarray,
                  num_procs:      int = 1) -> List[np.ndarray]:
    """ Project target vectors onto a POD basis to obtain normalized theta

    Without constraints, ``theta = pod_matrix.T @ target / scaling_vector``.  Each
    ``(gamma, W)`` constraint adds ``gamma * W @ pod_matrix @ theta ~= gamma * W @ target``
    to a least-squares solve.

    Parameters
    ----------
    targets : Sequence[np.ndarray]
        The target vectors to project, each of shape (output_dim,)
    pod_matrix : np.ndarray
        The POD moments of shape (output_dim, num_moments)
    constraints : List[Tuple[float, np.ndarray]]
        Pairs of (gamma, W) constraints for computing theta
    scaling_vector : np.ndarray
        The vector by which the POD weights are divided
    num_procs : int
        The number of parallel processors to use

    Returns
    -------
    List[np.ndarray]
        The normalized theta for each target, each of shape (num_moments,)
    """
    assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
    if num_procs > 1:
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            return list(executor.map(_project_theta_one, targets,
                                     repeat(pod_matrix), repeat(constraints), repeat(scaling_vector)))
    return [_project_theta_one(target, pod_matrix, constraints, scaling_vector) for target in targets]


class ThetaModel:
    """ A regression model from input features to normalized POD weights (theta)

    GBM theta models use a separate model per moment, predicting features
    ``theta-1``, ``theta-2``, ...  NN and sklearn theta models use a single model
    predicting the vector feature ``theta``.

    Parameters
    ----------
    input_features : FeatureSpec
        The features to use as input for the theta model(s)
    num_moments : int
        The number of POD moments to predict
    model_type : Optional[str]
        The type of model to use.  Supported types are 'GBM', 'NN', and 'sklearn'.  Default is 'GBM'.
    model_settings : Optional[Dict[str, Any]]
        Additional settings to pass to the model constructor(s)

    Attributes
    ----------
    model_type : str
        The upper-cased model type
    num_moments : int
        The number of POD moments predicted
    models : List[PredictionStrategy]
        The underlying prediction strategies
    """

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def num_moments(self) -> int:
        return self._num_moments

    @property
    def models(self) -> List[PredictionStrategy]:
        return self._models

    @property
    def is_vector_model(self) -> bool:
        """ Whether a single model predicts all moments at once """
        return self._model_type in ["NN", "SKLEARN"]

    def __init__(self,
                 input_features: FeatureSpec,
                 num_moments:    int,
                 model_type:     Optional[str] = 'GBM',
                 model_settings: Optional[Dict[str, Any]] = None) -> None:
        self._num_moments = num_moments
        self._model_type  = model_type.upper() if model_type else 'GBM'
        model_settings    = dict(model_settings) if model_settings is not None else {}
        if self._model_type == "GBM":
            self._models = [GBMStrategy(input_features, f'theta-{i+1}', **model_settings)
                            for i in range(num_moments)]
        elif self._model_type == "NN":
            # NN can predict all moments at once
            self._models = [NNStrategy(input_features, 'theta', **model_settings)]
        elif self._model_type == "SKLEARN":
            estimator = model_settings.pop('estimator', LinearRegression)
            self._models = [SklearnStrategy(input_features, 'theta', estimator, **model_settings)]
        else:
            raise ValueError(f"Unsupported theta model type: {self._model_type}")

    @staticmethod
    def set_targets(collection: SeriesCollection, thetas: Sequence[np.ndarray]) -> None:
        """ Add theta target features to each state of a collection (in place)

        Parameters
        ----------
        collection : SeriesCollection
            The collection to add the theta features to
        thetas : Sequence[np.ndarray]
            The normalized theta for each state, in series-major order
        """
        states = [state for series in collection for state in series]
        assert len(states) == len(thetas), f"Expected {len(states)} thetas, got {len(thetas)}"
        for state, theta in zip(states, thetas):
            state.features['theta'] = theta
            for i, theta_i in enumerate(theta):
                state.features[f'theta-{i+1}'] = [theta_i]

    def train(self,
              train_data: SeriesCollection,
              validation_data: Optional[SeriesCollection] = None,
              num_procs: int = 1,
              *,
              validation_split: float = 0.2,
              validation_seed: Optional[int] = 42) -> None:
        """ Train the theta model(s) on collections whose states carry theta targets

        See :meth:`set_targets` and :meth:`PredictionStrategy.train`.
        """
        for model in self._models:
            model.train(train_data,
                        validation_data,
                        num_procs=num_procs,
                        validation_split=validation_split,
                        validation_seed=validation_seed)

    def predict(self, processed_input: np.ndarray) -> np.ndarray:
        """ Predict normalized theta for a single preprocessed input state

        Parameters
        ----------
        processed_input : np.ndarray
            The preprocessed input features of one state, of shape (input_dim,)

        Returns
        -------
        np.ndarray
            The normalized theta of shape (num_moments,)
        """
        X = processed_input.reshape(1, -1)
        if self.is_vector_model:
            theta = self._models[0].predict_processed_inputs(X[np.newaxis, :])[0].flatten()
        else:
            theta = np.asarray([model.predict_processed_inputs(X[np.newaxis, :])[0]
                                for model in self._models])
        return theta.flatten()

    def write_to_hdf5(self, h5_group: h5py.Group) -> None:
        """ Write the theta model(s) to an HDF5 group

        Parameters
        ----------
        h5_group : h5py.Group
            The opened, writeable HDF5 group or file handle
        """
        h5_group.create_dataset('theta_model_type', data=self._model_type)
        if self.is_vector_model:
            self._models[0].write_model_to_hdf5(h5_group.require_group('theta'))
        else:
            for i, model in enumerate(self._models):
                model.write_model_to_hdf5(h5_group.require_group(f'theta-{i+1}'))

    @classmethod
    def read_from_hdf5(cls,
                       h5_group:       h5py.Group,
                       input_features: FeatureSpec,
                       num_moments:    int) -> ThetaModel:
        """ Read theta model(s) written by :meth:`write_to_hdf5`

        Parameters
        ----------
        h5_group : h5py.Group
            The opened HDF5 group or file handle
        input_features : FeatureSpec
            The input features of the theta model(s)
        num_moments : int
            The number of POD moments predicted

        Returns
        -------
        ThetaModel
            The loaded theta model
        """
        model_type = bytes(h5_group['theta_model_type'][()]).decode('utf-8')
        # sklearn uses LinearRegression as a placeholder; the estimator is replaced by load_model
        instance = cls(input_features, num_moments, model_type)
        if instance.is_vector_model:
            instance._models[0].load_model(h5_group['theta'])
        else:
            for i, model in enumerate(instance._models):
                model.load_model(h5_group[f'theta-{i+1}'])
        return instance

    def to_dict(self) -> List[Dict[str, Any]]:
        """ Serialize the underlying model(s) to canonical strategy payloads """
        return [model.to_dict() for model in self._models]

    @classmethod
    def from_dict(cls,
                  payload:        List[Dict[str, Any]],
                  input_features: FeatureSpec,
                  num_moments:    int,
                  model_type:     Optional[str] = 'GBM') -> ThetaModel:
        """ Construct a theta model from payloads written by :meth:`to_dict`

        Parameters
        ----------
        payload : List[Dict[str, Any]]
            The serialized underlying model(s)
        input_features : FeatureSpec
            The input features of the theta model(s)
        num_moments : int
            The number of POD moments predicted
        model_type : Optional[str]
            The type of model

        Returns
        -------
        ThetaModel
            A configured, untrained theta model
        """
        assert isinstance(payload, list) and len(payload) > 0, \
            "'theta_models' must be a non-empty list"
        instance = cls(input_features, num_moments, model_type)
        if instance.is_vector_model:
            assert len(payload) == 1, \
                f"Expected exactly 1 theta model for {instance.model_type}"
        else:
            assert len(payload) == num_moments, \
                f"Expected {num_moments} theta models for GBM"
        instance._models = [PredictionStrategy.from_dict(model_payload) for model_payload in payload]
        return instance
