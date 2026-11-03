from __future__ import annotations

import os
import warnings
from typing import Optional, Dict, Type, List, Any
import numpy as np
import h5py

from ml_tools.model.state import State, StateSeries, SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
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

INITIALIZATIONS = ('project', 'infer')


@register_prediction_strategy()  # registers under 'DMDPODStrategy' by default
class DMDPODStrategy(PODBasedStrategy):
    """ A concrete class for a DMD-on-POD prediction strategy

    This prediction strategy evolves the POD weights of the predicted features through
    time with a linear step operator fit by dynamic mode decomposition with control:

    .. math::

        w_t = A w_{t-1} + B u_t + c, \\qquad y_t = U w_t

    where ``U`` are the POD moments, ``w`` the POD weights, ``u`` the processed input
    features, and ``y`` the processed predicted features.  All time steps are assumed
    to be uniform; a varying step size can be supplied as an ordinary input feature.

    Each series starts from initial weights ``w_0``, either projected from a known
    initial state (``initialization='project'``) or estimated from the initial inputs
    by a theta model (``initialization='infer'``).  :meth:`predict` has no known
    initial state and therefore always uses the theta model.  For incremental
    prediction, see :meth:`create_trajectory`.

    Parameters
    ----------
    input_features : FeatureSpec
        The features to use as input for this model
    predicted_features : FeatureSpec
        The string or FeatureSpec specifying the feature(s) to be predicted
    num_moments : int
        The number of POD moments to use
    max_svd_size : Optional[int]
        The maximum number of training states to use for the SVD.  If None, all states are used.
    theta_scaling : Optional[str]
        The scaling used to normalize the POD weights for regression. Supported types are
        'singular_values', 'sqrt_singular_values', 'max_singular_value', 'max_sqrt_singular_value',
        and 'ones'. Default is 'sqrt_singular_values'.
    regularization : float
        Ridge penalty on the fit of the step operators, applied in normalized weight space
    fit_intercept : bool
        Whether to fit the constant term ``c``, which captures constant forcing in the dynamics.
        Note that affine predicted-feature processors (e.g. MinMaxNormalize) add a constant
        direction in processed space, which typically needs one additional POD moment.
    initial_theta_model_type : Optional[str]
        The type of model used to estimate initial weights from inputs.  Supported types are
        'GBM', 'NN', and 'sklearn'.  If None, no model is trained, :meth:`predict` is unavailable,
        and trajectories must be initialized by projection.  Default is 'GBM'.
    initial_theta_model_settings : Optional[Dict[str, Any]]
        Additional settings to pass to the initial theta model constructor

    Attributes
    ----------
    num_moments : int
        The number of POD moments
    moments : np.ndarray
        The POD moments of shape (output_dim, num_moments)
    step_operator : np.ndarray
        The operator ``A`` of shape (num_moments, num_moments)
    input_operator : np.ndarray
        The operator ``B`` of shape (num_moments, input_dim)
    offset : np.ndarray
        The constant term ``c`` of shape (num_moments,)
    spectral_radius : float
        The largest eigenvalue magnitude of ``A``.  Values >= 1 indicate the autonomous
        dynamics grow over long rollouts.
    """

    class Trajectory:
        """ The evolving state of a single DMD-on-POD prediction

        Trajectories are created by :meth:`DMDPODStrategy.create_trajectory`.  The trajectory
        references, but does not own, the strategy that created it.

        Parameters
        ----------
        strategy : DMDPODStrategy
            The trained strategy that evolves this trajectory
        weights : np.ndarray
            The current POD weights of shape (num_moments,)
        step : int
            The current time step index

        Attributes
        ----------
        strategy : DMDPODStrategy
            The strategy that evolves this trajectory
        weights : np.ndarray
            The current POD weights of shape (num_moments,)
        step : int
            The current time step index
        state : State
            The current predicted features (post-processed)
        """

        @property
        def strategy(self) -> DMDPODStrategy:
            return self._strategy

        @property
        def weights(self) -> np.ndarray:
            return self._weights.copy()

        @property
        def step(self) -> int:
            return self._step

        @property
        def state(self) -> State:
            return self._strategy._postprocess_state(self._strategy.moments @ self._weights)  # pylint: disable=protected-access

        def __init__(self, strategy: DMDPODStrategy, weights: np.ndarray, step: int = 0) -> None:
            self._strategy = strategy
            self._weights  = np.array(weights, dtype=float)
            self._step     = step

        def predict_next(self, inputs: State) -> State:
            """ Advance one step and predict the next state

            Parameters
            ----------
            inputs : State
                A state containing the input features of the next step

            Returns
            -------
            State
                The predicted features of the next step (post-processed)
            """
            self.predict_next_processed_inputs(self._strategy._preprocess_state(inputs))  # pylint: disable=protected-access
            return self.state

        def predict_next_processed_inputs(self, processed_inputs: np.ndarray) -> np.ndarray:
            """ Advance one step from preprocessed inputs

            Parameters
            ----------
            processed_inputs : np.ndarray
                The preprocessed input features of the next step, of shape (input_dim,)

            Returns
            -------
            np.ndarray
                The predicted features of the next step (processed), of shape (output_dim,)
            """
            self._weights = self._strategy._step_weights(self._weights, processed_inputs)  # pylint: disable=protected-access
            self._step   += 1
            return self._strategy.moments @ self._weights

        def copy(self) -> DMDPODStrategy.Trajectory:
            """ An independent copy of this trajectory, e.g. for branching what-if predictions """
            return type(self)(self._strategy, self._weights, self._step)

    @property
    def num_moments(self) -> int:
        return self._num_moments

    @property
    def max_svd_size(self) -> Optional[int]:
        return self._max_svd_size

    @property
    def theta_scaling(self) -> str:
        return self._theta_scaling

    @property
    def regularization(self) -> float:
        return self._regularization

    @property
    def fit_intercept(self) -> bool:
        return self._fit_intercept

    @property
    def initial_theta_model_type(self) -> Optional[str]:
        return None if self._initial_theta_model is None else self._initial_theta_model.model_type

    @property
    def isTrained(self) -> bool:
        return self._pod_matrix is not None and self._A is not None

    @property
    def moments(self) -> np.ndarray:
        assert self.isTrained
        return self._pod_matrix

    @property
    def step_operator(self) -> np.ndarray:
        assert self.isTrained
        return self._A

    @property
    def input_operator(self) -> np.ndarray:
        assert self.isTrained
        return self._B

    @property
    def offset(self) -> np.ndarray:
        assert self.isTrained
        return self._c

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.step_operator))))

    def __init__(self,
                 input_features:               FeatureSpec,
                 predicted_features:           FeatureSpec,
                 num_moments:                  int = 1,
                 max_svd_size:                 Optional[int] = None,
                 theta_scaling:                Optional[str] = 'sqrt_singular_values',
                 regularization:               float = 0.0,
                 fit_intercept:                bool = True,
                 initial_theta_model_type:     Optional[str] = 'GBM',
                 initial_theta_model_settings: Optional[Dict[str, Any]] = None) -> None:

        super().__init__()
        self.input_features     = input_features
        self.predicted_features = predicted_features
        assert len(self.input_features) > 0, "DMDPODStrategy requires at least one input feature"
        assert num_moments > 0, f"num_moments must be > 0, got {num_moments}"
        assert regularization >= 0., f"regularization must be >= 0, got {regularization}"
        if theta_scaling not in THETA_SCALINGS:
            raise ValueError(f"Unsupported theta scaling type: {theta_scaling}")

        self._num_moments     = num_moments
        self._max_svd_size    = max_svd_size
        self._theta_scaling   = theta_scaling
        self._regularization  = float(regularization)
        self._fit_intercept   = bool(fit_intercept)
        self._pod_matrix      = None
        self._singular_values = None
        self._A               = None
        self._B               = None
        self._c               = None
        self._initial_theta_model = None if initial_theta_model_type is None else \
            ThetaModel(input_features, num_moments, initial_theta_model_type, initial_theta_model_settings)

    def __eq__(self, other: object) -> bool:
        return (super().__eq__(other) and
                isinstance(other, DMDPODStrategy) and
                self.num_moments              == other.num_moments and
                self.max_svd_size             == other.max_svd_size and
                self.theta_scaling            == other.theta_scaling and
                self.regularization           == other.regularization and
                self.fit_intercept            == other.fit_intercept and
                self.initial_theta_model_type == other.initial_theta_model_type)

    def _scaling_vector(self) -> np.ndarray:
        return theta_scaling_vector(self._theta_scaling, self._singular_values)

    def train(self,
              train_data: SeriesCollection,
              validation_data: Optional[SeriesCollection] = None,
              num_procs: int = 1,
              *,
              validation_split: float = 0.2,
              validation_seed: Optional[int] = 42) -> None:
        """Train the POD basis, step operators, and initial theta model.

        The step operators are fit by least squares over consecutive steps within each
        series, so only series of length >= 2 contribute to them.  The POD basis and the
        initial theta model use every state.  Validation data is only used by the
        initial theta model.
        """
        lengths = [len(series) for series in train_data]
        targets = self._get_targets(train_data, num_procs=num_procs)
        inputs  = PredictionStrategy.preprocess_features(train_data, self.input_features, num_procs=num_procs)

        # POD basis over every (unpadded) training state
        Y = np.vstack([series[:n] for series, n in zip(targets, lengths)])
        self._pod_matrix, self._singular_values = compute_pod_basis(Y, self.num_moments, self.max_svd_size)
        scaling = self._scaling_vector()

        # Normalized weights of every state, in series-major order
        thetas = np.asarray(project_theta(list(Y), self._pod_matrix, [], scaling, num_procs))
        starts = np.cumsum([0] + lengths)
        thetas = [thetas[starts[i]:starts[i+1]] for i in range(len(lengths))]

        self._fit_step_operators(thetas, [series[:n] for series, n in zip(inputs, lengths)], scaling)

        if self.spectral_radius >= 1.:
            warnings.warn(f"DMDPODStrategy step operator has spectral radius {self.spectral_radius:.4g} >= 1; "
                          "long rollouts may grow without bound.", RuntimeWarning, stacklevel=2)

        if self._initial_theta_model is not None:
            theta_train = self._theta_target_collection(train_data, np.vstack(thetas))
            theta_validation = None
            if validation_data is not None:
                validation_targets = self._get_targets(validation_data, num_procs=num_procs)
                Y_validation = np.vstack([series[:len(s)] for series, s in zip(validation_targets, validation_data)])
                theta_validation = self._theta_target_collection(
                    validation_data, np.asarray(project_theta(list(Y_validation), self._pod_matrix, [], scaling, num_procs)))
            self._initial_theta_model.train(theta_train,
                                            theta_validation,
                                            num_procs=num_procs,
                                            validation_split=validation_split,
                                            validation_seed=validation_seed)

    def _fit_step_operators(self,
                            thetas:  List[np.ndarray],
                            inputs:  List[np.ndarray],
                            scaling: np.ndarray) -> None:
        """ Fit theta_t = A~ theta_{t-1} + B~ u_t + c~ over all consecutive pairs, then convert to weight space """
        prev, nxt, controls = [], [], []
        for theta, u in zip(thetas, inputs):
            prev.append(theta[:-1])
            nxt.append(theta[1:])
            controls.append(u[1:])
        prev, nxt, controls = np.vstack(prev), np.vstack(nxt), np.vstack(controls)
        if prev.shape[0] == 0:
            raise ValueError("DMDPODStrategy requires at least one training series of length >= 2")

        columns = [prev, controls] + ([np.ones((prev.shape[0], 1))] if self._fit_intercept else [])
        Z = np.hstack(columns)
        if self._regularization > 0.:
            penalty = self._regularization * np.eye(Z.shape[1])
            if self._fit_intercept:
                penalty[-1, -1] = 0.  # do not shrink the constant term
            G = np.linalg.solve(Z.T @ Z + penalty, Z.T @ nxt)
        else:
            G, *_ = np.linalg.lstsq(Z, nxt, rcond=None)

        r, m = prev.shape[1], controls.shape[1]
        A_scaled = G[:r].T
        B_scaled = G[r:r+m].T
        c_scaled = G[r+m] if self._fit_intercept else np.zeros(r)

        # weights w = scaling * theta
        self._A = scaling[:, np.newaxis] * A_scaled / scaling[np.newaxis, :]
        self._B = scaling[:, np.newaxis] * B_scaled
        self._c = scaling * c_scaled

    def _theta_target_collection(self, collection: SeriesCollection, thetas: np.ndarray) -> SeriesCollection:
        """ A copy of the collection's input features with theta targets, leaving the original untouched """
        copy = SeriesCollection([StateSeries([State({name: state[name] for name in self.input_features})
                                              for state in series]) for series in collection])
        ThetaModel.set_targets(copy, list(thetas))
        return copy

    def _step_weights(self, weights: np.ndarray, processed_inputs: np.ndarray) -> np.ndarray:
        return self._A @ weights + self._B @ processed_inputs + self._c

    def _infer_initial_weights(self, processed_inputs: np.ndarray) -> np.ndarray:
        if self._initial_theta_model is None:
            raise RuntimeError("DMDPODStrategy was built without an initial theta model "
                               "(initial_theta_model_type=None), so initial weights cannot be inferred from inputs. "
                               "Use create_trajectory(initial_state, initialization='project') instead.")
        return self._initial_theta_model.predict(processed_inputs) * self._scaling_vector()

    def _preprocess_state(self, state: State) -> np.ndarray:
        return PredictionStrategy.preprocess_features(SeriesCollection([StateSeries([state])]), self.input_features)[0, 0]

    def _postprocess_state(self, processed: np.ndarray) -> State:
        return PredictionStrategy.postprocess_features(np.asarray(processed)[np.newaxis, np.newaxis, :],
                                                       [1],
                                                       self.predicted_feature_names,
                                                       self._predicted_feature_sizes,
                                                       self.predicted_features)[0][0]

    def create_trajectory(self, initial_state: State, initialization: str = 'project') -> DMDPODStrategy.Trajectory:
        """ Create a trajectory starting at an initial state

        Parameters
        ----------
        initial_state : State
            The state at step 0.  For 'project' it must contain the predicted features;
            for 'infer' it must contain the input features.
        initialization : str
            'project' to project the known predicted features of ``initial_state`` onto the
            POD moments, or 'infer' to estimate the initial weights from its input features
            with the initial theta model.

        Returns
        -------
        DMDPODStrategy.Trajectory
            A trajectory at step 0
        """
        assert self.isTrained
        if initialization == 'project':
            processed = PredictionStrategy.preprocess_features(SeriesCollection([StateSeries([initial_state])]),
                                                               self.predicted_features)[0, 0]
            weights = self._pod_matrix.T @ processed
        elif initialization == 'infer':
            weights = self._infer_initial_weights(self._preprocess_state(initial_state))
        else:
            raise ValueError(f"Unsupported initialization: {initialization}. Supported: {INITIALIZATIONS}")
        return DMDPODStrategy.Trajectory(self, weights)

    def _predict_weights_one(self, state_series: np.ndarray) -> np.ndarray:
        assert self.isTrained
        trajectory = DMDPODStrategy.Trajectory(self, self._infer_initial_weights(state_series[0]))
        weights = [trajectory.weights]
        for processed_inputs in state_series[1:]:
            trajectory.predict_next_processed_inputs(processed_inputs)
            weights.append(trajectory.weights)
        return np.asarray(weights)

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        return self._predict_weights_one(state_series) @ self._pod_matrix.T

    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
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
        h5_group.create_dataset('theta_scaling', data=self._theta_scaling)
        h5_group.create_dataset('regularization', data=self._regularization)
        h5_group.create_dataset('fit_intercept', data=self._fit_intercept)
        h5_group.create_dataset('step_operator', data=self._A)
        h5_group.create_dataset('input_operator', data=self._B)
        h5_group.create_dataset('offset', data=self._c)
        if self._max_svd_size is not None:
            h5_group.create_dataset('max_svd_size', data=self._max_svd_size)
        if self._initial_theta_model is not None:
            self._initial_theta_model.write_to_hdf5(h5_group.require_group('initial_theta_model'))

    def load_model(self, h5_group: h5py.Group) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            An open HDF5 group or file object from which to load the model
        """
        super().load_model(h5_group)
        self._num_moments     = int(h5_group['num_moments'][()])
        self._pod_matrix      = h5_group['pod_mat'][()]
        self._singular_values = h5_group['singular_values'][()]
        self._theta_scaling   = bytes(h5_group['theta_scaling'][()]).decode('utf-8')
        self._regularization  = float(h5_group['regularization'][()])
        self._fit_intercept   = bool(h5_group['fit_intercept'][()])
        self._A               = h5_group['step_operator'][()]
        self._B               = h5_group['input_operator'][()]
        self._c               = h5_group['offset'][()]
        self._max_svd_size    = int(h5_group['max_svd_size'][()]) if 'max_svd_size' in h5_group else None
        self._initial_theta_model = ThetaModel.read_from_hdf5(h5_group['initial_theta_model'],
                                                              self.input_features,
                                                              self.num_moments) \
                                    if 'initial_theta_model' in h5_group else None

    @classmethod
    def read_from_file(cls: DMDPODStrategy, file_name: str) -> Type[DMDPODStrategy]:
        """ A basic factory method for building a DMD-on-POD Strategy from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the file from which to read the model

        Returns
        -------
        DMDPODStrategy:
            The model from the hdf5 file
        """
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        assert os.path.exists(file_name), f"file name = {file_name}"

        new_dmd = cls(input_features={'dummy_input': NoProcessing()},
                      predicted_features='dummy_output',
                      initial_theta_model_type=None)
        with h5py.File(file_name, "r") as h5_file:
            new_dmd.load_model(h5_file)
        return new_dmd

    @classmethod
    def _from_params_dict(cls,
                          params: Dict,
                          input_features: FeatureSpec,
                          predicted_features: FeatureSpec) -> DMDPODStrategy:
        initial_theta_model_type = params.get("initial_theta_model_type", "GBM")
        instance = cls(input_features=input_features,
                       predicted_features=predicted_features,
                       num_moments=params.get("num_moments", 1),
                       max_svd_size=params.get("max_svd_size"),
                       theta_scaling=params.get("theta_scaling", "sqrt_singular_values"),
                       regularization=params.get("regularization", 0.0),
                       fit_intercept=params.get("fit_intercept", True),
                       initial_theta_model_type=initial_theta_model_type)
        if params.get("initial_theta_models") is not None:
            instance._initial_theta_model = ThetaModel.from_dict(params["initial_theta_models"],
                                                                 input_features,
                                                                 instance.num_moments,
                                                                 initial_theta_model_type)
        return instance

    def _params_to_dict(self) -> Dict[str, Any]:
        return {
            "num_moments": self._num_moments,
            "max_svd_size": self._max_svd_size,
            "theta_scaling": self._theta_scaling,
            "regularization": self._regularization,
            "fit_intercept": self._fit_intercept,
            "initial_theta_model_type": self.initial_theta_model_type,
            "initial_theta_models": None if self._initial_theta_model is None else self._initial_theta_model.to_dict(),
        }
