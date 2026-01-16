from __future__ import annotations
from typing import Dict, Optional, Type, Any
import importlib
import json
import pickle

import h5py
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
from ml_tools.model import register_prediction_strategy

@register_prediction_strategy()
class SklearnStrategy(PredictionStrategy):
    """A wrapper for scikit-learn estimators as a prediction strategy.

    This prediction strategy wraps any scikit-learn estimator that implements
    the fit() and predict() methods. It is intended for use with static State-Points,
    meaning non-temporal series, or State Series with series lengths of one.

    Parameters
    ----------
    input_features : FeatureSpec
        Input feature/processor pairs (Dict) or feature name(s) (str/List[str],
        automatically mapped to NoProcessing).
    predicted_features : FeatureSpec
        Output feature/processor pairs (Dict) or feature name(s) (str/List[str],
        automatically mapped to NoProcessing).
    estimator : Type[BaseEstimator]
        A scikit-learn estimator class (e.g., RandomForestRegressor, SVR, etc.).
        The class will be instantiated with estimator_args.
    estimator_args : Dict[str, Any], optional
        Dictionary of keyword arguments to pass to the estimator class constructor.

    Attributes
    ----------
    estimator : BaseEstimator
        The scikit-learn estimator instance used for training and prediction.
    estimator_class : Type[BaseEstimator]
        The class of the estimator.
    estimator_args : Dict[str, Any]
        The parameters used to initialize the estimator.
    isTrained : bool
        Whether the model has been trained.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from ml_tools.model.sklearn_strategy import SklearnStrategy
    >>> 
    >>> strategy = SklearnStrategy(
    ...     input_features=['feature1', 'feature2'],
    ...     predicted_features=['target'],
    ...     estimator=RandomForestRegressor,
    ...     estimator_args={'n_estimators': 100, 'max_depth': 10}
    ... )
    >>> 
    >>> strategy.train(train_data)
    >>> predictions = strategy.predict(test_data)
    """

    @property
    def estimator(self) -> Any:
        """The wrapped scikit-learn estimator instance."""
        return self._estimator

    @property
    def estimator_class(self) -> Optional[Type]:
        """The class of the estimator."""
        return self._estimator_class

    @property
    def estimator_args(self) -> Dict[str, Any]:
        """The parameters used to initialize the estimator."""
        return self._estimator_args.copy() if self._estimator_args else {}

    @property
    def isTrained(self) -> bool:
        """Check if the estimator has been trained.

        Returns
        -------
        bool
            True if the estimator has been fitted, False otherwise.
        """
        if self._estimator is None:
            return False
        try:
            check_is_fitted(self._estimator)
            return True
        except Exception:
            return False

    def __init__(self,
                 input_features: FeatureSpec,
                 predicted_features: FeatureSpec,
                 estimator: Type[BaseEstimator],
                 estimator_args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the SklearnStrategy.

        Parameters
        ----------
        input_features : FeatureSpec
            Input features specification.
        predicted_features : FeatureSpec
            Predicted features specification.
        estimator : Type[BaseEstimator]
            A scikit-learn estimator class.
        estimator_args : Optional[Dict[str, Any]]
            Keyword arguments to pass to the estimator class constructor.
        """
        super().__init__()

        self.input_features = input_features
        self.predicted_features = predicted_features
        assert len(self.predicted_features) == 1, \
            "SklearnStrategy currently supports only one predicted feature"

        self._estimator_class = estimator
        self._estimator_args = estimator_args if estimator_args is not None else {}

        # Instantiate the estimator
        self._estimator = estimator(**self._estimator_args)

        # Validate the estimator has required methods
        assert hasattr(self._estimator, 'fit'), "Estimator must have a 'fit' method"
        assert hasattr(self._estimator, 'predict'), "Estimator must have a 'predict' method"

    def train(self,
              train_data: SeriesCollection,
              test_data: Optional[SeriesCollection] = None,
              num_procs: int = 1) -> None:
        """Train the scikit-learn estimator.

        Parameters
        ----------
        train_data : SeriesCollection
            The state series collection to use for training.
        test_data : SeriesCollection, optional
            The state series collection to use for validation (currently unused,
            but kept for interface compatibility).
        num_procs : int, optional
            Number of parallel processors to use when preprocessing data, by default 1.

        Raises
        ------
        AssertionError
            If no estimator has been set.
        """
        assert self._estimator is not None, "No estimator provided. Set estimator before training."

        # Preprocess the input features
        X_train = self.preprocess_features(train_data, self.input_features, num_procs)
        # Reshape from (n_series, n_timesteps, n_features) to (n_samples, n_features)
        X_train = X_train.reshape(-1, X_train.shape[-1])

        # Get the target values
        y_train = np.vstack([np.array(series) for series in self._get_targets(train_data, num_procs=num_procs)])

        # Train the estimator
        self._estimator.fit(X_train, y_train.ravel())

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        """Predict for a single state series.

        Parameters
        ----------
        state_series : np.ndarray
            Input state series of shape (n_timesteps, n_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_timesteps, n_outputs).
        """
        return self._predict_all([state_series])[0]

    def _predict_all(self, series_collection: np.ndarray, num_procs: int = 1) -> np.ndarray:
        """Predict for all series in the collection.

        Parameters
        ----------
        series_collection : np.ndarray
            Collection of state series of shape (n_series, n_timesteps, n_features).
        num_procs : int, optional
            Number of parallel processors (unused for sklearn, kept for compatibility).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_series, n_timesteps, n_outputs).
        """
        assert self.isTrained, "Model must be trained before prediction"
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"

        n_series, n_timesteps, n_features = series_collection.shape

        # Reshape to (n_samples, n_features)
        X = series_collection.reshape(-1, n_features)

        # Predict
        y = self._estimator.predict(X)

        # Ensure y is 2D
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Reshape back to (n_series, n_timesteps, n_outputs)
        n_outputs = y.shape[1]
        return y.reshape(n_series, n_timesteps, n_outputs)

    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        """Write the model to an HDF5 group.

        Parameters
        ----------
        h5_group : h5py.Group
            An opened, writeable HDF5 group or file handle.
        """
        # Write base class data
        super().write_model_to_hdf5(h5_group)

        # Save the estimator class and args
        if self._estimator_class is not None:
            h5_group.attrs['estimator_class_module'] = self._estimator_class.__module__
            h5_group.attrs['estimator_class_name'] = self._estimator_class.__name__

        # Save estimator args as JSON string
        if self._estimator_args:
            h5_group.attrs['estimator_args'] = json.dumps(self._estimator_args)

        # Save the trained estimator using pickle
        if self._estimator is not None:
            estimator_bytes = pickle.dumps(self._estimator)
            h5_group.create_dataset('estimator', data=np.void(estimator_bytes))

    def load_model_from_hdf5(self, h5_group: h5py.Group) -> None:
        """Load the model from an HDF5 group.

        Parameters
        ----------
        h5_group : h5py.Group
            An opened HDF5 group or file handle.
        """
        # Load base class data
        super().load_model(h5_group)

        # Load estimator class info
        if 'estimator_class_module' in h5_group.attrs:
            module_name = h5_group.attrs['estimator_class_module']
            class_name = h5_group.attrs['estimator_class_name']
            module = importlib.import_module(module_name)
            self._estimator_class = getattr(module, class_name)

        # Load estimator args
        if 'estimator_args' in h5_group.attrs:
            self._estimator_args = json.loads(h5_group.attrs['estimator_args'])
        else:
            self._estimator_args = {}

        # Load the trained estimator
        if 'estimator' in h5_group:
            estimator_bytes = bytes(h5_group['estimator'][()])
            self._estimator = pickle.loads(estimator_bytes)

    @classmethod
    def read_from_file(cls, file_name: str) -> SklearnStrategy:
        """Load a trained model from a file.

        Parameters
        ----------
        file_name : str
            The name of the file to load the model from.

        Returns
        -------
        SklearnStrategy
            The loaded strategy instance.
        """
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        with h5py.File(file_name, 'r') as h5_file:
            # Create an instance without setting features
            instance = cls.__new__(cls)
            # Call parent __init__ to set up base attributes
            PredictionStrategy.__init__(instance)
            instance._estimator = None
            # Load all data from file
            instance.load_model_from_hdf5(h5_file)
        return instance

    def __eq__(self, other: object) -> bool:
        """Check equality with another SklearnStrategy.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        bool
            True if the strategies are equal, False otherwise.
        """
        if not super().__eq__(other):
            return False

        assert isinstance(other, SklearnStrategy)

        # Compare estimator classes and estimator args
        if self._estimator_class != other._estimator_class:
            return False

        return self._estimator_args == other._estimator_args
