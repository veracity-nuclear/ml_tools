from __future__ import annotations
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy


class PODBasedStrategy(PredictionStrategy):
    """ An abstract class for POD-based prediction strategies. Not meant to be instantiated directly.

    POD-based strategies predict their target values as a linear combination of POD moments,
    i.e. ``target = moments @ weights``.  This class exposes the two pieces of that product
    separately for workflows that want to work with the predicted weights directly rather
    than the reconstructed predicted features.

    Note that the moments live in the processed predicted-feature space (i.e. before
    postprocessing by the predicted feature processors), with multiple predicted features
    concatenated in the order of ``predicted_feature_names``.

    Attributes
    ----------
    moments : np.ndarray
        The POD moments of shape (output_dim, num_moments)
    """

    @property
    @abstractmethod
    def moments(self) -> np.ndarray:
        """ The POD moments of shape (output_dim, num_moments) """

    def predict_weights(self, series_collection: SeriesCollection, num_procs: int = 1) -> np.ndarray:
        """ Predict the POD moment weights for each state in each series

        Parameters
        ----------
        series_collection : SeriesCollection
            The input state series collection which to predict weights for
        num_procs : int, optional
            The number of processes to use for parallel processing, by default 1

        Returns
        -------
        np.ndarray
            Array of shape (num_series, max_timesteps, num_moments) containing the predicted
            weights, such that ``predict_processed_inputs(...) == weights @ moments.T``.
            No NaN padding adjustments are made.
        """
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        assert self.isTrained

        processed_inputs = PredictionStrategy.preprocess_features(series_collection,
                                                                  self.input_features,
                                                                  num_procs=num_procs)
        return np.asarray(self._predict_weights_all(processed_inputs, num_procs=num_procs))

    def _predict_weights_all(self, series_collection: np.ndarray, num_procs: int = 1) -> np.ndarray:
        """ The method that predicts the POD moment weights for the given state series collection

        Parameters
        ----------
        series_collection : np.ndarray
            The preprocessed and padded state series collection in np.array format for which
            to predict the weights
        num_procs : int
            The number of parallel processors to use when predicting the data

        Returns
        -------
        np.ndarray
            The predicted weights for each state in each series
        """
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        if num_procs <= 1:
            return np.asarray([self._predict_weights_one(series) for series in series_collection])

        series_list = list(series_collection)
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            return np.asarray(list(executor.map(self._predict_weights_one, series_list)))

    @abstractmethod
    def _predict_weights_one(self, state_series: np.ndarray) -> np.ndarray:
        """ The method that predicts the POD moment weights for the given state series

        Parameters
        ----------
        state_series : np.ndarray
            The preprocessed input state series for which to predict the weights

        Returns
        -------
        np.ndarray
            The predicted weights of shape (num_timesteps, num_moments)
        """
