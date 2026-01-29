from __future__ import annotations
from typing import Dict, Optional, List, Type
import os
from math import isclose

import numpy as np
import h5py


from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
from ml_tools.model import register_prediction_strategy
from ml_tools.model.feature_processor import FeatureProcessor, NoProcessing

@register_prediction_strategy()
class ResidualCorrectionStrategy(PredictionStrategy):
    """ A concrete class for a residual correction prediction strategy

    This prediction strategy uses a reference prediction strategy to make an initial prediction,
    then trains a secondary model to predict the residual error of the reference model's predictions.
    The final prediction is the sum of the reference model's prediction and the residual correction.
    ResidualCorrectionStrategy can be daisy-chained to create a chain of residual correction models.

    Parameters
    ----------
    input_features : FeatureSpec
        Input feature/processor pairs (Dict) or feature name(s) (str/List[str], automatically mapped to NoProcessing).
        The preprocessing defined here is only applied to the residual correction model; the reference model's
        preprocessing is defined in its own PredictionStrategy.  The input features here must include all features
        required by the reference model.
    predicted_features : FeatureSpec
        Output feature/processor pairs (Dict) or feature name(s) (str/List[str], automatically mapped to NoProcessing).
        The postprocessing defined here is only applied to the residual correction model; the reference model's
        postprocessing is defined in its own PredictionStrategy.  The predicted features here must be a subset of those
        predicted by the reference model.
    reference_model : PredictionStrategy
        The reference prediction strategy to use for initial predictions.
    frozen_reference_model : Optional[bool]
        If True, the reference model is not trained when training this ResidualCorrectionStrategy.
        Default is False.

    Attributes
    ----------
    reference_model : PredictionStrategy
        The reference prediction strategy used for the initial predictions of this residual correction model.
    frozen_reference_model : bool
        If True, the reference model is not trained when training this ResidualCorrectionStrategy.
    residual_model : PredictionStrategy
        The prediction strategy used to predict the residuals of the reference model.
    """

    @property
    def reference_model(self) -> PredictionStrategy:
        return self._reference_model

    @reference_model.setter
    def reference_model(self, model: PredictionStrategy):
        assert all(feature in self.input_features.keys() for feature in model.input_features.keys()), \
            "All input features required by the reference model must be included in the this model's input features."
        assert all(feature in model.predicted_features.keys() for feature in self.predicted_features.keys()), \
            "All features predicted by this model must be a subset of those predicted by the reference model."
        self._reference_model = model

    @property
    def residual_model(self) -> PredictionStrategy:
        return self._residual_model

    @property
    def frozen_reference_model(self) -> bool:
        return self._frozen_reference_model

    @frozen_reference_model.setter
    def frozen_reference_model(self, value: bool):
        self._frozen_reference_model = value

    @property
    def isTrained(self) -> bool:
        return (self.reference_model.isTrained and self.residual_model.isTrained)

    def __init__(self,
                 input_features:         FeatureSpec,
                 predicted_features:     FeatureSpec,
                 reference_model:        PredictionStrategy,
                 residual_model:         PredictionStrategy,
                 frozen_reference_model: Optional[bool] = False):

        super().__init__()
        self.input_features         = input_features
        self.predicted_features     = predicted_features
        self.reference_model        = reference_model
        self.residual_model         = residual_model
        self.frozen_reference_model = frozen_reference_model


    def train(self, train_data: SeriesCollection, test_data: Optional[SeriesCollection] = None, num_procs: int = 1) -> None:

        if not self.frozen_reference_model:
            self.reference_model.train(train_data, test_data, num_procs)

        assert self.reference_model.isTrained, "Reference model must be trained successfully before training residual model."

        train_residuals = train_data.featurewise(op       = np.subtract,
                                                 other    = self.reference_model.predict(train_data, num_procs),
                                                 features = list(self.predicted_features.keys()))

        test_residuals = None
        if test_data is not None:
             test_residuals = test_data.featurewise(op       = np.subtract,
                                                    other    = self.reference_model.predict(test_data, num_procs),
                                                    features = list(self.predicted_features.keys()))

        self.residual_model.train(train_residuals, test_residuals, num_procs)


    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        reference_prediction = self.reference_model._predict_one(state_series)
        residual_prediction  = self.residual_model._predict_one(state_series)
        return reference_prediction + residual_prediction

    def _predict_all(self, series_collection: np.ndarray, num_procs: int = 1) -> np.ndarray:
        reference_prediction = self.reference_model._predict_all(series_collection, num_procs)
        residual_prediction  = self.residual_model._predict_all(series_collection, num_procs)
        return reference_prediction + residual_prediction


    def __eq__(self, other: object) -> bool:
        return (super().__eq__(other) and
                isinstance(other, ResidualCorrectionStrategy)        and
                self.reference_model        == other.reference_model and
                self.residual_model         == other.residual_model  and
                self.frozen_reference_model == other.frozen_reference_model)


    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        pass


    def load_model(self, h5_group: h5py.Group) -> None:
        pass


    @classmethod
    def read_from_file(cls: ResidualCorrectionStrategy, file_name: str) -> Type[ResidualCorrectionStrategy]:
        pass


    @classmethod
    def from_dict(cls,
                  params:             Dict,
                  input_features:     FeatureSpec,
                  predicted_features: FeatureSpec,
                  biasing_model:      Optional[PredictionStrategy] = None) -> ResidualCorrectionStrategy:
        pass

    def to_dict(self) -> dict:
        pass
