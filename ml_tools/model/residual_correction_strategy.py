from __future__ import annotations
from typing import Dict, Optional, Type, Any
import os

import numpy as np
import h5py


from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
from ml_tools.model import register_prediction_strategy, _PREDICTION_STRATEGY_REGISTRY, build_prediction_strategy

@register_prediction_strategy()
class ResidualCorrectionStrategy(PredictionStrategy):
    """ A concrete class for a residual correction prediction strategy

    This prediction strategy uses a reference prediction strategy to make an initial prediction,
    then trains a secondary model to predict the residual error of the reference model's predictions.
    The final prediction is the sum of the reference model's prediction and the residual correction.
    ResidualCorrectionStrategy can be daisy-chained to create a chain of residual correction models.

    Parameters
    ----------
    residual_model : Optional[PredictionStrategy]
        The prediction strategy used to predict residual corrections. This model
        dictates the input and predicted features of this strategy. If not provided,
        use ``residual_strategy_type`` + ``residual_params`` + ``input_features`` +
        ``predicted_features`` to build it.
    residual_strategy_type : Optional[str]
        Strategy type name for building the residual model (e.g., "NNStrategy").
    residual_params : Optional[Dict]
        Parameters for building the residual model (passed to ``build_prediction_strategy``).
    input_features : Optional[FeatureSpec]
        Feature spec used when building the residual model. Must match the residual model
        if one is provided directly.
    predicted_features : Optional[FeatureSpec]
        Feature spec used when building the residual model. Must match the residual model
        if one is provided directly.
    reference_model : PredictionStrategy
        The reference prediction strategy to use for initial predictions.
    reference_model_frozen : Optional[bool]
        If True, the reference model is not trained when training this ResidualCorrectionStrategy.
        Default is False.

    Attributes
    ----------
    residual_model : PredictionStrategy
        The prediction strategy used to predict the residuals of the reference model.
    reference_model : PredictionStrategy
        The reference prediction strategy used for the initial predictions of this residual correction model.
    reference_model_frozen : bool
        If True, the reference model is not trained when training this ResidualCorrectionStrategy.
    """

    @property
    def residual_model(self) -> PredictionStrategy:
        return self._residual_model

    @property
    def reference_model(self) -> PredictionStrategy:
        return self._reference_model

    @property
    def reference_model_frozen(self) -> bool:
        return self._reference_model_frozen

    @reference_model_frozen.setter
    def reference_model_frozen(self, value: bool):
        self._reference_model_frozen = value

    @property
    def isTrained(self) -> bool:
        return (self.reference_model.isTrained and self.residual_model.isTrained)

    def __init__(self,
                 reference_model:        PredictionStrategy,
                 residual_model:         Optional[PredictionStrategy] = None,
                 reference_model_frozen: Optional[bool] = False,
                 residual_strategy_type: Optional[str] = None,
                 residual_params:        Optional[Dict] = None,
                 input_features:         Optional[FeatureSpec] = None,
                 predicted_features:     Optional[FeatureSpec] = None):

        super().__init__()

        self._reference_model = None
        self._residual_model = None
        self._reference_model_frozen = False

        if residual_model is None:
            assert residual_strategy_type is not None, \
                "residual_strategy_type is required when residual_model is not provided"
            assert input_features is not None and predicted_features is not None, \
                "input_features and predicted_features are required to build residual_model"
            residual_model = build_prediction_strategy(strategy_type      = residual_strategy_type,
                                                       params             = residual_params or {},
                                                       input_features     = input_features,
                                                       predicted_features = predicted_features,
                                                       biasing_model      = None)
        else:
            assert input_features is None and predicted_features is None, \
                "input_features/predicted_features are derived from residual_model and should not be provided"
            assert residual_strategy_type is None and residual_params is None, \
                "residual_strategy_type/params should not be provided when residual_model is passed"

        self._set_residual_model(residual_model)
        self._set_reference_model(reference_model)
        self.reference_model_frozen = reference_model_frozen


    def _set_residual_model(self, model: PredictionStrategy) -> None:
        ordered_pred            = {name: model.predicted_features[name] for name in model.predicted_feature_names}
        self.input_features     = dict(model.input_features)
        self.predicted_features = ordered_pred
        self._residual_model    = model


    def _set_reference_model(self, model: PredictionStrategy) -> None:
        assert all(feature in self.input_features for feature in model.input_features), \
            "All input features required by the reference model must be included in this model's input features."
        for name, processor in model.input_features.items():
            assert self.input_features[name] == processor, \
                f"Input processor mismatch for '{name}'"
        assert all(feature in model.predicted_features for feature in self.predicted_features), \
            "All features predicted by this model must be a subset of those predicted by the reference model."
        self._reference_model = model




    def train(self, train_data: SeriesCollection, test_data: Optional[SeriesCollection] = None, num_procs: int = 1) -> None:

        if not self.reference_model_frozen:
            self.reference_model.train(train_data, test_data, num_procs)

        assert self.reference_model.isTrained, "Reference model must be trained successfully before training residual model."

        pred_names            = self.predicted_feature_names
        reference_predictions = self.reference_model.predict(train_data, num_procs)
        train_residuals       = train_data.featurewise(op                 = np.subtract,
                                                       other              = reference_predictions,
                                                       features           = pred_names,
                                                       num_procs          = num_procs)

        test_residuals = None
        if test_data is not None:
            test_reference = self.reference_model.predict(test_data, num_procs)
            test_residuals = test_data.featurewise(op                 = np.subtract,
                                                   other              = test_reference,
                                                   features           = pred_names,
                                                   num_procs          = num_procs)

        self.residual_model.train(train_residuals, test_residuals, num_procs)


    def predict_processed_inputs(self, processed_inputs: np.ndarray, num_procs: int = 1) -> np.ndarray:
        raise NotImplementedError("ResidualCorrectionStrategy does not support predict_processed_inputs")


    def predict(self, series_collection: SeriesCollection, num_procs: int = 1) -> SeriesCollection:
        """Predict using reference and residual models, then sum their outputs."""
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        reference_prediction = self.reference_model.predict(series_collection, num_procs=num_procs)
        residual_prediction = self.residual_model.predict(series_collection, num_procs=num_procs)
        return reference_prediction.featurewise(op       = np.add,
                                                other    = residual_prediction,
                                                features = self.predicted_feature_names,
                                                num_procs = num_procs)

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ResidualCorrectionStrategy does not support _predict_one")

    def _predict_all(self, series_collection: np.ndarray, num_procs: int = 1) -> np.ndarray:
        raise NotImplementedError("ResidualCorrectionStrategy does not support _predict_all")


    def __eq__(self, other: object) -> bool:
        return (super().__eq__(other) and
                isinstance(other, ResidualCorrectionStrategy)        and
                self.reference_model        == other.reference_model and
                self.residual_model         == other.residual_model  and
                self.reference_model_frozen == other.reference_model_frozen)


    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        super().write_model_to_hdf5(h5_group)
        h5_group.create_dataset('reference_model_frozen', data=int(self.reference_model_frozen))

        reference_group = h5_group.create_group('reference_model')
        reference_group.attrs['strategy_type'] = type(self.reference_model).__name__
        self.reference_model.write_model_to_hdf5(reference_group)

        residual_group = h5_group.create_group('residual_model')
        residual_group.attrs['strategy_type'] = type(self.residual_model).__name__
        self.residual_model.write_model_to_hdf5(residual_group)


    def load_model(self, h5_group: h5py.Group) -> None:
        super().load_model(h5_group)
        self._set_residual_model(self._load_strategy_from_group(h5_group['residual_model']))
        self._set_reference_model(self._load_strategy_from_group(h5_group['reference_model']))
        self.reference_model_frozen = bool(h5_group['reference_model_frozen'][()])

    @staticmethod
    def _load_strategy_from_group(group: h5py.Group) -> PredictionStrategy:
        strategy_type = group.attrs.get('strategy_type')
        if isinstance(strategy_type, bytes):
            strategy_type = strategy_type.decode('utf-8')
        if isinstance(strategy_type, np.ndarray):
            strategy_type = strategy_type[()].decode('utf-8')
        if strategy_type is None:
            strategy_type = group['strategy_type'][()].decode('utf-8')

        if strategy_type not in _PREDICTION_STRATEGY_REGISTRY:
            raise KeyError(f"Unknown PredictionStrategy type: {strategy_type}")

        cls = _PREDICTION_STRATEGY_REGISTRY[strategy_type]
        instance = cls.__new__(cls)
        PredictionStrategy.__init__(instance)
        if hasattr(instance, 'load_model') and callable(getattr(instance, 'load_model')):
            instance.load_model(group)
        elif hasattr(instance, 'load_model_from_hdf5') and callable(getattr(instance, 'load_model_from_hdf5')):
            instance.load_model_from_hdf5(group)
        else:
            raise NotImplementedError(f"{strategy_type} does not support HDF5 loading")
        return instance


    @classmethod
    def read_from_file(cls: ResidualCorrectionStrategy, file_name: str) -> Type[ResidualCorrectionStrategy]:
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        assert os.path.exists(file_name), f"file name = {file_name}"

        instance = cls.__new__(cls)
        PredictionStrategy.__init__(instance)
        instance._reference_model        = None
        instance._residual_model         = None
        instance._reference_model_frozen = False
        instance.load_model(h5py.File(file_name, "r"))
        return instance


    @classmethod
    def from_dict(cls,
                  params:             Dict,
                  input_features:     FeatureSpec,
                  predicted_features: FeatureSpec,
                  biasing_model:      Optional[PredictionStrategy] = None) -> ResidualCorrectionStrategy:

        assert "reference_strategy_type"   in params, "reference_strategy_type is required in params"
        assert "residual_strategy_type"    in params, "residual_strategy_type is required in params"
        assert "reference_strategy_params" in params, "reference_strategy_params is required in params"
        assert "residual_strategy_params"  in params, "residual_strategy_params is required in params"

        reference_model_frozen  = params.get("reference_model_frozen", False)
        reference_strategy_type = params.get("reference_strategy_type")
        reference_params        = params.get("reference_strategy_params")
        reference_model         = build_prediction_strategy(strategy_type      = reference_strategy_type,
                                                            params             = reference_params,
                                                            input_features     = input_features,
                                                            predicted_features = predicted_features)

        residual_strategy_type = params.get("residual_strategy_type")
        residual_params        = params.get("residual_strategy_params")

        instance = cls(residual_model         = None,
                       reference_model        = reference_model,
                       reference_model_frozen = reference_model_frozen,
                       residual_strategy_type = residual_strategy_type,
                       residual_params        = residual_params,
                       input_features         = input_features,
                       predicted_features     = predicted_features)

        if biasing_model is not None:
            instance.biasing_model = biasing_model
        return instance

    def to_dict(self) -> dict:
        return {"reference_strategy_type": type(self.reference_model).__name__,
                "residual_strategy_type": type(self.residual_model).__name__,
                "reference_strategy_params": self.reference_model.to_dict(),
                "residual_strategy_params": self.residual_model.to_dict(),
                "reference_model_frozen": self.reference_model_frozen}
