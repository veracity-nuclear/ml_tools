from __future__ import annotations
from typing import Dict, Optional, Type
import os

import numpy as np
import h5py


from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
from ml_tools.model.feature_processor import FeatureProcessor
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
    reference_model : Optional[PredictionStrategy]
        Optional reference prediction strategy to use for initial predictions.
        This can be set after initialization via the reference_model setter.
    reference_model_frozen : Optional[bool]
        If True, the reference model is not trained when training this ResidualCorrectionStrategy.
        Default is False.

    Attributes
    ----------
    residual_model : Optional[PredictionStrategy]
        The prediction strategy used to predict the residuals of the reference model.
    reference_model : Optional[PredictionStrategy]
        The reference prediction strategy used for the initial predictions of this residual correction model.
    reference_model_frozen : bool
        If True, the reference model is not trained when training this ResidualCorrectionStrategy.
    """

    @property
    def residual_model(self) -> Optional[PredictionStrategy]:
        return self._residual_model if hasattr(self, "_residual_model") else None

    @residual_model.setter
    def residual_model(self, model: Optional[PredictionStrategy]) -> None:
        self._residual_model = model
        self._check_predicted_features_are_consistent()
        if model is not None:
            PredictionStrategy.predicted_features.fset(self, model.predicted_features)

    @property
    def reference_model(self) -> Optional[PredictionStrategy]:
        return self._reference_model if hasattr(self, "_reference_model") else None

    @reference_model.setter
    def reference_model(self, model: Optional[PredictionStrategy]) -> None:
        self._reference_model = model
        self._check_predicted_features_are_consistent()


    def _check_predicted_features_are_consistent(self) -> None:
        if self.residual_model is None:
            return
        if self.reference_model is None:
            return
        res_outputs = self.residual_model.predicted_features
        ref_outputs = self.reference_model.predicted_features
        for name, processor in res_outputs.items():
            assert name in ref_outputs and ref_outputs[name] == processor, \
                f"Predicted feature '{name}' processor mismatch between residual and reference models."
        return


    @property
    def reference_model_frozen(self) -> bool:
        return self._reference_model_frozen

    @reference_model_frozen.setter
    def reference_model_frozen(self, frozen: bool) -> None:
        self._reference_model_frozen = frozen

    @property
    def input_features(self) -> Dict[str, FeatureProcessor]:
        res_inputs = self.residual_model.input_features if self.residual_model else {}
        ref_inputs = self.reference_model.input_features if self.reference_model else {}
        combined = dict(res_inputs)
        for name, processor in ref_inputs.items():
            if name in combined and combined[name] != processor:
                raise AssertionError(
                    f"Input processor mismatch for '{name}' between residual and reference models; "
                    "access via residual_model.input_features or reference_model.input_features."
                )
            combined.setdefault(name, processor)
        return combined

    @input_features.setter
    def input_features(self, features: FeatureSpec) -> None:
        raise AttributeError("input_features is a read-only property for ResidualCorrectionStrategy")

    @property
    def predicted_features(self) -> Dict[str, FeatureProcessor]:
        return self._predicted_features

    @predicted_features.setter
    def predicted_features(self, features: FeatureSpec) -> None:
        raise AttributeError("predicted_features is a read-only property for ResidualCorrectionStrategy")

    @property
    def isTrained(self) -> bool:
        if self.reference_model is not None and self.residual_model is not None:
            if self.reference_model.isTrained and self.residual_model.isTrained:
                return True
        return False


    def __init__(self,
                 residual_model:         Optional[PredictionStrategy] = None,
                 reference_model:        Optional[PredictionStrategy] = None,
                 reference_model_frozen: Optional[bool] = False):

        super().__init__()

        self.residual_model         = residual_model
        self.reference_model        = reference_model
        self.reference_model_frozen = reference_model_frozen


    def train(self, train_data: SeriesCollection, test_data: Optional[SeriesCollection] = None, num_procs: int = 1) -> None:

        assert self.residual_model is not None, "residual_model must be set before training"
        assert self.reference_model is not None, "reference_model must be set before training"

        if not self.reference_model_frozen:
            self.reference_model.train(train_data, test_data, num_procs)

        assert self.reference_model.isTrained, "Reference model must be trained successfully before training residual model."

        predicted_features = self.residual_model.predicted_feature_names
        train_reference    = self.reference_model.predict(train_data, num_procs)
        train_residuals    = train_data.featurewise(op        = np.subtract,
                                                    other     = train_reference,
                                                    features  = predicted_features,
                                                    num_procs = num_procs)

        test_residuals = None
        if test_data is not None:
            test_reference = self.reference_model.predict(test_data, num_procs)
            test_residuals = test_data.featurewise(op        = np.subtract,
                                                   other     = test_reference,
                                                   features  = predicted_features,
                                                   num_procs = num_procs)

        self.residual_model.train(train_residuals, test_residuals, num_procs)


    def predict(self, series_collection: SeriesCollection, num_procs: int = 1) -> SeriesCollection:

        assert self.isTrained, "Model must be trained before prediction"
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"

        reference_prediction = self.reference_model.predict(series_collection, num_procs=num_procs)
        residual_prediction  = self.residual_model.predict(series_collection, num_procs=num_procs)
        return reference_prediction.featurewise(op        = np.add,
                                                other     = residual_prediction,
                                                features  = self.residual_model.predicted_feature_names,
                                                num_procs = num_procs)

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        """Note: This method expects already-preprocessed inputs and therefore
        requires identical input feature definitions for reference and residual
        models. If input features differ, preprocess and predict with each model
        separately before combining outputs."""
        assert self.isTrained, "Both reference_model and residual_model must be trained before prediction."

        assert self.residual_model.input_features == self.reference_model.input_features, \
            "Input features of residual_model and reference_model must match for this method."

        reference_preds = self.reference_model._predict_one(state_series) # pylint: disable=protected-access
        residual_preds  = self.residual_model._predict_one(state_series) # pylint: disable=protected-access
        final_preds     = reference_preds + residual_preds
        return final_preds

    def _predict_all(self, series_collection: np.ndarray, num_procs: int = 1) -> np.ndarray:
        """Note: This method expects already-preprocessed inputs and therefore
        requires identical input feature definitions for reference and residual
        models. If input features differ, preprocess and predict with each model
        separately before combining outputs."""
        assert self.isTrained, "Both reference_model and residual_model must be trained before prediction."

        assert self.residual_model.input_features == self.reference_model.input_features, \
            "Input features of residual_model and reference_model must match for this method."

        reference_preds = self.reference_model._predict_all(series_collection, num_procs) # pylint: disable=protected-access
        residual_preds  = self.residual_model._predict_all(series_collection, num_procs) # pylint: disable=protected-access
        final_preds     = reference_preds + residual_preds
        return final_preds


    def __eq__(self, other: object) -> bool:
        return (super().__eq__(other) and
                isinstance(other, ResidualCorrectionStrategy)        and
                self.reference_model        == other.reference_model and
                self.residual_model         == other.residual_model  and
                self.reference_model_frozen == other.reference_model_frozen)


    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        h5_group.create_dataset('reference_model_frozen', data=int(self.reference_model_frozen))

        assert self.residual_model is not None, "residual_model must be set before saving"
        residual_group = h5_group.create_group('residual_model')
        residual_group.attrs['strategy_type'] = type(self.residual_model).__name__
        self.residual_model.write_model_to_hdf5(residual_group)

        if self.reference_model is not None:
            reference_group = h5_group.create_group('reference_model')
            reference_group.attrs['strategy_type'] = type(self.reference_model).__name__
            self.reference_model.write_model_to_hdf5(reference_group)

    def load_model(self, h5_group: h5py.Group) -> None:
        self.reference_model_frozen = bool(h5_group['reference_model_frozen'][()])

        self.residual_model = self._load_strategy_from_group(h5_group['residual_model'])

        if 'reference_model' in h5_group:
            self.reference_model = self._load_strategy_from_group(h5_group['reference_model'])
        else:
            self.reference_model = None

    @staticmethod
    def _load_strategy_from_group(group: h5py.Group) -> PredictionStrategy:
        strategy_type = group.attrs.get('strategy_type')
        if isinstance(strategy_type, bytes):
            strategy_type = strategy_type.decode('utf-8')
        if isinstance(strategy_type, np.ndarray):
            strategy_type = strategy_type[()].decode('utf-8')
        if strategy_type is None and 'strategy_type' in group:
            strategy_type = group['strategy_type'][()].decode('utf-8')

        cls = _PREDICTION_STRATEGY_REGISTRY.get(strategy_type, None)
        if cls is None:
            raise KeyError(f"Unknown PredictionStrategy type in HDF5: {strategy_type}")

        instance = cls.__new__(cls)
        PredictionStrategy.__init__(instance)

        instance.load_model(group)
        return instance

    @classmethod
    def read_from_file(cls: ResidualCorrectionStrategy, file_name: str) -> Type[ResidualCorrectionStrategy]:
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        assert os.path.exists(file_name), f"file name = {file_name}"

        instance = cls.__new__(cls)
        PredictionStrategy.__init__(instance)
        instance.load_model(h5py.File(file_name, "r"))
        return instance

    @classmethod
    def from_dict(cls,
                  params:             Dict,
                  input_features:     FeatureSpec,
                  predicted_features: FeatureSpec) -> ResidualCorrectionStrategy:

        residual_strategy_type = params.get("residual_strategy_type")
        residual_params        = params.get("residual_strategy_params", {})
        reference_model_frozen = params.get("reference_model_frozen", False)
        assert residual_strategy_type is not None, "residual_strategy_type is required in params"

        residual_model = build_prediction_strategy(strategy_type      = residual_strategy_type,
                                                   params             = residual_params,
                                                   input_features     = input_features,
                                                   predicted_features = predicted_features)

        reference_model = None
        reference_strategy_type = params.get("reference_strategy_type", None)
        if reference_strategy_type is not None:
            ref_input_features     = params.get("reference_input_features")
            ref_predicted_features = params.get("reference_predicted_features")
            reference_model = build_prediction_strategy(
                strategy_type      = reference_strategy_type,
                params             = params.get("reference_strategy_params", {}),
                input_features     = cls.features_from_dict(ref_input_features),
                predicted_features = cls.features_from_dict(ref_predicted_features),
            )

        return cls(residual_model, reference_model, reference_model_frozen)


    def to_dict(self) -> dict:
        assert self.residual_model is not None, "residual_model must be set before saving to dict"

        params = {"residual_strategy_type": type(self.residual_model).__name__,
                  "residual_strategy_params": self.residual_model.to_dict(),
                  "reference_model_frozen": self.reference_model_frozen}

        if self.reference_model is not None:
            params["reference_strategy_type"] = type(self.reference_model).__name__
            params["reference_strategy_params"] = self.reference_model.to_dict()
            params["reference_input_features"] = self.features_to_dict(self.reference_model.input_features)
            params["reference_predicted_features"] = self.features_to_dict(self.reference_model.predicted_features)

        return params
