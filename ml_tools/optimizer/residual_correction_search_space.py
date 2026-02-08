from __future__ import annotations
from typing import Optional

from ml_tools.model.prediction_strategy import FeatureSpec
from ml_tools.optimizer.search_space import (
    SearchSpace,
    StructDimension,
)


class ResidualCorrectionSearchSpace(SearchSpace):
    """ResidualCorrectionStrategy hyperparameter search space.

    Parameters
    ----------
    dimensions : ResidualCorrectionSearchSpace.Dimension
        The root hyperparameter search space to explore.
    input_features : Optional[FeatureSpec]
        Default input feature/processor pairs used when nested model strategy
        dicts omit ``input_features``. If ``None``, both nested model
        dimensions must include ``input_features``.
    predicted_features : Optional[FeatureSpec]
        Default output feature/processor pairs used when nested model strategy
        dicts omit ``predicted_features``. If ``None``, both nested model
        dimensions must include ``predicted_features``.
    """

    class Dimension(StructDimension):
        """Residual correction chain hyperparameter domains.

        Parameters
        ----------
        residual_model : StructDimension
            Struct dimension that samples the nested residual-model
            ``strategy_dict`` with keys like ``strategy_type`` and ``params``
            (and optionally ``input_features`` / ``predicted_features``).
        reference_model : StructDimension
            Struct dimension that samples the nested reference-model
            ``strategy_dict`` with keys like ``strategy_type`` and ``params``
            (and optionally ``input_features`` / ``predicted_features``).
        """

        @property
        def residual_model(self) -> StructDimension:
            return self.fields["residual_model"]

        @residual_model.setter
        def residual_model(self, value: StructDimension) -> None:
            assert isinstance(value, StructDimension), \
                f"residual_model must be a StructDimension, got {type(value)}"
            self.fields["residual_model"] = value

        @property
        def reference_model(self) -> StructDimension:
            return self.fields["reference_model"]

        @reference_model.setter
        def reference_model(self, value: StructDimension) -> None:
            assert isinstance(value, StructDimension), \
                f"reference_model must be a StructDimension, got {type(value)}"
            self.fields["reference_model"] = value

        def __init__(self,
                     residual_model: StructDimension,
                     reference_model: StructDimension) -> None:
            self.fields = {}
            self.residual_model = residual_model
            self.reference_model = reference_model
            super().__init__(self.fields)

    def __init__(self,
                 dimensions: StructDimension,
                 input_features: Optional[FeatureSpec],
                 predicted_features: Optional[FeatureSpec]) -> None:
        assert isinstance(dimensions, ResidualCorrectionSearchSpace.Dimension), (
            f"dimensions must be a ResidualCorrectionSearchSpace.Dimension, got {type(dimensions)}"
        )
        if input_features is None:
            assert "input_features" in dimensions.residual_model.fields, \
                "Residual model dimension must include 'input_features' when SearchSpace.input_features is None"
            assert "input_features" in dimensions.reference_model.fields, \
                "Reference model dimension must include 'input_features' when SearchSpace.input_features is None"
        if predicted_features is None:
            assert "predicted_features" in dimensions.residual_model.fields, \
                "Residual model dimension must include 'predicted_features' when SearchSpace.predicted_features is None"
            assert "predicted_features" in dimensions.reference_model.fields, \
                "Reference model dimension must include 'predicted_features' when SearchSpace.predicted_features is None"
        super().__init__(prediction_strategy_type="ResidualCorrectionStrategy",
                         dimensions=dimensions,
                         input_features=input_features,
                         predicted_features=predicted_features)
