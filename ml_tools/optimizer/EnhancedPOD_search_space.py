from __future__ import annotations
from typing import Optional
from ml_tools.model.prediction_strategy import FeatureSpec
from ml_tools.optimizer.search_space import (
    SearchSpace,
    StructDimension,
    IntDimension,
    CategoricalDimension,
)


class EnhancedPODSearchSpace(SearchSpace):
    """EnhancedPODStrategy hyperparameter search space.

    Parameters
    ----------
    dimensions : EnhancedPODSearchSpace.Dimension
        The root hyperparameter search space to explore.
    input_features : Optional[FeatureSpec]
        Default input features. Must be provided for EnhancedPOD search spaces.
    predicted_features : Optional[FeatureSpec]
        Default predicted features. Must be provided for EnhancedPOD search spaces.
    """

    class Dimension(StructDimension):
        """EnhancedPODStrategy hyperparameter domains (to be sampled; not final values).

        Parameters
        ----------
        theta_model_type : CategoricalDimension, optional
            Domain for the theta model type ('GBM', 'NN', or 'sklearn')
        max_svd_size : CategoricalDimension, optional
            Domain for the maximum number of training states used in the SVD
        num_moments : IntDimension, optional
            Domain for the number of POD moments
        constraints : CategoricalDimension, optional
            Domain of constraint lists, each a list of (gamma, W) pairs
        theta_model_settings : StructDimension, optional
            Domains for settings passed to the theta model constructor(s), e.g. a
            ``GBMSearchSpace.Dimension`` for GBM theta models.  If None, defaults are used.

        Attributes
        ----------
        theta_model_type : CategoricalDimension
            Domain for the theta model type
        max_svd_size : CategoricalDimension
            Domain for the maximum number of training states used in the SVD
        num_moments : IntDimension
            Domain for the number of POD moments
        constraints : CategoricalDimension
            Domain of constraint lists
        theta_model_settings : Optional[StructDimension]
            Domains for settings passed to the theta model constructor(s)
        """
        @property
        def constraints(self) -> CategoricalDimension:
            return self.fields["constraints"]

        @constraints.setter
        def constraints(self, value: CategoricalDimension) -> None:
            self.fields["constraints"] = value

        @property
        def num_moments(self) -> IntDimension:
            return self.fields["num_moments"]

        @num_moments.setter
        def num_moments(self, value: IntDimension) -> None:
            self.fields["num_moments"] = value

        @property
        def max_svd_size(self) -> CategoricalDimension:
            return self.fields["max_svd_size"]

        @max_svd_size.setter
        def max_svd_size(self, value: CategoricalDimension) -> None:
            self.fields["max_svd_size"] = value

        @property
        def theta_model_type(self) -> CategoricalDimension:
            return self.fields["theta_model_type"]

        @theta_model_type.setter
        def theta_model_type(self, value: CategoricalDimension) -> None:
            self.fields["theta_model_type"] = value

        @property
        def theta_model_settings(self) -> Optional[StructDimension]:
            return self.fields.get("theta_model_settings")

        @theta_model_settings.setter
        def theta_model_settings(self, value: Optional[StructDimension]) -> None:
            if value is None:
                self.fields.pop("theta_model_settings", None)
            else:
                self.fields["theta_model_settings"] = value

        def __init__(self,
                     theta_model_type: CategoricalDimension = CategoricalDimension(["GBM"]),
                     max_svd_size:   CategoricalDimension = CategoricalDimension([None]),
                     num_moments:    IntDimension = IntDimension(6, 6),
                     constraints:    CategoricalDimension=CategoricalDimension([[]]),
                     theta_model_settings:   Optional[StructDimension] = None) -> None:

            self.fields = {}
            self.theta_model_type = theta_model_type
            self.max_svd_size = max_svd_size
            self.num_moments  = num_moments
            self.constraints = constraints
            self.theta_model_settings = theta_model_settings

            super().__init__(self.fields)

    def __init__(self,
                 dimensions: StructDimension,
                 input_features: Optional[FeatureSpec] = None,
                 predicted_features: Optional[FeatureSpec] = None) -> None:
        assert isinstance(dimensions, EnhancedPODSearchSpace.Dimension), (
            f"dimensions must be a EnhancedPODSearchSpace.Dimension, got {type(dimensions)}"
        )
        if input_features is None:
            assert "input_features" in dimensions.fields, \
                "EnhancedPODSearchSpace requires 'input_features' either as defaults or in dimensions"
        if predicted_features is None:
            assert "predicted_features" in dimensions.fields, \
                "EnhancedPODSearchSpace requires 'predicted_features' either as defaults or in dimensions"
        super().__init__(prediction_strategy_type="EnhancedPODStrategy",
                         dimensions=dimensions,
                         input_features=input_features,
                         predicted_features=predicted_features)
