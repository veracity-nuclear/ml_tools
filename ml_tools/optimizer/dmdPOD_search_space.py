from __future__ import annotations
from typing import Optional

from ml_tools.model.prediction_strategy import FeatureSpec
from ml_tools.optimizer.search_space import (
    SearchSpace,
    StructDimension,
    IntDimension,
    FloatDimension,
    CategoricalDimension,
)


class DMDPODSearchSpace(SearchSpace):
    """DMDPODStrategy hyperparameter search space.

    Parameters
    ----------
    dimensions : DMDPODSearchSpace.Dimension
        The root hyperparameter search space to explore.
    input_features : Optional[FeatureSpec]
        Default input features. Must be provided for DMD-POD search spaces.
    predicted_features : Optional[FeatureSpec]
        Default predicted features. Must be provided for DMD-POD search spaces.

    Notes
    -----
    - For parameter semantics and valid ranges, see
      ``ml_tools.model.dmdPOD_strategy.DMDPODStrategy``.
    - Search-space scoring uses :meth:`DMDPODStrategy.predict`, which infers initial weights
      from inputs, so ``initial_theta_model_type`` should not be None.
    """

    class Dimension(StructDimension):
        """DMDPODStrategy hyperparameter domains (to be sampled; not final values).

        Parameters
        ----------
        num_moments : IntDimension, optional
            Domain for the number of POD moments
        max_svd_size : CategoricalDimension, optional
            Domain for the maximum number of training states used in the SVD
        theta_scaling : CategoricalDimension, optional
            Domain for the POD weight normalization used in regression
        regularization : FloatDimension, optional
            Domain for the ridge penalty on the step operators
        fit_intercept : CategoricalDimension, optional
            Domain for whether to fit the constant term of the dynamics
        initial_theta_model_type : CategoricalDimension, optional
            Domain for the initial theta model type ('GBM', 'NN', or 'sklearn')
        initial_theta_model_settings : StructDimension, optional
            Domains for settings passed to the initial theta model constructor(s), e.g. a
            ``GBMSearchSpace.Dimension`` for GBM theta models.  If None, defaults are used.

        Attributes
        ----------
        num_moments : IntDimension
            Domain for the number of POD moments
        max_svd_size : CategoricalDimension
            Domain for the maximum number of training states used in the SVD
        theta_scaling : CategoricalDimension
            Domain for the POD weight normalization used in regression
        regularization : FloatDimension
            Domain for the ridge penalty on the step operators
        fit_intercept : CategoricalDimension
            Domain for whether to fit the constant term of the dynamics
        initial_theta_model_type : CategoricalDimension
            Domain for the initial theta model type
        initial_theta_model_settings : Optional[StructDimension]
            Domains for settings passed to the initial theta model constructor(s)
        """

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
        def theta_scaling(self) -> CategoricalDimension:
            return self.fields["theta_scaling"]

        @theta_scaling.setter
        def theta_scaling(self, value: CategoricalDimension) -> None:
            self.fields["theta_scaling"] = value

        @property
        def regularization(self) -> FloatDimension:
            return self.fields["regularization"]

        @regularization.setter
        def regularization(self, value: FloatDimension) -> None:
            self.fields["regularization"] = value

        @property
        def fit_intercept(self) -> CategoricalDimension:
            return self.fields["fit_intercept"]

        @fit_intercept.setter
        def fit_intercept(self, value: CategoricalDimension) -> None:
            self.fields["fit_intercept"] = value

        @property
        def initial_theta_model_type(self) -> CategoricalDimension:
            return self.fields["initial_theta_model_type"]

        @initial_theta_model_type.setter
        def initial_theta_model_type(self, value: CategoricalDimension) -> None:
            self.fields["initial_theta_model_type"] = value

        @property
        def initial_theta_model_settings(self) -> Optional[StructDimension]:
            return self.fields.get("initial_theta_model_settings")

        @initial_theta_model_settings.setter
        def initial_theta_model_settings(self, value: Optional[StructDimension]) -> None:
            if value is None:
                self.fields.pop("initial_theta_model_settings", None)
            else:
                self.fields["initial_theta_model_settings"] = value

        def __init__(self,
                     num_moments:                  IntDimension = IntDimension(6, 6),
                     max_svd_size:                 CategoricalDimension = CategoricalDimension([None]),
                     theta_scaling:                CategoricalDimension = CategoricalDimension(["sqrt_singular_values"]),
                     regularization:               FloatDimension = FloatDimension(0.0, 0.0),
                     fit_intercept:                CategoricalDimension = CategoricalDimension([True]),
                     initial_theta_model_type:     CategoricalDimension = CategoricalDimension(["GBM"]),
                     initial_theta_model_settings: Optional[StructDimension] = None) -> None:

            self.fields = {}
            self.num_moments                  = num_moments
            self.max_svd_size                 = max_svd_size
            self.theta_scaling                = theta_scaling
            self.regularization               = regularization
            self.fit_intercept                = fit_intercept
            self.initial_theta_model_type     = initial_theta_model_type
            self.initial_theta_model_settings = initial_theta_model_settings

            super().__init__(self.fields)

    def __init__(self,
                 dimensions: StructDimension,
                 input_features: Optional[FeatureSpec] = None,
                 predicted_features: Optional[FeatureSpec] = None) -> None:
        assert isinstance(dimensions, DMDPODSearchSpace.Dimension), (
            f"dimensions must be a DMDPODSearchSpace.Dimension, got {type(dimensions)}"
        )
        if input_features is None:
            assert "input_features" in dimensions.fields, \
                "DMDPODSearchSpace requires 'input_features' either as defaults or in dimensions"
        if predicted_features is None:
            assert "predicted_features" in dimensions.fields, \
                "DMDPODSearchSpace requires 'predicted_features' either as defaults or in dimensions"
        super().__init__(prediction_strategy_type="DMDPODStrategy",
                         dimensions=dimensions,
                         input_features=input_features,
                         predicted_features=predicted_features)
