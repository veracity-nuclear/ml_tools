from __future__ import annotations
from ml_tools.optimizer.search_space import (
    SearchSpace,
    StructDimension,
    IntDimension,
    CategoricalDimension,
)
from ml_tools.optimizer.gbm_search_space import GBMSearchSpace


class EnhancedPODSearchSpace(SearchSpace):
    """EnhancedPODStrategy hyperparameter search space.

    Parameters
    ----------
    dimensions : EnhancedPODSearchSpace.Dimension
        The root hyperparameter search space to explore.
    input_features : Dict[str, FeatureProcessor]
        Input feature processors keyed by feature name.
    predicted_feature : str
        Name of the target feature to predict.
    biasing_model : Optional[PredictionStrategy], optional
        Optional prior model to bias predictions, by default None.
    """

    class Dimension(StructDimension):
        """GBM hyperparameter domains (to be sampled; not final values).

        Parameters
        ----------
        num_moments : CategoricalDimension, optional
            Number of POD modments
		gamma : FloatDimension, optional
		    Comment here
		gbm_settings : CategoricalDimension, optional
	        Arguments to pass to underlying GBMStrategy
		

        Attributes
        ----------
        max_svd_size : CategoricalDimension
            Comment here
        num_moments : CategoricalDimension
            Domain for number of POD moments
		gamma : FloatDimension
		    Comment here
		gbm_settings : CategoricalDimension
		    Arguments to pass to underlying GBMStrategy
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
        def gbm_settings(self) -> GBMSearchSpace.Dimension:
            return self.fields["gbm_settings"]

        @gbm_settings.setter
        def gbm_settings(self, value: GBMSearchSpace.Dimension) -> None:
            self.fields["gbm_settings"] = value


        def __init__(self,
                     max_svd_size:   CategoricalDimension = CategoricalDimension([None]),
                     num_moments:    IntDimension = IntDimension(6, 6),
                     constraints:    CategoricalDimension=CategoricalDimension([[]]),
                     gbm_settings:   GBMSearchSpace.Dimension = GBMSearchSpace.Dimension()) -> None:

            self.fields = {}
            self.max_svd_size = max_svd_size
            self.num_moments  = num_moments
            self.constraints = constraints
            self.gbm_settings = gbm_settings

            super().__init__(self.fields)

    def __init__(self,
                 dimensions: StructDimension,
                 input_features=None,
                 predicted_feature=None,
                 biasing_model=None) -> None:
        assert isinstance(dimensions, EnhancedPODSearchSpace.Dimension), (
            f"dimensions must be a EnhancedPODSearchSpace.Dimension, got {type(dimensions)}"
        )
        super().__init__(prediction_strategy_type="EnhancedPODStrategy",
                         dimensions=dimensions,
                         input_features=input_features,
                         predicted_feature=predicted_feature,
                         biasing_model=biasing_model)
