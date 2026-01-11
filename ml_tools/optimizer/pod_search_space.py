from __future__ import annotations
from typing import Optional

from ml_tools.model.prediction_strategy import FeatureSpec, PredictionStrategy
from ml_tools.optimizer.search_space import (
    SearchSpace,
    StructDimension,
    IntDimension,
    CategoricalDimension,
)


class PODSearchSpace(SearchSpace):
    """POD (Proper Orthogonal Decomposition) hyperparameter search space.

    Parameters
    ----------
    dimensions : PODSearchSpace.Dimension
        The root hyperparameter search space to explore.
    input_features : FeatureSpec
        Input feature/processor pairs (Dict) or feature name(s) (str/List[str], automatically mapped to NoProcessing).
    predicted_features : FeatureSpec
        Output feature/processor pairs (Dict) or feature name(s) (str/List[str], automatically mapped to NoProcessing).
    biasing_model : Optional[PredictionStrategy], optional
        Optional prior model to bias predictions, by default None.

    Notes
    -----
    - For parameter semantics and valid ranges, see
      ``ml_tools.model.pod_strategy.PODStrategy``.
    """

    class Dimension(StructDimension):
        """POD hyperparameter domains (to be sampled; not final values).

        Parameters
        ----------
        fine_to_coarse_map : CategoricalDimension
            Choices over fine-to-coarse mapping matrices as nested lists. Each
            row must sum to 1.0 in the final model, and have shape (M, N).
        nclusters : IntDimension, optional
            Inclusive range for the number of k-means clusters used to partition data.
        max_svd_size : CategoricalDimension, optional
            Choices for an optional maximum number of training samples to use for the SVD of a
            cluster POD model. May include ``None``.
        ndims : CategoricalDimension, optional
            Choices for dimensionality of PCA projection prior to clustering. May include ``None``.

        Attributes
        ----------
        fine_to_coarse_map : CategoricalDimension
            Domain for mapping matrix choices (nested lists with rows summing to 1.0).
        nclusters : IntDimension
            Domain for number of clusters.
        max_svd_size : CategoricalDimension
            Domain for optional SVD sample cap (including ``None``).
        ndims : CategoricalDimension
            Domain for optional PCA dimensionality (including ``None``).
        """

        @property
        def fine_to_coarse_map(self) -> CategoricalDimension:
            return self.fields["fine_to_coarse_map"]

        @fine_to_coarse_map.setter
        def fine_to_coarse_map(self, value: CategoricalDimension) -> None:
            self.fields["fine_to_coarse_map"] = value

        @property
        def nclusters(self) -> IntDimension:
            return self.fields["nclusters"]

        @nclusters.setter
        def nclusters(self, value: IntDimension) -> None:
            self.fields["nclusters"] = value

        @property
        def max_svd_size(self) -> CategoricalDimension:
            return self.fields["max_svd_size"]

        @max_svd_size.setter
        def max_svd_size(self, value: CategoricalDimension) -> None:
            self.fields["max_svd_size"] = value

        @property
        def ndims(self) -> CategoricalDimension:
            return self.fields["ndims"]

        @ndims.setter
        def ndims(self, value: CategoricalDimension) -> None:
            self.fields["ndims"] = value

        def __init__(self,
                     fine_to_coarse_map: CategoricalDimension,
                     nclusters:          IntDimension         = IntDimension(1, 1),
                     max_svd_size:       CategoricalDimension = CategoricalDimension([None]),
                     ndims:              CategoricalDimension = CategoricalDimension([None])) -> None:

            self.fields = {}
            self.fine_to_coarse_map = fine_to_coarse_map
            self.nclusters          = nclusters
            self.max_svd_size       = max_svd_size
            self.ndims              = ndims

            super().__init__(self.fields)

    def __init__(self,
                 dimensions: StructDimension,
                 input_features: FeatureSpec,
                 predicted_features: FeatureSpec,
                 biasing_model: Optional[PredictionStrategy] = None) -> None:
        assert isinstance(dimensions, PODSearchSpace.Dimension), (
            f"dimensions must be a PODSearchSpace.Dimension, got {type(dimensions)}"
        )
        super().__init__(prediction_strategy_type="PODStrategy",
                         dimensions=dimensions,
                         input_features=input_features,
                         predicted_features=predicted_features,
                         biasing_model=biasing_model)
