from __future__ import annotations
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

    Notes
    -----
    - For parameter semantics and valid ranges, see
      ``ml_tools.model.pod_strategy.PODStrategy``.
    """

    class Dimension(StructDimension):
        """POD hyperparameter dimensions.

        Parameters
        ----------
        fine_to_coarse_map : CategoricalDimension
            Choice of fine-to-coarse mapping matrices as nested lists. Each
            row must sum to 1.0 in the final model, and have shape (M, N).
        nclusters : IntDimension, optional
            Number of k-means clusters to create separate POD models for.
        max_svd_size : CategoricalDimension, optional
            Optional maximum number of training samples to use for the SVD of a
            cluster POD model. May be ``None``.
        ndims : CategoricalDimension, optional
            Optional number of dimensions for PCA projection prior to clustering.

        Attributes
        ----------
        fine_to_coarse_map : CategoricalDimension
            Choice of fine-to-coarse mapping matrices as nested lists. Each
            row must sum to 1.0 in the final model, and have shape (M, N).
        nclusters : IntDimension
            Number of k-means clusters to create separate POD models for.
        max_svd_size : CategoricalDimension
            Optional maximum number of training samples to use for the SVD of a
            cluster POD model. May be ``None``.
        ndims : CategoricalDimension
            Optional number of dimensions for PCA projection prior to clustering.
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

    def __init__(self, dimensions: StructDimension) -> None:
        assert isinstance(dimensions, PODSearchSpace.Dimension), (
            f"dimensions must be a PODSearchSpace.Dimension, got {type(dimensions)}"
        )
        super().__init__(prediction_strategy_type="PODStrategy", dimensions=dimensions)

