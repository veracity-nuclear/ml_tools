from __future__ import annotations
from typing import List


from ml_tools.optimizer.search_space import SearchSpace, StructDimension, IntDimension, FloatDimension, ListDimension
from ml_tools.optimizer.nn_search_space.layer import Layer


class NNSearchSpace(SearchSpace):
    """Neural network hyperparameter search space.

    Parameters
    ----------
    dimensions : NNSearchSpace.Dimension
        Root struct of dimensions to sample. Each field is a
        SearchSpace dimension (range or choices), not a finalized value.
    input_features : Dict[str, FeatureProcessor]
        Input feature processors keyed by feature name.
    predicted_features : Dict[str, FeatureProcessor]
        Output features and their processors.
    biasing_model : Optional[PredictionStrategy], optional
        Optional prior model to bias predictions, by default None.
    """

    class Dimension(StructDimension):
        """Dimensions for neural network optimization (domains, not values).

        Parameters
        ----------
        layers : List[Layer]
            Sequence of layer dimension objects. Wrapped internally as a
            ListDimension; each element is itself a layer dimension to sample.
        initial_learning_rate : FloatDimension
            Range/domain for the initial learning rate (inclusive bounds; may be log-scaled if `log=True`).
        learning_decay_rate : FloatDimension
            Range/domain for multiplicative LR decay per step/epoch (e.g., 1.0 means no decay).
        epoch_limit : IntDimension
            Inclusive range for the maximum number of training epochs to allow.
        convergence_criteria : FloatDimension
            Range/domain for early-stopping tolerance (smaller -> stricter stopping).
        convergence_patience : IntDimension
            Inclusive range for patience (epochs without improvement) before stopping.
        batch_size_log2 : IntDimension
            Inclusive range for log2(batch size). For example, 7 -> batch size of 128.

        Attributes
        ----------
        layers : List[Layer]
            Underlying list of layer dimensions (via ListDimension wrapper).
        initial_learning_rate : FloatDimension
            Domain for initial LR to sample from.
        learning_decay_rate : FloatDimension
            Domain for LR decay factor to sample from.
        epoch_limit : IntDimension
            Domain for epoch limit to sample from.
        convergence_criteria : FloatDimension
            Domain for early-stopping tolerance to sample from.
        convergence_patience : IntDimension
            Domain for early-stopping patience to sample from.
        batch_size_log2 : IntDimension
            Domain for log2(batch size) to sample from.
        """

        @property
        def layers(self) -> List[Layer]:
            return self.fields["layers"].items

        @layers.setter
        def layers(self, value: List[Layer]) -> None:
            self.fields["layers"] = ListDimension(items=value, label="layer")

        @property
        def initial_learning_rate(self) -> FloatDimension:
            return self.fields["initial_learning_rate"]

        @initial_learning_rate.setter
        def initial_learning_rate(self, value: FloatDimension) -> None:
            assert value.low > 0, f"initial_learning_rate lower bound {value.low} must be > 0"
            self.fields["initial_learning_rate"] = value

        @property
        def learning_decay_rate(self) -> FloatDimension:
            return self.fields["learning_decay_rate"]

        @learning_decay_rate.setter
        def learning_decay_rate(self, value: FloatDimension) -> None:
            assert value.low > 0, f"learning_decay_rate lower bound {value.low} must be > 0"
            assert value.high <= 1.0, f"learning_decay_rate upper bound {value.high} must be <= 1.0"
            self.fields["learning_decay_rate"] = value

        @property
        def epoch_limit(self) -> IntDimension:
            return self.fields["epoch_limit"]

        @epoch_limit.setter
        def epoch_limit(self, value: IntDimension) -> None:
            assert value.low > 0, f"epoch_limit lower bound {value.low} must be > 0"
            self.fields["epoch_limit"] = value

        @property
        def convergence_criteria(self) -> FloatDimension:
            return self.fields["convergence_criteria"]

        @convergence_criteria.setter
        def convergence_criteria(self, value: FloatDimension) -> None:
            assert value.low > 0, f"convergence_criteria lower bound {value.low} must be > 0"
            self.fields["convergence_criteria"] = value

        @property
        def convergence_patience(self) -> IntDimension:
            return self.fields["convergence_patience"]

        @convergence_patience.setter
        def convergence_patience(self, value: IntDimension) -> None:
            assert value.low >= 0, f"convergence_patience lower bound {value.low} must be >= 0"
            self.fields["convergence_patience"] = value

        @property
        def batch_size_log2(self) -> IntDimension:
            return self.fields["batch_size_log2"]

        @batch_size_log2.setter
        def batch_size_log2(self, value: IntDimension) -> None:
            assert value.low > 0, f"batch_size_log2 lower bound {value.low} must be > 0"
            self.fields["batch_size_log2"] = value

        def __init__(self,
                     layers:                List[Layer],
                     initial_learning_rate: FloatDimension = FloatDimension(0.1, 0.1),
                     learning_decay_rate:   FloatDimension = FloatDimension(1.0, 1.0),
                     epoch_limit:           IntDimension   = IntDimension(1000, 1000),
                     convergence_criteria:  FloatDimension = FloatDimension(1e-3, 1e-3),
                     convergence_patience:  IntDimension   = IntDimension(5, 5),
                     batch_size_log2:       IntDimension   = IntDimension(7, 7)):

            self.fields                 = {}
            self.layers                 = layers
            self.initial_learning_rate  = initial_learning_rate
            self.learning_decay_rate    = learning_decay_rate
            self.epoch_limit            = epoch_limit
            self.convergence_criteria   = convergence_criteria
            self.convergence_patience   = convergence_patience
            self.batch_size_log2        = batch_size_log2

            super().__init__(self.fields)


    def __init__(self,
                 dimensions: StructDimension,
                 input_features=None,
                 predicted_features=None,
                 biasing_model=None) -> None:
        assert isinstance(dimensions, NNSearchSpace.Dimension), \
            f"dimensions must be a NNSearchSpace.Dimension, got {type(dimensions)}"
        super().__init__(prediction_strategy_type="NNStrategy",
                         dimensions=dimensions,
                         input_features=input_features,
                         predicted_features=predicted_features,
                         biasing_model=biasing_model)
