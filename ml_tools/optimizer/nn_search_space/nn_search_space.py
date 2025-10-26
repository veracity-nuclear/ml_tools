from __future__ import annotations
from typing import List


from ml_tools.optimizer.search_space import SearchSpace, StructDimension, IntDimension, FloatDimension, ListDimension
from ml_tools.optimizer.nn_search_space.layer import Layer

class NNSearchSpace(SearchSpace):
    """ A class representing a neural network hyperparameter search space

    Parameters
    ----------
    dimensions : NNSearchSpace.Dimension
        The root hyperparameter search space to explore
    """

    class Dimension(StructDimension):
        """ A class representing the dimensions of a neural network hyperparameter search space

        Parameters
        ----------
        layers : List[Layer]
            A sequence of neural network layers
        initial_learning_rate : FloatDimension
            The initial learning rate for training (default 0.1)
        learning_decay_rate : FloatDimension
            The learning rate decay factor (default 1.0, meaning no decay)
        epoch_limit : IntDimension
            The maximum number of training epochs (default 1000)
        convergence_criteria : FloatDimension
            The convergence criteria for early stopping (default 1e-3)
        convergence_patience : IntDimension
            The number of epochs with no improvement to wait before stopping (default 5)
        batch_size_log2 : IntDimension
            The base-2 logarithm of the batch size to use during training (default 7, meaning batch size of 128)

        Attributes
        ----------
        layers : List[Layer]
            A sequence of neural network layers
        initial_learning_rate : FloatDimension
            The initial learning rate for training (default 0.1)
        learning_decay_rate : FloatDimension
            The learning rate decay factor (default 1.0, meaning no decay)
        epoch_limit : IntDimension
            The maximum number of training epochs (default 1000)
        convergence_criteria : FloatDimension
            The convergence criteria for early stopping (default 1e-3)
        convergence_patience : IntDimension
            The number of epochs with no improvement to wait before stopping (default 5)
        batch_size_log2 : IntDimension
            The base-2 logarithm of the batch size to use during training (default 7, meaning batch size of 128)
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


    def __init__(self, dimensions: StructDimension) -> None:
        assert isinstance(dimensions, NNSearchSpace.Dimension), f"dimensions must be a NNSearchSpace.Dimension, got {type(dimensions)}"
        super().__init__(prediction_strategy_type="NNStrategy", dimensions=dimensions)
