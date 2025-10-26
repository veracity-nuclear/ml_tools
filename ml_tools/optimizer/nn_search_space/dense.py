from typing import get_args

from ml_tools.model.nn_strategy.layer import Activation
from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import IntDimension, CategoricalDimension, FloatDimension, BoolDimension

class Dense(Layer):
    """ A class representing a dense layer in a neural network hyperparameter search space

    Parameters
    ----------
    units : IntDimension
        The number of units in the dense layer
    activation : CategoricalDimension
        The activation function to use in the dense layer (defaults to "relu")
    dropout_rate : FloatDimension
        The dropout rate to use in the layer (defaults to 0.0, meaning no dropout)
    batch_normalize : BoolDimension
        Whether to apply batch normalization after the layer (defaults to False)
    layer_normalize : BoolDimension
        Whether to apply layer normalization after the layer (defaults to False)

    Attributes
    ----------
    units : IntDimension
        The number of units in the dense layer
    activation : CategoricalDimension
        The activation function to use in the dense layer
    """

    @property
    def units(self) -> IntDimension:
        return self.fields["units"]

    @units.setter
    def units(self, value: IntDimension) -> None:
        assert value.low > 0, f"units.low = {value.low}"
        self.fields["units"] = value

    @property
    def activation(self) -> CategoricalDimension:
        return self.fields["activation"]

    @activation.setter
    def activation(self, value: CategoricalDimension) -> None:
        valid = set(get_args(Activation))
        assert all(isinstance(choice, str) and choice in valid for choice in value.choices), \
            f"activation.choices must be among {sorted(valid)}, got {value.choices}"
        self.fields["activation"] = value

    def __init__(self,
                 units:           IntDimension,
                 activation:      CategoricalDimension,
                 dropout_rate:    FloatDimension = FloatDimension(0., 0.),
                 batch_normalize: BoolDimension  = BoolDimension([False]),
                 layer_normalize: BoolDimension  = BoolDimension([False])) -> None:
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.units      = units
        self.activation = activation
