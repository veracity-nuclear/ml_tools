from typing import get_args

from ml_tools.model.nn_strategy.layer import Activation
from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import IntDimension, CategoricalDimension, FloatDimension, BoolDimension

class Dense(Layer):
    """Search-space dimension for a Dense layer (domains, not values).

    Parameters
    ----------
    units : IntDimension
        Inclusive range for the number of units in the layer.
    activation : CategoricalDimension
        Choices over allowed activation names (e.g., 'relu', 'tanh').
    dropout_rate : FloatDimension
        Domain for dropout fraction after the layer output.
    batch_normalize : BoolDimension
        Domain for whether to apply batch normalization.
    layer_normalize : BoolDimension
        Domain for whether to apply layer normalization.

    Attributes
    ----------
    units : IntDimension
        Domain for unit count.
    activation : CategoricalDimension
        Domain for activation choices.
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
