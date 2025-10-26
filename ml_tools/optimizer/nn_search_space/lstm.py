from typing import get_args

from ml_tools.model.nn_strategy.layer import Activation
from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import IntDimension, CategoricalDimension, FloatDimension, BoolDimension


class LSTM(Layer):
    """A search-space dimension for an LSTM layer.

    Parameters
    ----------
    units : IntDimension
        Dimensionality of the output space.
    activation : CategoricalDimension
        Activation function to use for the output/apply step.
    recurrent_activation : CategoricalDimension, optional
        Activation for the recurrent step. Defaults to ['sigmoid'].
    recurrent_dropout_rate : FloatDimension, optional
        Recurrent dropout fraction. Defaults to 0.0.
    dropout_rate : FloatDimension, optional
        Dropout rate after the layer output. Defaults to 0.0.
    batch_normalize : BoolDimension, optional
        Whether to apply batch normalization. Defaults to False.
    layer_normalize : BoolDimension, optional
        Whether to apply layer normalization. Defaults to False.

    Attributes
    ----------
    units : IntDimension
        Output dimensionality.
    activation : CategoricalDimension
        Output activation choices.
    recurrent_activation : CategoricalDimension
        Recurrent activation choices.
    recurrent_dropout_rate : FloatDimension
        Recurrent dropout rate.
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

    @property
    def recurrent_activation(self) -> CategoricalDimension:
        return self.fields["recurrent_activation"]

    @recurrent_activation.setter
    def recurrent_activation(self, value: CategoricalDimension) -> None:
        valid = set(get_args(Activation))
        assert all(isinstance(choice, str) and choice in valid for choice in value.choices), \
            f"recurrent_activation.choices must be among {sorted(valid)}, got {value.choices}"
        self.fields["recurrent_activation"] = value

    @property
    def recurrent_dropout_rate(self) -> FloatDimension:
        return self.fields["recurrent_dropout_rate"]

    @recurrent_dropout_rate.setter
    def recurrent_dropout_rate(self, value: FloatDimension) -> None:
        assert 0.0 <= value.low <= 1.0, f"recurrent_dropout_rate.low = {value.low}"
        self.fields["recurrent_dropout_rate"] = value

    def __init__(self,
                 units:                  IntDimension,
                 activation:             CategoricalDimension,
                 recurrent_activation:   CategoricalDimension = CategoricalDimension(["sigmoid"]),
                 recurrent_dropout_rate: FloatDimension       = FloatDimension(0.0, 0.0),
                 dropout_rate:           FloatDimension       = FloatDimension(0.0, 0.0),
                 batch_normalize:        BoolDimension        = BoolDimension([False]),
                 layer_normalize:        BoolDimension        = BoolDimension([False])) -> None:
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.units                  = units
        self.activation             = activation
        self.recurrent_activation   = recurrent_activation
        self.recurrent_dropout_rate = recurrent_dropout_rate
