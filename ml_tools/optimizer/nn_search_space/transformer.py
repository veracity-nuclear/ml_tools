from typing import get_args

from ml_tools.model.nn_strategy.layer import Activation
from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import IntDimension, CategoricalDimension, FloatDimension, BoolDimension


class Transformer(Layer):
    """Search-space dimension for a Transformer layer (domains, not values).

    Parameters
    ----------
    num_heads : IntDimension
        Inclusive range for the number of attention heads.
    model_dim : IntDimension
        Inclusive range for the model/embedding dimension.
    ff_dim : IntDimension
        Inclusive range for the FFN hidden dimension.
    activation : CategoricalDimension, optional
        Choices for FFN activation (default ['relu']).
    dropout_rate : FloatDimension, optional
        Domain for dropout within the transformer block (default 0.0).

    Attributes
    ----------
    num_heads : IntDimension
        Domain for attention head count.
    model_dim : IntDimension
        Domain for embedding dimension.
    ff_dim : IntDimension
        Domain for FFN hidden dimension.
    activation : CategoricalDimension
        Domain for FFN activation choices.
    """

    @property
    def num_heads(self) -> IntDimension:
        return self.fields["num_heads"]

    @num_heads.setter
    def num_heads(self, value: IntDimension) -> None:
        assert value.low > 0, f"num_heads.low = {value.low}"
        self.fields["num_heads"] = value

    @property
    def model_dim(self) -> IntDimension:
        return self.fields["model_dim"]

    @model_dim.setter
    def model_dim(self, value: IntDimension) -> None:
        assert value.low > 0, f"model_dim.low = {value.low}"
        self.fields["model_dim"] = value

    @property
    def ff_dim(self) -> IntDimension:
        return self.fields["ff_dim"]

    @ff_dim.setter
    def ff_dim(self, value: IntDimension) -> None:
        assert value.low > 0, f"ff_dim.low = {value.low}"
        self.fields["ff_dim"] = value

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
                 num_heads:    IntDimension,
                 model_dim:    IntDimension,
                 ff_dim:       IntDimension,
                 activation:   CategoricalDimension = CategoricalDimension(["relu"]),
                 dropout_rate: FloatDimension       = FloatDimension(0.0, 0.0),
                 batch_normalize: BoolDimension     = BoolDimension([False]),
                 layer_normalize: BoolDimension     = BoolDimension([False])) -> None:
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.num_heads  = num_heads
        self.model_dim  = model_dim
        self.ff_dim     = ff_dim
        self.activation = activation
