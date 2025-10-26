from typing import get_args

from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import IntDimension, CategoricalDimension, FloatDimension, BoolDimension
from ml_tools.model.nn_strategy.layer import Activation


class SpatialConv(Layer):
    """A search-space dimension for a SpatialConv CNN layer.

    Parameters
    ----------
    input_shape : CategoricalDimension
        Choice of spatial input shape tuples matching `ShapeType` (e.g., `(H, W)` or `(H, W, D)`).
    activation : CategoricalDimension
        Choice of activation function name. Must be one of the allowed `Activation` values.
    filters : IntDimension
        Number of convolution filters.
    kernel_size : CategoricalDimension
        Choice of convolution kernel sizes as tuples matching `ShapeType`.
    strides : CategoricalDimension
        Choice of convolution strides as tuples matching `ShapeType`.
    padding : BoolDimension
        Whether to apply padding (`True` -> same, `False` -> valid).
    dropout_rate : FloatDimension, optional
        Dropout rate after activation. Defaults to 0.0.
    batch_normalize : BoolDimension, optional
        Whether to apply batch normalization. Defaults to False.
    layer_normalize : BoolDimension, optional
        Whether to apply layer normalization. Defaults to False.

    Attributes
    ----------
    input_shape : CategoricalDimension
        Spatial input shape choices.
    activation : CategoricalDimension
        Activation function choices.
    filters : IntDimension
        Number of filters.
    kernel_size : CategoricalDimension
        Kernel size choices.
    strides : CategoricalDimension
        Stride choices.
    padding : BoolDimension
        Padding choice.
    """

    @property
    def input_shape(self) -> CategoricalDimension:
        return self.fields["input_shape"]

    @input_shape.setter
    def input_shape(self, value: CategoricalDimension) -> None:
        assert isinstance(value, CategoricalDimension)
        self.fields["input_shape"] = value

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
    def filters(self) -> IntDimension:
        return self.fields["filters"]

    @filters.setter
    def filters(self, value: IntDimension) -> None:
        assert value.low > 0
        self.fields["filters"] = value

    @property
    def kernel_size(self) -> CategoricalDimension:
        return self.fields["kernel_size"]

    @kernel_size.setter
    def kernel_size(self, value: CategoricalDimension) -> None:
        assert isinstance(value, CategoricalDimension)
        self.fields["kernel_size"] = value

    @property
    def strides(self) -> CategoricalDimension:
        return self.fields["strides"]

    @strides.setter
    def strides(self, value: CategoricalDimension) -> None:
        assert isinstance(value, CategoricalDimension)
        self.fields["strides"] = value

    @property
    def padding(self) -> BoolDimension:
        return self.fields["padding"]

    @padding.setter
    def padding(self, value: BoolDimension) -> None:
        self.fields["padding"] = value

    def __init__(self,
                 input_shape:  CategoricalDimension,
                 activation:   CategoricalDimension,
                 filters:      IntDimension,
                 kernel_size:  CategoricalDimension,
                 strides:      CategoricalDimension,
                 padding:      BoolDimension,
                 dropout_rate: FloatDimension = FloatDimension(0., 0.),
                 batch_normalize: BoolDimension = BoolDimension([False]),
                 layer_normalize: BoolDimension = BoolDimension([False])) -> None:
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.input_shape = input_shape
        self.activation  = activation
        self.filters     = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.padding     = padding


class SpatialMaxPool(Layer):
    """A search-space dimension for a SpatialMaxPool layer.

    Parameters
    ----------
    input_shape : CategoricalDimension
        Choice of spatial input shape tuples matching `ShapeType` (e.g., `(H, W)` or `(H, W, D)`).
    pool_size : CategoricalDimension
        Choice of pooling window sizes as tuples matching `ShapeType`.
    strides : CategoricalDimension
        Choice of pooling strides as tuples matching `ShapeType`.
    padding : BoolDimension
        Whether to apply padding (`True` -> same, `False` -> valid).
    dropout_rate : FloatDimension, optional
        Dropout rate after pooling. Defaults to 0.0.
    batch_normalize : BoolDimension, optional
        Whether to apply batch normalization. Defaults to False.
    layer_normalize : BoolDimension, optional
        Whether to apply layer normalization. Defaults to False.

    Attributes
    ----------
    input_shape : CategoricalDimension
        Spatial input shape choices.
    pool_size : CategoricalDimension
        Pool size choices.
    strides : CategoricalDimension
        Stride choices.
    padding : BoolDimension
        Padding choice.
    """

    @property
    def input_shape(self) -> CategoricalDimension:
        return self.fields["input_shape"]

    @input_shape.setter
    def input_shape(self, value: CategoricalDimension) -> None:
        assert isinstance(value, CategoricalDimension)
        self.fields["input_shape"] = value

    @property
    def pool_size(self) -> CategoricalDimension:
        return self.fields["pool_size"]

    @pool_size.setter
    def pool_size(self, value: CategoricalDimension) -> None:
        assert isinstance(value, CategoricalDimension)
        self.fields["pool_size"] = value

    @property
    def strides(self) -> CategoricalDimension:
        return self.fields["strides"]

    @strides.setter
    def strides(self, value: CategoricalDimension) -> None:
        assert isinstance(value, CategoricalDimension)
        self.fields["strides"] = value

    @property
    def padding(self) -> BoolDimension:
        return self.fields["padding"]

    @padding.setter
    def padding(self, value: BoolDimension) -> None:
        self.fields["padding"] = value

    def __init__(self,
                 input_shape:  CategoricalDimension,
                 pool_size:    CategoricalDimension,
                 strides:      CategoricalDimension,
                 padding:      BoolDimension,
                 dropout_rate: FloatDimension = FloatDimension(0., 0.),
                 batch_normalize: BoolDimension = BoolDimension([False]),
                 layer_normalize: BoolDimension = BoolDimension([False])) -> None:
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.input_shape = input_shape
        self.pool_size   = pool_size
        self.strides     = strides
        self.padding     = padding
