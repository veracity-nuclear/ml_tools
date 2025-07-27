from __future__ import annotations
from typing import Any, Union, Tuple
from math import isclose
from decimal import Decimal
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf

from ml_tools.model.nn_strategy.layer import Layer, Activation


ShapeType = Union[
    Tuple[int],           # 1D shape: (H,)
    Tuple[int, int],      # 2D shape: (H, W)
    Tuple[int, int, int]  # 3D shape: (H, W, D)
]

def _extend_shape(shape: ShapeType) -> Tuple[int, int, int]:
    """Extends a 1D or 2D tuple to a 3D tuple by appending ones.

    Parameters
    ----------
    shape : ShapeType
        A tuple representing 1D, 2D, or 3D spatial dimensions.

    Returns
    -------
    Tuple[int, int, int]
        A 3D tuple where missing dimensions are filled with 1s.
    """
    if len(shape) == 1:
        return (shape[0], 1, 1)  # Convert (H,) -> (H, 1, 1)
    if len(shape) == 2:
        return (shape[0], shape[1], 1)  # Convert (H, W) -> (H, W, 1)
    if len(shape) == 3:
        return shape  # Already 3D
    raise ValueError(f"Invalid shape {shape}. Expected a 1D, 2D, or 3D tuple.")


@Layer.register_subclass("SpatialConv")
class SpatialConv(Layer):
    """ A Spatial Convolutional Neural Network (CNN) layer

    Parameters
    ----------
    input_shape : ShapeType
        The height, width, and depth of the input data before convolution (i.e. its 3D shape)
    activation : Activation
        Activation function to use
    filters : int
        Number of convolution filters
    kernel_size : ShapeType
        Size of the convolution kernel
    strides : ShapeType
        Strides of the convolution
    padding : bool
        Whether or not padding should be applied to the convolution
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout


    Attributes
    ----------
    input_shape : Tuple[int, int, int]
        The height, width, and depth of the input data before convolution (i.e. its 3D shape)
    activation : Activation
        Activation function to use
    filters : int
        Number of convolution filters
    kernel_size : Tuple[int, int, int]
        Size of the convolution kernel
    strides : Tuple[int, int, int]
        Strides of the convolution
    padding : bool
        Whether or not padding should be applied to the convolution
    """

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: ShapeType) -> None:
        input_shape = _extend_shape(input_shape)
        assert input_shape[0] > 0, f"input_shape[0] = {input_shape[0]}"
        assert input_shape[1] > 0, f"input_shape[1] = {input_shape[1]}"
        assert input_shape[2] > 0, f"input_shape[2] = {input_shape[2]}"
        self._input_shape = input_shape

    @property
    def activation(self) -> Activation:
        return self._activation

    @activation.setter
    def activation(self, activation: Activation) -> None:
        self._activation = activation

    @property
    def filters(self) -> int:
        return self._filters

    @filters.setter
    def filters(self, filters: int) -> None:
        assert filters > 0, f"filters = {filters}"
        self._filters = filters

    @property
    def kernel_size(self) -> Tuple[int, int, int]:
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size: ShapeType) -> None:
        kernel_size = _extend_shape(kernel_size)
        assert kernel_size[0] > 0, f"kernel_size[0] = {kernel_size[0]}"
        assert kernel_size[1] > 0, f"kernel_size[1] = {kernel_size[1]}"
        assert kernel_size[2] > 0, f"kernel_size[2] = {kernel_size[2]}"
        self._kernel_size = kernel_size

    @property
    def strides(self) -> Tuple[int, int, int]:
        return self._strides

    @strides.setter
    def strides(self, strides: ShapeType) -> None:
        strides = _extend_shape(strides)
        assert strides[0] > 0, f"strides[0] = {strides[0]}"
        assert strides[1] > 0, f"strides[1] = {strides[1]}"
        assert strides[2] > 0, f"strides[2] = {strides[2]}"
        self._strides = strides

    @property
    def padding(self) -> bool:
        return self._padding

    @padding.setter
    def padding(self, padding: bool) -> None:
        self._padding = padding


    def __init__(self,
                 input_shape:      ShapeType,
                 activation:       str = 'relu',
                 filters:          int = 1,
                 kernel_size:      ShapeType = (1,),
                 strides:          ShapeType = (1,),
                 padding:          bool = True,
                 dropout_rate:     float = 0.,
                 batch_normalize:  bool = False,
                 layer_normalize:  bool = False):
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.input_shape  = input_shape
        self.activation   = activation
        self.filters      = filters
        self.kernel_size  = kernel_size
        self.strides      = strides
        self.padding      = padding

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, SpatialConv) and
                  self.input_shape == other.input_shape and
                  self.activation == other.activation and
                  self.filters == other.filters and
                  self.kernel_size == other.kernel_size and
                  self.strides == other.strides and
                  self.padding == other.padding and
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                  self.batch_normalize == other.batch_normalize and
                  self.layer_normalize == other.layer_normalize)
        )

    def __hash__(self) -> int:
        return hash(tuple(self.input_shape,
                          self.activation,
                          self.filters,
                          self.kernel_size,
                          self.strides,
                          self.padding,
                          Decimal(self.dropout_rate).quantize(Decimal('1e-9'))),
                          self.batch_normalize,
                          self.layer_normalize
                   )

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        assert input_tensor.shape[-1] % (self.input_shape[0] * self.input_shape[1] * self.input_shape[2]) == 0, \
            "Input tensor shape is not divisible by the expected input 3D shape"

        number_of_channels = input_tensor.shape[-1] // (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        input_shape = (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2], number_of_channels)
        x = tf.keras.layers.Reshape(target_shape=input_shape)(input_tensor)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(filters     = self.filters,
                                                                   kernel_size = self.kernel_size,
                                                                   strides     = self.strides,
                                                                   padding     = 'same' if self.padding else 'valid',
                                                                   activation  = None))(x)

        if self.batch_normalize:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)

        if self.layer_normalize:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(x)

        x = tf.keras.layers.Activation(self.activation)(x)

        if self.dropout_rate > 0.:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.SpatialDropout3D(rate=self.dropout_rate))(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type',                data='SpatialConv', dtype=h5py.string_dtype())
        group.create_dataset('input_shape',         data=self.input_shape)
        group.create_dataset('activation_function', data=self.activation, dtype=h5py.string_dtype())
        group.create_dataset('number_of_filters',   data=self.filters)
        group.create_dataset('kernel_size',         data=self.kernel_size)
        group.create_dataset('strides',             data=self.strides)
        group.create_dataset('padding',             data=self.padding)
        group.create_dataset('dropout_rate',        data=self.dropout_rate)
        group.create_dataset('batch_normalize',     data=self.batch_normalize)
        group.create_dataset('layer_normalize',     data=self.layer_normalize)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> SpatialConv:
        return cls(input_shape            = tuple(int(x) for x in group['input_shape'  ][()]),
                   activation             =       group["activation_function"          ][()].decode('utf-8'),
                   filters                =   int(group["number_of_filters"            ][()]),
                   kernel_size            = tuple(int(x) for x in group['kernel_size'  ][()]),
                   strides                = tuple(int(x) for x in group['strides'      ][()]),
                   padding                =  bool(group["padding"                      ][()]),
                   dropout_rate           = float(group["dropout_rate"                 ][()]),
                   batch_normalize        =  bool(group["batch_normalize"              ][()]),
                   layer_normalize        =  bool(group["layer_normalize"              ][()]))


@Layer.register_subclass("SpatialMaxPool")
class SpatialMaxPool(Layer):
    """ A Spatial Max Pool layer

    Parameters
    ----------
    input_shape : ShapeType
        The height, width, and depth of the input data before convolution (i.e. its 3D shape)
    pool_size : ShapeType
        Size of the pooling window
    strides : ShapeType
        Strides of the convolution
    padding : bool
        Whether or not padding should be applied to the convolution
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout


    Attributes
    ----------
    input_shape : Tuple[int, int, int]
        The height and width of the input data before convolution (i.e. its 2D shape)
    pool_size : Tuple[int, int, int]
        Size of the pooling window
    strides : Tuple[int, int, int]
        Strides of the convolution
    padding : bool
        Whether or not padding should be applied to the convolution
    """

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: ShapeType) -> None:
        input_shape = _extend_shape(input_shape)
        assert input_shape[0] > 0, f"input_shape[0] = {input_shape[0]}"
        assert input_shape[1] > 0, f"input_shape[1] = {input_shape[1]}"
        assert input_shape[2] > 0, f"input_shape[2] = {input_shape[2]}"
        self._input_shape = input_shape

    @property
    def pool_size(self) -> Tuple[int, int, int]:
        return self._pool_size

    @pool_size.setter
    def pool_size(self, pool_size: ShapeType) -> None:
        pool_size = _extend_shape(pool_size)
        assert pool_size[0] > 0, f"pool_size[0] = {pool_size[0]}"
        assert pool_size[1] > 0, f"pool_size[1] = {pool_size[1]}"
        assert pool_size[2] > 0, f"pool_size[2] = {pool_size[2]}"
        self._pool_size = pool_size

    @property
    def strides(self) -> Tuple[int, int, int]:
        return self._strides

    @strides.setter
    def strides(self, strides: ShapeType) -> None:
        strides = _extend_shape(strides)
        assert strides[0] > 0, f"strides[0] = {strides[0]}"
        assert strides[1] > 0, f"strides[1] = {strides[1]}"
        assert strides[2] > 0, f"strides[2] = {strides[2]}"
        self._strides = strides

    @property
    def padding(self) -> bool:
        return self._padding

    @padding.setter
    def padding(self, padding: bool) -> None:
        self._padding = padding


    def __init__(self,
                 input_shape:      ShapeType,
                 pool_size:        ShapeType = (1,),
                 strides:          ShapeType = (1,),
                 padding:          bool = True,
                 dropout_rate:     float = 0.,
                 batch_normalize:  bool = False,
                 layer_normalize:  bool = False):
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.input_shape = input_shape
        self.pool_size   = pool_size
        self.strides     = strides
        self.padding     = padding

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, SpatialMaxPool) and
                  self.input_shape == other.input_shape and
                  self.pool_size == other.pool_size and
                  self.strides == other.strides and
                  self.padding == other.padding and
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                  self.batch_normalize == other.batch_normalize and
                  self.layer_normalize == other.layer_normalize)
        )

    def __hash__(self) -> int:
        return hash(tuple(self.input_shape,
                          self.pool_size,
                          self.strides,
                          self.padding,
                          Decimal(self.dropout_rate).quantize(Decimal('1e-9'))),
                          self.batch_normalize,
                          self.layer_normalize
                   )

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        assert input_tensor.shape[-1] % (self.input_shape[0] * self.input_shape[1] * self.input_shape[2]) == 0, \
            "Input tensor shape is not divisible by the expected input 3D shape"

        number_of_channels = input_tensor.shape[-1] // (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        input_shape = (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2], number_of_channels)
        x = tf.keras.layers.Reshape(target_shape=input_shape)(input_tensor)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling3D(pool_size = self.pool_size,
                                                                         strides   = self.strides,
                                                                         padding   = 'same' if self.padding else 'valid'))(x)

        if self.batch_normalize:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)

        if self.layer_normalize:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(x)

        if self.dropout_rate > 0.:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.SpatialDropout3D(rate=self.dropout_rate))(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type',             data='SpatialMaxPool', dtype=h5py.string_dtype())
        group.create_dataset('input_shape',      data=self.input_shape)
        group.create_dataset('pool_size',        data=self.pool_size)
        group.create_dataset('strides',          data=self.strides)
        group.create_dataset('padding',          data=self.padding)
        group.create_dataset('dropout_rate',     data=self.dropout_rate)
        group.create_dataset('batch_normalize',  data=self.batch_normalize)
        group.create_dataset('layer_normalize',  data=self.layer_normalize)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> SpatialMaxPool:
        return cls(input_shape      = tuple(int(x) for x in group['input_shape'][()]),
                   pool_size        = tuple(int(x) for x in group['pool_size'  ][()]),
                   strides          = tuple(int(x) for x in group['strides'    ][()]),
                   padding          =  bool(group["padding"                    ][()]),
                   dropout_rate     = float(group["dropout_rate"               ][()]),
                   batch_normalize  =  bool(group["batch_normalize"            ][()]),
                   layer_normalize  =  bool(group["layer_normalize"            ][()]))
