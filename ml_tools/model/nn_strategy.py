from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Literal, Tuple
from dataclasses import dataclass, field
import os
from math import isclose
import h5py
import numpy as np

import tensorflow as tf
from tensorflow.keras import KerasTensor
import tensorflow.keras.layers
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

from ml_tools.model.state import State, StateSeries
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import FeatureProcessor


LayerType  = Literal['Dense', 'PassThrough', 'LSTM', 'LayerSequence', 'CompoundLayer']
Activation = Literal['elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'mish',
                     'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh']


class Layer(ABC):
    """ Abstract class for neural network layers

    Attributes
    ----------
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    """

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate: float) -> None:
        assert 0.0 <= dropout_rate <= 1.0
        self._dropout_rate = dropout_rate


    def __init__(self, dropout_rate: float = 0.0) -> None:
        self.dropout_rate = dropout_rate


    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        """ Method for constructing the layer

        Parameters
        ----------
        input_tensor : KerasTensor
            The input tensor for the layer
        """
        pass

    @abstractmethod
    def save(self, group: h5py.Group) -> None:
        """ Method for saving the layer to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            The h5py group to save the layer to
        """
        pass

    @classmethod
    @abstractmethod
    def from_h5(cls, group: h5py.Group) -> Layer:
        """ Method for creating a new layer from an HDF5 Group

        Parameters
        ----------
        group : h5py.Group
            The h5py group to build the layer from

        Returns
        -------
        Layer
            The layer constructed from the HDF5 group
        """
        pass



class LayerSequence(Layer):
    """ A class for a sequence of layers

    A layer sequence does not require a dropout rate specification because
    this will be dictated by the final layer's dropout rate specification.
    If a dropout rate is provided, it will be ignored in favor of the final
    layer's dropout rate

    Attributes
    ----------
    layers : List[Layer]
        The list of layers that comprise the sequence
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @layers.setter
    def layers(self, value: List[Layer]) -> None:
        assert len(value) > 0
        self._layers = value

    def __init__(self, layers: List[Layer]) -> None:
        super().__init__(0.0)
        self.layers = layers

    def __eq__(self, other: Any) -> bool:
        if self is other:                                                                       return True
        if not isinstance(other, LayerSequence):                                                return False
        if not(len(self.layers) == len(other.layers)):                                          return False
        if not(all(s_layer == o_layer for s_layer, o_layer in zip(self.layers, other.layers))): return False
        return True

    def __hash__(self) -> int:
        return hash(tuple(self.layers))

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = input_tensor
        for layer in self.layers:
            x = layer.build(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type', data='LayerSequence', dtype=h5py.string_dtype())
        for i, layer in enumerate(self.layers):
            layer_group = group.create_group('layer_' + str(i))
            layer.save(layer_group)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> LayerSequence:
        layers = []
        layer_names = [key for key in group.keys() if key.startswith("layer_")]
        layer_names = sorted(layer_names, key=lambda x: int(x.split('_')[1]))
        for layer_name in layer_names:
            layer_group = group[layer_name]
            layer_type: LayerType = layer_group['type'][()].decode('utf-8')
            if   layer_type == 'Dense':         layers.append(Dense.from_h5(layer_group))
            elif layer_type == 'LSTM':          layers.append(LSTM.from_h5(layer_group))
            elif layer_type == 'Conv2D':        layers.append(Conv2D.from_h5(layer_group))
            elif layer_type == 'MaxPool2D':     layers.append(MaxPool2D.from_h5(layer_group))
            elif layer_type == 'PassThrough':   layers.append(PassThrough.from_h5(layer_group))
            elif layer_type == 'LayerSequence': layers.append(LayerSequence.from_h5(layer_group))
            elif layer_type == 'CompoundLayer': layers.append(CompoundLayer.from_h5(layer_group))
        return cls(layers=layers)


class CompoundLayer(Layer):
    """ A class for compound / composite layers consisting layers that are executed in parallel

    This class effectively splits the input into the layer across multiple layers which will
    each execute in parallel and then merge their output at the end.

    A compound layer does require a dropout rate on account of the merged outputs.  If any
    of the composite layers are provided a drop out rate, said rate will be ignored in favor
    of the compound layer's dropout rate.

    Also, input features need not be "exclusive" to a given layer, but rather may be used by multiple
    constituent layers.

    Attributes
    ----------
    layers : List[Layer]
        The list of constituent layers that will be executed in parallel
    input_specifications : List[List[int]]
        The list of input indices each layer should use to pull from the incoming input
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @property
    def input_specifications(self) -> List[List[int]]:
        return self._input_specifications

    def __init__(self, layers: List[Layer], input_specifications: List[Union[slice, List[int]]], dropout_rate: float = 0.0) -> None:
        super().__init__(dropout_rate)
        assert len(layers) > 0
        assert len(layers) == len(input_specifications)
        assert all(not(spec.stop is None) for spec in input_specifications if isinstance(spec, slice)) # Input layer length is not known until at build
        self._layers = layers

        self._input_specifications = []
        for specification in input_specifications:
            if isinstance(specification, slice):
                specification = list(range(specification.start if specification.start is not None else 0,
                                           specification.stop,
                                           specification.step  if specification.step is not None else 1))
                self._input_specifications.append(specification)
            else:
                self._input_specifications.append(specification)

    def __eq__(self, other: Any) -> bool:
        if self is other:                                                                       return True
        if not isinstance(other, CompoundLayer):                                                return False
        if not(isclose(self.dropout_rate, other.dropout_rate)):                                 return False
        if not(len(self.layers) == len(other.layers)):                                          return False
        if not(all(s_layer == o_layer for s_layer, o_layer in zip(self.layers, other.layers))): return False
        return True

    def __hash__(self) -> int:
        return hash((tuple(self.layers),
                     tuple([tuple(specification) for specification in self.input_specifications]),
                     self.dropout_rate))

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        @register_keras_serializable()
        def gather_indices(x, indices):
            return tf.gather(x, indices, axis=-1)

        assert all(index < input_tensor.shape[2] for spec in self.input_specifications for index in spec)
        split_inputs = [tensorflow.keras.layers.TimeDistributed(tensorflow.keras.layers.Lambda(gather_indices, arguments={'indices': indices}))(input_tensor) for indices in self.input_specifications]

        outputs = [layer.build(split) for layer, split in zip(self._layers, split_inputs)]

        x = tensorflow.keras.layers.Concatenate(axis=-1)(outputs)

        if self.dropout_rate > 0.0:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=self.dropout_rate))(x)

        return x


    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type', data='CompoundLayer', dtype=h5py.string_dtype())
        specs_array = np.array([np.array(inner, dtype=np.int32) for inner in self.input_specifications], dtype=object)
        group.create_dataset('input_specifications', data=specs_array, dtype=h5py.vlen_dtype(np.int32))
        for i, layer in enumerate(self.layers):
            layer_group = group.create_group('layer_' + str(i))
            layer.save(layer_group)
        group.attrs['dropout_rate'] = self._dropout_rate


    @classmethod
    def from_h5(cls, group: h5py.Group) -> CompoundLayer:
        input_specifications = [list(item) for item in group['input_specifications'][:]]
        layers = []
        layer_names = [key for key in group.keys() if key.startswith("layer_")]
        layer_names = sorted(layer_names, key=lambda x: int(x.split('_')[1]))
        for layer_name in layer_names:
            layer_group = group[layer_name]
            layer_type: LayerType = layer_group['type'][()].decode('utf-8')
            if   layer_type == 'Dense':         layers.append(Dense.from_h5(layer_group))
            elif layer_type == 'LSTM':          layers.append(LSTM.from_h5(layer_group))
            elif layer_type == 'Conv2D':        layers.append(Conv2D.from_h5(layer_group))
            elif layer_type == 'MaxPool2D':     layers.append(MaxPool2D.from_h5(layer_group))
            elif layer_type == 'PassThrough':   layers.append(PassThrough.from_h5(layer_group))
            elif layer_type == 'LayerSequence': layers.append(LayerSequence.from_h5(layer_group))
            elif layer_type == 'CompoundLayer': layers.append(CompoundLayer.from_h5(layer_group))
        dropout_rate = group.attrs.get('dropout_rate', 0.0)
        return cls(layers=layers, input_specifications=input_specifications, dropout_rate=dropout_rate)


class PassThrough(Layer):
    """ An layer for passing through input features

    This layer type is useful when constructing composite layers that require passing some features
    straight through to the next layer while other features pass through an actual processing layer.
    """

    def __init__(self, dropout_rate: float = 0.):
        super().__init__(dropout_rate)

    def __eq__(self, other: Any) -> bool:
        if self is other:                                       return True
        if not isinstance(other, PassThrough):                  return False
        if not(isclose(self.dropout_rate, other.dropout_rate)): return False
        return True

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = input_tensor
        if self.dropout_rate > 0.0:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=self.dropout_rate))(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type'         , data='PassThrough', dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate' , data=self.dropout_rate)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> PassThrough:
        return cls(dropout_rate = float(group["dropout_rate"][()]))


class Dense(Layer):
    """ A Dense Neural Network Layer

    Attributes
    ----------
    units : int
        Number of nodes (i.e. neurons) to use in the dense layer
    activation : Activation
        Activation function to use
    """

    @property
    def units(self) -> int:
        return self._units

    @units.setter
    def units(self, units: int) -> None:
        assert units > 0
        self._units = units

    @property
    def activation(self) -> Activation:
        return self._activation

    @activation.setter
    def activation(self, activation: Activation) -> None:
        self._activation = activation


    def __init__(self, units: int, activation: Activation, dropout_rate: float = 0.):
        super().__init__(dropout_rate)
        self.units      = units
        self.activation = activation

    def __eq__(self, other: Any) -> bool:
        if self is other:                                       return True
        if not isinstance(other, Dense):                        return False
        if not(self.units == other.units):                      return False
        if not(self.activation == other.activation):            return False
        if not(isclose(self.dropout_rate, other.dropout_rate)): return False
        return True

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.units, activation=self.activation))(input_tensor)
        if self.dropout_rate > 0.0:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=self.dropout_rate))(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type'                , data='Dense', dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate'        , data=self.dropout_rate)
        group.create_dataset('number_of_units'     , data=self.units)
        group.create_dataset('activation_function' , data=self.activation, dtype=h5py.string_dtype())

    @classmethod
    def from_h5(cls, group: h5py.Group) -> Dense:
        return cls(units        =   int(group["number_of_units"    ][()]),
                   activation   =       group["activation_function"][()].decode('utf-8'),
                   dropout_rate = float(group["dropout_rate"       ][()]))

class LSTM(Layer):
    """ A Long Short-Term Memory (LSTM) neural network layer

    Attributes
    ----------
    units : int
        Dimensionality of the output space
    activation : Activation
        Activation function to use
    recurrent_activation : Activation
        Activation function to use for the recurrent step
    recurrent_dropout : float
        Fraction of the units to drop for the linear transformation of the recurrent state
    """

    @property
    def units(self) -> int:
        return self._units

    @units.setter
    def units(self, units: int) -> None:
        assert units > 0
        self._units = units

    @property
    def activation(self) -> Activation:
        return self._activation

    @activation.setter
    def activation(self, activation: Activation) -> None:
        self._activation = activation

    @property
    def recurrent_activation(self) -> Activation:
        return self._recurrent_activation

    @recurrent_activation.setter
    def recurrent_activation(self, activation: Activation) -> None:
        self._recurrent_activation = activation

    @property
    def recurrent_dropout_rate(self) -> float:
        return self._recurrent_dropout_rate

    @recurrent_dropout_rate.setter
    def recurrent_dropout_rate(self, dropout_rate: float) -> None:
        assert 0.0 <= dropout_rate <= 1.0
        self._recurrent_dropout_rate = dropout_rate


    def __init__(self, units: int, activation: Activation, dropout_rate: float = 0., recurrent_activation: Activation = 'sigmoid', recurrent_dropout_rate: float = 0.):
        super().__init__(dropout_rate)
        self.units                  = units
        self.activation             = activation
        self.recurrent_activation   = recurrent_activation
        self.recurrent_dropout_rate = recurrent_dropout_rate

    def __eq__(self, other: Any) -> bool:
        if self is other:                                                           return True
        if not isinstance(other, LSTM):                                             return False
        if not(self.units == other.units):                                          return False
        if not(self.activation == other.activation):                                return False
        if not(self.recurrent_activation == other.recurrent_activation):            return False
        if not(isclose(self.recurrent_dropout_rate, other.recurrent_dropout_rate)): return False
        if not(isclose(self.dropout_rate, other.dropout_rate)):                     return False
        return True

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = tf.keras.layers.LSTM(units                = self.units,
                                 activation           = self.activation,
                                 recurrent_activation = self.recurrent_activation,
                                 recurrent_dropout    = self.recurrent_dropout_rate)(input_tensor)
        if self.dropout_rate > 0.0:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=self.dropout_rate))(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type'                          , data='LSTM', dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate'                  , data=self.dropout_rate)
        group.create_dataset('number_of_units'               , data=self.units)
        group.create_dataset('activation_function'           , data=self.activation,           dtype=h5py.string_dtype())
        group.create_dataset('recurrent_activation_function' , data=self.recurrent_activation, dtype=h5py.string_dtype())
        group.create_dataset('recurrent_dropout_rate'        , data=self.recurrent_dropout_rate)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> LSTM:
        return cls(units                  =   int(group["number_of_units"              ][()]),
                   activation             =       group["activation_function"          ][()].decode('utf-8'),
                   dropout_rate           = float(group["dropout_rate"                 ][()]),
                   recurrent_activation   =       group["recurrent_activation_function"][()].decode('utf-8'),
                   recurrent_dropout_rate = float(group["recurrent_dropout_rate"       ][()]))


class Conv2D(Layer):
    """ A 2D Convolutional Neural Network (CNN) layer

    Attributes
    ----------
    input_shape : Tuple[int, int]
        The height and width of the input data before convolution (i.e. its 2D shape)
    activation : Activation
        Activation function to use
    filters : int
        Number of convolution filters
    kernel_size : Tuple[int, int]
        Size of the convolution kernel
    strides : Tuple[int, int]
        Strides of the convolution
    padding : bool
        Whether or not padding should be applied to the convolution
    """

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: Tuple[int, int]) -> None:
        assert input_shape[0] > 0
        assert input_shape[1] > 0
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
        assert filters > 0
        self._filters = filters

    @property
    def kernel_size(self) -> Tuple[int, int]:
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size: Tuple[int, int]) -> None:
        assert kernel_size[0] > 0
        assert kernel_size[1] > 0
        self._kernel_size = kernel_size

    @property
    def strides(self) -> Tuple[int, int]:
        return self._strides

    @strides.setter
    def strides(self, strides: Tuple[int, int]) -> None:
        assert strides[0] > 0
        assert strides[1] > 0
        self._strides = strides

    @property
    def padding(self) -> bool:
        return self._padding

    @padding.setter
    def padding(self, padding: bool) -> None:
        self._padding = padding


    def __init__(self, input_shape: Tuple[int, int], activation = 'relu', filters: int = 1, kernel_size : Tuple[int, int] = (1, 1), strides : Tuple[int, int] = (1, 1), padding: bool = True, dropout_rate: float = 0.):
        super().__init__(dropout_rate)
        self.input_shape  = input_shape
        self.activation   = activation
        self.filters      = filters
        self.kernel_size  = kernel_size
        self.strides      = strides
        self.padding      = padding

    def __eq__(self, other: Any) -> bool:
        if self is other:                                       return True
        if not isinstance(other, Conv2D):                       return False
        if not(self.input_shape == other.input_shape):          return False
        if not(self.activation == other.activation):            return False
        if not(self.filters == other.filters):                  return False
        if not(self.kernel_size == other.kernel_size):          return False
        if not(self.strides == other.strides):                  return False
        if not(self.padding == other.padding):                  return False
        if not(isclose(self.dropout_rate, other.dropout_rate)): return False
        return True

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))


    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        assert input_tensor.shape[-1] % (self.input_shape[0] * self.input_shape[1]) == 0, "Input tensor shape is not divisible by the expected input 2D shape"

        number_of_channels = input_tensor.shape[-1] // (self.input_shape[0] * self.input_shape[1])
        input_shape = (-1, self.input_shape[0], self.input_shape[1], number_of_channels)
        x = tf.keras.layers.Reshape(target_shape=input_shape)(input_tensor)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters     = self.filters,
                                                                   kernel_size = self.kernel_size,
                                                                   strides     = self.strides,
                                                                   padding     = 'same' if self.padding else 'valid',
                                                                   activation  = self.activation))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        if self.dropout_rate > 0.0:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=self.dropout_rate))(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type'               , data='Conv2D', dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate'       , data=self.dropout_rate)
        group.create_dataset('input_shape'        , data=self.input_shape)
        group.create_dataset('activation_function', data=self.activation, dtype=h5py.string_dtype())
        group.create_dataset('number_of_filters'  , data=self.filters)
        group.create_dataset('kernel_size'        , data=self.kernel_size)
        group.create_dataset('strides'            , data=self.strides)
        group.create_dataset('padding'            , data=self.padding)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> Conv2D:
        return cls(input_shape            = tuple(int(x) for x in group['input_shape'  ][()]),
                   activation             =       group["activation_function"          ][()].decode('utf-8'),
                   dropout_rate           = float(group["dropout_rate"                 ][()]),
                   filters                =   int(group["number_of_filters"            ][()]),
                   kernel_size            = tuple(int(x) for x in group['kernel_size'  ][()]),
                   strides                = tuple(int(x) for x in group['strides'      ][()]),
                   padding                =  bool(group["padding"                      ][()]))


class MaxPool2D(Layer):
    """ A 2D Max Pool layer

    Attributes
    ----------
    input_shape : Tuple[int, int]
        The height and width of the input data before convolution (i.e. its 2D shape)
    pool_size : Tuple[int, int]
        Size of the pooling window
    strides : Tuple[int, int]
        Strides of the convolution
    padding : bool
        Whether or not padding should be applied to the convolution
    """

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: Tuple[int, int]) -> None:
        assert input_shape[0] > 0
        assert input_shape[1] > 0
        self._input_shape = input_shape

    @property
    def pool_size(self) -> Tuple[int, int]:
        return self._pool_size

    @pool_size.setter
    def pool_size(self, pool_size: Tuple[int, int]) -> None:
        assert pool_size[0] > 0
        assert pool_size[1] > 0
        self._pool_size = pool_size

    @property
    def strides(self) -> Tuple[int, int]:
        return self._strides

    @strides.setter
    def strides(self, strides: Tuple[int, int]) -> None:
        assert strides[0] > 0
        assert strides[1] > 0
        self._strides = strides

    @property
    def padding(self) -> bool:
        return self._padding

    @padding.setter
    def padding(self, padding: bool) -> None:
        self._padding = padding


    def __init__(self, input_shape: Tuple[int, int], pool_size : Tuple[int, int] = (1, 1), strides : Tuple[int, int] = (1, 1), padding: bool = True, dropout_rate: float = 0.):
        super().__init__(dropout_rate)
        self.input_shape = input_shape
        self.pool_size   = pool_size
        self.strides     = strides
        self.padding     = padding

    def __eq__(self, other: Any) -> bool:
        if self is other:                                       return True
        if not isinstance(other, MaxPool2D):                    return False
        if not(self.input_shape == other.input_shape):          return False
        if not(self.pool_size == other.pool_size):              return False
        if not(self.strides == other.strides):                  return False
        if not(self.padding == other.padding):                  return False
        if not(isclose(self.dropout_rate, other.dropout_rate)): return False
        return True

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))


    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        assert input_tensor.shape[-1] % (self.input_shape[0] * self.input_shape[1]) == 0, "Input tensor shape is not divisible by the expected input 2D shape"

        number_of_channels = input_tensor.shape[-1] // (self.input_shape[0] * self.input_shape[1])
        input_shape = (-1, self.input_shape[0], self.input_shape[1], number_of_channels)
        x = tf.keras.layers.Reshape(target_shape=input_shape)(input_tensor)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size   = self.pool_size,
                                                                         strides     = self.strides,
                                                                         padding     = 'same' if self.padding else 'valid'))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        if self.dropout_rate > 0.0:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=self.dropout_rate))(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type'        , data='MaxPool2D', dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate', data=self.dropout_rate)
        group.create_dataset('input_shape' , data=self.input_shape)
        group.create_dataset('pool_size'   , data=self.pool_size)
        group.create_dataset('strides'     , data=self.strides)
        group.create_dataset('padding'     , data=self.padding)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> MaxPool2D:
        return cls(input_shape  = tuple(int(x) for x in group['input_shape'][()]),
                   dropout_rate = float(group["dropout_rate"               ][()]),
                   pool_size    = tuple(int(x) for x in group['pool_size'  ][()]),
                   strides      = tuple(int(x) for x in group['strides'    ][()]),
                   padding      =  bool(group["padding"                    ][()]))

#class Transformer(Layer):
#    """
#    """
#



class NNStrategy(PredictionStrategy):
    """ A concrete class for a Neural-Network-based prediction strategy

    This prediction strategy is only intended for use with static State-Points, meaning
    non-temporal series, or said another way, State Series with series lengths of one.

    Attributes
    ----------
    layers : LayerSequence
        The hidden layers of the neural network
    initial_learning_rate : float
        The initial learning rate of the training
    learning_decay_rate : float
        The decay rate of the learning using Exponential Decay
    epoch_limit : int
        The limit on the number of training epochs conducted during training
    convergence_criteria : float
        The convergence criteria for training
    batch_size : int
        The training batch sizes
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layer_sequence.layers

    @layers.setter
    def layers(self, layers: List[Layer]):
        self._layer_sequence = LayerSequence(layers)

    @property
    def initial_learning_rate(self) -> float:
        return self._initial_learning_rate

    @initial_learning_rate.setter
    def initial_learning_rate(self, initial_learning_rate: float):
        assert initial_learning_rate >= 0.
        self._initial_learning_rate = initial_learning_rate

    @property
    def learning_decay_rate(self) -> float:
        return self._learning_decay_rate

    @learning_decay_rate.setter
    def learning_decay_rate(self, learning_decay_rate: float):
        assert 0. <= learning_decay_rate <= 1.
        self._learning_decay_rate = learning_decay_rate

    @property
    def epoch_limit(self) -> int:
        return self._epoch_limit

    @epoch_limit.setter
    def epoch_limit(self, epoch_limit: int):
        assert epoch_limit > 0
        self._epoch_limit = epoch_limit

    @property
    def convergence_criteria(self) -> float:
        return self._convergence_criteria

    @convergence_criteria.setter
    def convergence_criteria(self, convergence_criteria: float):
        assert convergence_criteria > 0.
        self._convergence_criteria = convergence_criteria

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        assert batch_size > 0
        self._batch_size = batch_size

    @property
    def isTrained(self) -> bool:
        return self._model is not None


    def __init__(self,
                 input_features        : Dict[str, FeatureProcessor],
                 predicted_feature     : str,
                 layers                : List[Layer]=[Dense(units=5, activation='relu')],
                 initial_learning_rate : float=0.01,
                 learning_decay_rate   : float=1.,
                 epoch_limit           : int=400,
                 convergence_criteria  : float=1E-14,
                 batch_size            : int=32) -> None:

        super().__init__()
        self.input_features         = input_features
        self.predicted_feature      = predicted_feature
        self.layers                 = layers
        self.initial_learning_rate  = initial_learning_rate
        self.learning_decay_rate    = learning_decay_rate
        self.epoch_limit            = epoch_limit
        self.convergence_criteria   = convergence_criteria
        self.batch_size             = batch_size

        self._model = None


    def train(self, train_data: List[StateSeries], test_data: List[StateSeries] = [], num_procs: int = 1) -> None:
        assert len(test_data) == 0  # This model does not use Test Data as part of training
        assert all(len(series) == len(train_data[0]) for series in train_data)

        X = self.preprocess_inputs(train_data, num_procs)
        y = self._get_targets(train_data)

        input_tensor = tf.keras.layers.Input(shape=(None, len(X[0][0])))

        x = self._layer_sequence.build(input_tensor)
        output = tf.keras.layers.Dense(y.shape[2])(x)

        self._model = tf.keras.Model(inputs=input_tensor, outputs=output)

        learning_rate_schedule = ExponentialDecay(self.initial_learning_rate, decay_steps=self.epoch_limit, decay_rate=self.learning_decay_rate, staircase=True)
        self._model.compile(optimizer=Adam(learning_rate=learning_rate_schedule), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

        early_stop = EarlyStopping(monitor='mean_absolute_error', min_delta=self.convergence_criteria, patience=5, verbose=1, mode='auto', restore_best_weights=True)
        dataset    = tf.data.Dataset.from_tensor_slices((X, y))
        dataset    = dataset.batch(self.batch_size, drop_remainder=True)
        self._model.fit(dataset, epochs=self.epoch_limit, batch_size=self.batch_size, callbacks=[early_stop])


    def _predict_one(self, state_series: StateSeries) -> np.ndarray:
        return self._predict_all([state_series])[0]


    def _predict_all(self, state_series: List[StateSeries]) -> np.ndarray:
        assert(self.isTrained)

        X = self.preprocess_inputs(state_series)
        tf.convert_to_tensor(X, dtype=tf.float32)
        y = self._model.predict(X).flatten()
        return y


    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        This method handles everything via the HDF5 file format, so any file name specified
        that doesn't end with '.h5' will be assumed to end with '.h5'

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """

        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"

        self._model.save(file_name)

        with h5py.File(file_name, 'a') as h5_file:
            self.base_save_model(h5_file)
            h5_file.create_dataset('initial_learning_rate', data=self.initial_learning_rate)
            h5_file.create_dataset('learning_decay_rate',   data=self.learning_decay_rate)
            h5_file.create_dataset('epoch_limit',           data=self.epoch_limit)
            h5_file.create_dataset('convergence_criteria',  data=self.convergence_criteria)
            h5_file.create_dataset('batch_size',            data=self.batch_size)
            self._layer_sequence.save(h5_file.create_group('neural_network'))



    def load_model(self, file_name: str) -> None:
        """ A method for loading a trained model

        This method handles everything via the HDF5 file format, so any file name specified
        that doesn't end with '.h5' will be assumed to end with '.h5'

        Parameters
        ----------
        file_name : str
            The name of the file to load the model from
        """

        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"

        assert(os.path.exists(file_name))

        with h5py.File(file_name, 'r') as h5_file:
            self.base_load_model(h5_file)
            self.initial_learning_rate = float( h5_file['initial_learning_rate'][()] )
            self.learning_decay_rate   = float( h5_file['learning_decay_rate'][()]   )
            self.epoch_limit           = int(   h5_file['epoch_limit'][()]           )
            self.convergence_criteria  = float( h5_file['convergence_criteria'][()]  )
            self.batch_size            = int(   h5_file['batch_size'][()]            )
            self._layer_sequence       = LayerSequence.from_h5(h5_file['neural_network'])

        self._model = load_model(file_name)


    @classmethod
    def read_from_hdf5(cls: NNStrategy, file_name: str) -> NNStrategy:
        """ A basic factory method for building NN Strategy from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the file from which to read and build the model

        Returns
        -------
        NNStrategy:
            The model from the hdf5 file
        """
        assert(os.path.exists(file_name))

        new_model = cls({}, None)
        new_model.load_model(file_name)

        return new_model