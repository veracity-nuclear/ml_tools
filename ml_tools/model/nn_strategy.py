from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Literal, Type, Tuple, Optional, Union, Any
import os
from math import isclose
from decimal import Decimal
import h5py
import numpy as np

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module
import tensorflow as tf
from tensorflow.keras import KerasTensor
import tensorflow.keras.layers
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

from ml_tools.model.state import StateSeries
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import FeatureProcessor


LayerType  = Literal['Dense', 'PassThrough', 'LSTM', 'LayerSequence', 'CompoundLayer']
Activation = Literal['elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'mish',
                     'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh']

# Pylint mistakenly interpretting layer_group["activation_function"][()] as an HDF5 Group
# and subsequently complaining that it has no "decode" member
# pylint: disable=no-member

class Layer(ABC):
    """ Abstract class for neural network layers. Not meant to be instantiated directly.

    Parameters
    ----------
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout

    Attributes
    ----------
    dropout_rate : float
        Dropout rate for the layer.
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout
    """

    _registry: Dict[str, Type[Layer]] = {} # Registry for child classes

    @classmethod
    def register_subclass(cls, layer_type: str) -> None:
        """ Method for registering a subclass for a specific layer type

        Parameters
        ----------
        layer_type : str
            The string corresponding to the layer type to be registered
        """
        def decorator(subclass: Type[Layer]):
            cls._registry[layer_type] = subclass
            return subclass
        return decorator

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate: float) -> None:
        assert 0.0 <= dropout_rate <= 1.0, f"dropout rate = {dropout_rate}"
        self._dropout_rate = dropout_rate

    @property
    def batch_normalize(self) -> bool:
        return self._batch_normalize

    @batch_normalize.setter
    def batch_normalize(self, batch_normalize: bool) -> None:
        self._batch_normalize = batch_normalize

    @property
    def layer_normalize(self) -> bool:
        return self._layer_normalize

    @layer_normalize.setter
    def layer_normalize(self, layer_normalize: bool) -> None:
        self._layer_normalize = layer_normalize

    def __init__(self,
                 dropout_rate:     float = 0.0,
                 batch_normalize:  bool = False,
                 layer_normalize:  bool = False) -> None:
        self.dropout_rate     = dropout_rate
        self.batch_normalize  = batch_normalize
        self.layer_normalize  = layer_normalize

    @abstractmethod
    def __eq__(self, other: Layer) -> bool:
        """ Compare two layers for equality

        Parameters
        ----------
        other: Layer
            The other Layer to compare against

        Returns
        -------
        bool
            True if self and other are equal within the tolerance.  False otherwise

        Notes
        -----
        The relative tolerance is 1e-9 for float comparisons
        """

    @abstractmethod
    def __hash__(self) -> int:
        """ Generate a hash key for the layer

        Returns
        -------
        int
            The hash key corresponding to this layer

        Notes
        -----
        Hash generation is consistent with the 1e-9 float comparison equality relative tolerance
        """

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        """ Method for constructing the layer

        Parameters
        ----------
        input_tensor : KerasTensor
            The input tensor for the layer
        """
        x = self._build(input_tensor)

        if self.batch_normalize:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)

        if self.layer_normalize:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(x)

        if self.dropout_rate > 0.:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=self.dropout_rate))(x)

        return x

    @abstractmethod
    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        """ Method for constructing the layer without any dropout or normalization

        Parameters
        ----------
        input_tensor : KerasTensor
            The input tensor for the layer
        """

    @abstractmethod
    def save(self, group: h5py.Group) -> None:
        """ Method for saving the layer to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            The h5py group to save the layer to
        """

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

    @staticmethod
    def layers_from_h5(group: "h5py.Group") -> List[Layer]:
        """ Create a list of Layers from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group containing layer data.

        Returns
        -------
        List[Layer]
            A list of Layer instances created from the HDF5 group.
        """
        layers = []
        layer_names = [key for key in group.keys() if key.startswith("layer_")]
        layer_names = sorted(layer_names, key=lambda x: int(x.split('_')[1]))

        for layer_name in layer_names:
            layer_group = group[layer_name]
            layer_type = layer_group['type'][()].decode('utf-8')
            if layer_type not in Layer._registry:
                raise ValueError(f"Unknown layer type: {layer_type}")
            layers.append(Layer._registry[layer_type].from_h5(layer_group))

        return layers


@Layer.register_subclass("LayerSequence")
class LayerSequence(Layer):
    """ A class for a sequence of layers

    A layer sequence does not require dropout rate or normalization,
    specifications as these will this will be dictated by the final
    layer's specifications. If any of these specifications are provided,
    they will be ignored in favor of the final layer specifications.

    Parameters
    ----------
    layers : List[Layer]
        The list of layers that comprise the sequence

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
        assert len(value) > 0, f"len(value) = {len(value)}"
        self._layers = value

    def __init__(self, layers: List[Layer]) -> None:
        super().__init__(0.0, False, False)
        self.layers = layers

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, LayerSequence) and
                  len(self.layers) == len(other.layers) and
                  all(s_layer == o_layer for s_layer, o_layer in zip(self.layers, other.layers)))
               )

    def __hash__(self) -> int:
        return hash(tuple(self.layers))

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
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
        layers = Layer.layers_from_h5(group)
        return cls(layers=layers)


@Layer.register_subclass("CompoundLayer")
class CompoundLayer(Layer):
    """ A class for compound / composite layers consisting layers that are executed in parallel

    This class effectively splits the input into the layer across multiple layers which will
    each execute in parallel and then merge their output at the end.

    A compound layer does require a dropout rate on account of the merged outputs.  If any
    of the composite layers are provided a drop out rate, said rate will be ignored in favor
    of the compound layer's dropout rate.

    Also, input features need not be "exclusive" to a given layer, but rather may be used by multiple
    constituent layers.

    Parameters
    ----------
    layers : List[Layer]
        The list of constituent layers that will be executed in parallel
    input_specifications : List[Union[slice, List[int]]]
        The input indices each layer should use to pull from the incoming input.
        This may be provided either as a list or a slice.  If a slice is provided
        the end index must be explicitly stated and cannot be a negative value.
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout

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

    def __init__(self,
                 layers:               List[Layer],
                 input_specifications: List[Union[slice, List[int]]],
                 dropout_rate:         float = 0.0,
                 batch_normalize:      bool = False,
                 layer_normalize:      bool = False) -> None:

        super().__init__(dropout_rate, batch_normalize, layer_normalize)

        assert len(layers) > 0, f"len(layers) = {len(layers)}"
        assert len(layers) == len(input_specifications), \
            f"len(layers) = {len(layers)}, len(input_specifications) = {len(input_specifications)}"

         # Input layer length is not known until at build
        assert all(not(spec.stop is None) and spec.stop >= 0
                   for spec in input_specifications if isinstance(spec, slice)), \
                "Input specification slices must have explicit non-negative ending indeces"

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
        return (self is other or
                 (isinstance(other, CompoundLayer) and
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                  self.batch_normalize == other.batch_normalize and
                  self.layer_normalize == other.layer_normalize and
                  len(self.layers) == len(other.layers) and
                  all(s_layer == o_layer for s_layer, o_layer in zip(self.layers, other.layers)))
               )

    def __hash__(self) -> int:
        return hash(tuple(self.layers),
                    tuple(tuple(specification) for specification in self.input_specifications),
                    Decimal(self.dropout_rate).quantize(Decimal('1e-9')),
                    self.batch_normalize,
                    self.layer_normalize)

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        @register_keras_serializable()
        def gather_indices(x, indices):
            return tf.gather(x, indices, axis=-1)

        assert all(index < input_tensor.shape[2] for spec in self.input_specifications for index in spec), \
            "input specification index greater than input feature vector length"
        split_inputs = [tensorflow.keras.layers.TimeDistributed(
                        tensorflow.keras.layers.Lambda(gather_indices, arguments={'indices': indices}))(input_tensor)
                        for indices in self.input_specifications]

        outputs = [layer.build(split) for layer, split in zip(self._layers, split_inputs)]
        x = tensorflow.keras.layers.Concatenate(axis=-1)(outputs)
        return x


    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type', data='CompoundLayer', dtype=h5py.string_dtype())
        specs_array = np.array([np.array(inner, dtype=np.int32) for inner in self.input_specifications], dtype=object)
        group.create_dataset('input_specifications', data=specs_array, dtype=h5py.vlen_dtype(np.int32))
        for i, layer in enumerate(self.layers):
            layer_group = group.create_group('layer_' + str(i))
            layer.save(layer_group)
        group.attrs['dropout_rate'] = self._dropout_rate
        group.attrs['batch_normalize'] = self._batch_normalize
        group.attrs['layer_normalize'] = self._layer_normalize


    @classmethod
    def from_h5(cls, group: h5py.Group) -> CompoundLayer:
        input_specifications = [list(item) for item in group['input_specifications'][:]]
        dropout_rate         = group.attrs.get('dropout_rate', 0.0)
        batch_normalize      = group.attrs.get('batch_normalize', False)
        layer_normalize      = group.attrs.get('layer_normalize', False)
        layers               = Layer.layers_from_h5(group)
        return cls(layers               = layers,
                   input_specifications = input_specifications,
                   dropout_rate         = dropout_rate,
                   batch_normalize      = batch_normalize,
                   layer_normalize      = layer_normalize)


@Layer.register_subclass("PassThrough")
class PassThrough(Layer):
    """ A layer for passing through input features

    Parameters
    ----------
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout

    This layer type is useful when constructing composite layers that require passing some features
    straight through to the next layer while other features pass through an actual processing layer.
    """

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, PassThrough) and
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                  self.batch_normalize == other.batch_normalize and
                  self.layer_normalize == other.layer_normalize)
               )

    def __hash__(self) -> int:
        return hash(Decimal(self.dropout_rate).quantize(Decimal('1e-9')),
                    self.batch_normalize,
                    self.layer_normalize)

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        return input_tensor

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type'        ,     data='PassThrough', dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate',     data=self.dropout_rate)
        group.create_dataset('batch_normalize',  data=self.batch_normalize)
        group.create_dataset('layer_normalize',  data=self.layer_normalize)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> PassThrough:
        return cls(dropout_rate     = float(group["dropout_rate"    ][()]),
                   batch_normalize  =  bool(group["batch_normalize" ][()]),
                   layer_normalize  =  bool(group["layer_normalize" ][()]))


@Layer.register_subclass("Dense")
class Dense(Layer):
    """ A Dense Neural Network Layer

    Parameters
    ----------
    units : int
        Number of nodes (i.e. neurons) to use in the dense layer
    activation : str
        Activation function to use
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout

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
        assert units > 0, f"units = {units}"
        self._units = units

    @property
    def activation(self) -> Activation:
        return self._activation

    @activation.setter
    def activation(self, activation: Activation) -> None:
        self._activation = activation


    def __init__(self,
                 units:            int,
                 activation:       Activation,
                 dropout_rate:     float = 0.,
                 batch_normalize:  bool = False,
                 layer_normalize:  bool = False):
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.units      = units
        self.activation = activation

    def __eq__(self, other: Layer) -> bool:
        return (self is other or
                (isinstance(other, Dense) and
                 self.units == other.units and
                 self.activation == other.activation and
                 isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                 self.batch_normalize == other.batch_normalize and
                 self.layer_normalize == other.layer_normalize)
               )

    def __hash__(self) -> int:
        return hash(tuple(self.units,
                          self.activation,
                          Decimal(self.dropout_rate).quantize(Decimal('1e-9'))),
                          self.batch_normalize,
                          self.layer_normalize
                   )

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.units, activation=self.activation))(input_tensor)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type',                 data='Dense', dtype=h5py.string_dtype())
        group.create_dataset('number_of_units',      data=self.units)
        group.create_dataset('activation_function',  data=self.activation, dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate',         data=self.dropout_rate)
        group.create_dataset('batch_normalize',      data=self.batch_normalize)
        group.create_dataset('layer_normalize',      data=self.layer_normalize)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> Dense:
        return cls(units            =   int(group["number_of_units"    ][()]),
                   activation       =       group["activation_function"][()].decode('utf-8'),
                   dropout_rate     = float(group["dropout_rate"       ][()]),
                   batch_normalize  =  bool(group["batch_normalize"    ][()]),
                   layer_normalize  =  bool(group["layer_normalize"    ][()]))


@Layer.register_subclass("LSTM")
class LSTM(Layer):
    """ A Long Short-Term Memory (LSTM) neural network layer

    Parameters
    ----------
    units : int
        Dimensionality of the output space
    activation : Activation
        Activation function to use
    recurrent_activation : Activation
        Activation function to use for the recurrent step
    recurrent_dropout : float
        Fraction of the units to drop for the linear transformation of the recurrent state
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout

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
        assert units > 0, f"units = {units}"
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
        assert 0.0 <= dropout_rate <= 1.0, f"dropout rate = {dropout_rate}"
        self._recurrent_dropout_rate = dropout_rate


    def __init__(self,
                 units:                  int,
                 activation:             Activation,
                 recurrent_activation:   Activation = 'sigmoid',
                 recurrent_dropout_rate: float = 0.,
                 dropout_rate:           float = 0.,
                 batch_normalize:        bool = False,
                 layer_normalize:        bool = False):
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.units                  = units
        self.activation             = activation
        self.recurrent_activation   = recurrent_activation
        self.recurrent_dropout_rate = recurrent_dropout_rate

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, LSTM) and
                  self.units == other.units and
                  self.activation == other.activation and
                  self.recurrent_activation == other.recurrent_activation and
                  isclose(self.recurrent_dropout_rate, other.recurrent_dropout_rate, rel_tol=1e-9) and
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                  self.batch_normalize == other.batch_normalize and
                  self.layer_normalize == other.layer_normalize)
        )


    def __hash__(self) -> int:
        return hash(tuple(self.units,
                          self.activation,
                          self.recurrent_activation,
                          Decimal(self.recurrent_dropout_rate).quantize(Decimal('1e-9')),
                          Decimal(self.dropout_rate).quantize(Decimal('1e-9'))),
                          self.batch_normalize,
                          self.layer_normalize
                   )

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = tf.keras.layers.LSTM(units                = self.units,
                                 activation           = self.activation,
                                 return_sequences     = True,
                                 recurrent_activation = self.recurrent_activation,
                                 recurrent_dropout    = self.recurrent_dropout_rate)(input_tensor)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type',                          data='LSTM', dtype=h5py.string_dtype())
        group.create_dataset('number_of_units',               data=self.units)
        group.create_dataset('activation_function',           data=self.activation,           dtype=h5py.string_dtype())
        group.create_dataset('recurrent_activation_function', data=self.recurrent_activation, dtype=h5py.string_dtype())
        group.create_dataset('recurrent_dropout_rate',        data=self.recurrent_dropout_rate)
        group.create_dataset('dropout_rate',                  data=self.dropout_rate)
        group.create_dataset('batch_normalize',               data=self.batch_normalize)
        group.create_dataset('layer_normalize',               data=self.layer_normalize)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> LSTM:
        return cls(units                  =   int(group["number_of_units"              ][()]),
                   activation             =       group["activation_function"          ][()].decode('utf-8'),
                   recurrent_activation   =       group["recurrent_activation_function"][()].decode('utf-8'),
                   recurrent_dropout_rate = float(group["recurrent_dropout_rate"       ][()]),
                   dropout_rate           = float(group["dropout_rate"                 ][()]),
                   batch_normalize        =  bool(group["batch_normalize"              ][()]),
                   layer_normalize        =  bool(group["layer_normalize"              ][()]))


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

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        assert input_tensor.shape[-1] % (self.input_shape[0] * self.input_shape[1] * self.input_shape[2]) == 0, \
            "Input tensor shape is not divisible by the expected input 3D shape"

        number_of_channels = input_tensor.shape[-1] // (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        input_shape = (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2], number_of_channels)
        x = tf.keras.layers.Reshape(target_shape=input_shape)(input_tensor)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(filters     = self.filters,
                                                                   kernel_size = self.kernel_size,
                                                                   strides     = self.strides,
                                                                   padding     = 'same' if self.padding else 'valid',
                                                                   activation  = self.activation))(x)
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

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        assert input_tensor.shape[-1] % (self.input_shape[0] * self.input_shape[1] * self.input_shape[2]) == 0, \
            "Input tensor shape is not divisible by the expected input 3D shape"

        number_of_channels = input_tensor.shape[-1] // (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        input_shape = (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2], number_of_channels)
        x = tf.keras.layers.Reshape(target_shape=input_shape)(input_tensor)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling3D(pool_size = self.pool_size,
                                                                         strides   = self.strides,
                                                                         padding   = 'same' if self.padding else 'valid'))(x)
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


@Layer.register_subclass("Transformer")
class Transformer(Layer):
    """ A transformer layer

    Parameters
    ----------
    num_heads : int
        The number of attention heads
    model_dim : int
        The model dimensionality
    ff_dim : int
        The feed-forward network dimensionality
    activation : Activation
        Activation function to use for the Feed Forward Network of the Transformer
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout

    Attributes
    ----------
    num_heads : int
        The number of attention heads
    model_dim : int
        The model dimensionality
    ff_dim : int
        The feed-forward network dimensionality
    activation : Activation
        Activation function to use for the Feed Forward Network of the Transformer
    """

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @num_heads.setter
    def num_heads(self, num_heads: int) -> None:
        assert num_heads > 0, f"num_heads = {num_heads}"
        self._num_heads = num_heads

    @property
    def model_dim(self) -> int:
        return self._model_dim

    @model_dim.setter
    def model_dim(self, model_dim: int) -> None:
        assert model_dim > 0, f"model_dim = {model_dim}"
        self._model_dim = model_dim

    @property
    def ff_dim(self) -> int:
        return self._ff_dim

    @ff_dim.setter
    def ff_dim(self, ff_dim: int) -> None:
        assert ff_dim > 0, f"ff_dim = {ff_dim}"
        self._ff_dim = ff_dim

    @property
    def activation(self) -> Activation:
        return self._activation

    @activation.setter
    def activation(self, activation: Activation) -> None:
        self._activation = activation


    def __init__(self,
                 num_heads:        int,
                 model_dim:        int,
                 ff_dim:           int,
                 activation:       Activation = 'relu',
                 dropout_rate:     float = 0.,
                 batch_normalize:  bool = False,
                 layer_normalize:  bool = False):
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.num_heads        = num_heads
        self.model_dim        = model_dim
        self.ff_dim           = ff_dim
        self.activation       = activation

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, Transformer) and
                  self.num_heads        == other.num_heads and
                  self.model_dim        == other.model_dim and
                  self.ff_dim           == other.ff_dim and
                  self.activation       == other.activation and
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                  self.batch_normalize  == other.batch_normalize and
                  self.layer_normalize  == other.layer_normalize)
        )

    def __hash__(self) -> int:
        return hash(tuple(self.num_heads,
                          self.model_dim,
                          self.ff_dim,
                          Decimal(self.dropout_rate).quantize(Decimal('1e-9'))),
                          self.batch_normalize,
                          self.layer_normalize,
                          self.activation
                   )

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        # Project input_tensor to model dimensions if they are not the same
        input_tensor = tf.keras.layers.Dense(self.model_dim)(input_tensor) \
                       if input_tensor.shape[-1] != self.model_dim else input_tensor

        attention = tf.keras.layers.MultiHeadAttention(num_heads = self.num_heads,
                                                       key_dim   = self.model_dim)(input_tensor, input_tensor)
        attention = tf.keras.layers.Dropout(rate=self.dropout_rate)(attention) if self.dropout_rate > 0. else attention
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + input_tensor)

        feedfoward = tf.keras.layers.Dense(self.ff_dim, activation=self.activation)(attention)
        feedfoward = tf.keras.layers.Dense(self.model_dim)(feedfoward)
        feedfoward = tf.keras.layers.Dropout(rate=self.dropout_rate)(feedfoward) if self.dropout_rate > 0. else feedfoward

        return feedfoward + attention


    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type',                    data='Transformer', dtype=h5py.string_dtype())
        group.create_dataset('number_of_heads',         data=self.num_heads)
        group.create_dataset('model_dimensions',        data=self.model_dim)
        group.create_dataset('feed_forward_dimensions', data=self.ff_dim)
        group.create_dataset('activation_function',     data=self.activation, dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate',            data=self.dropout_rate)
        group.create_dataset('batch_normalize',         data=self.batch_normalize)
        group.create_dataset('layer_normalize',         data=self.layer_normalize)


    @classmethod
    def from_h5(cls, group: h5py.Group) -> Transformer:
        return cls(num_heads        =   int(group["number_of_heads"        ][()]),
                   model_dim        =   int(group["model_dimensions"       ][()]),
                   ff_dim           =   int(group["feed_forward_dimensions"][()]),
                   activation       =       group["activation_function"    ][()].decode('utf-8'),
                   dropout_rate     = float(group["dropout_rate"           ][()]),
                   batch_normalize  =  bool(group["batch_normalize"        ][()]),
                   layer_normalize  =  bool(group["layer_normalize"        ][()]))


class NNStrategy(PredictionStrategy):
    """ A concrete class for a Neural-Network-based prediction strategy

    This prediction strategy is only intended for use with static State-Points, meaning
    non-temporal series, or said another way, State Series with series lengths of one.

    Parameters
    ----------
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding feature processing strategy
    predicted_feature : str
        The string specifying the feature to be predicted
    layers : List[Layer]
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
        The convergence precision criteria for training
    convergence_patience : int
        Number of epochs with no improvement (i.e. error improves by greater than the convergence_criteria)
        after which training will be stopped
    batch_size : int
        The training batch sizes
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layer_sequence.layers

    @layers.setter
    def layers(self, layers: List[Layer]):
        assert len(layers) > 0, f"len(layers) = {len(layers)}"
        self._layer_sequence = LayerSequence(layers)

    @property
    def initial_learning_rate(self) -> float:
        return self._initial_learning_rate

    @initial_learning_rate.setter
    def initial_learning_rate(self, initial_learning_rate: float):
        assert initial_learning_rate >= 0., f"initial_learning_rate = {initial_learning_rate}"
        self._initial_learning_rate = initial_learning_rate

    @property
    def learning_decay_rate(self) -> float:
        return self._learning_decay_rate

    @learning_decay_rate.setter
    def learning_decay_rate(self, learning_decay_rate: float):
        assert 0. <= learning_decay_rate <= 1., f"learning_decay_rate = {learning_decay_rate}"
        self._learning_decay_rate = learning_decay_rate

    @property
    def epoch_limit(self) -> int:
        return self._epoch_limit

    @epoch_limit.setter
    def epoch_limit(self, epoch_limit: int):
        assert epoch_limit > 0, f"epoch_limit = {epoch_limit}"
        self._epoch_limit = epoch_limit

    @property
    def convergence_criteria(self) -> float:
        return self._convergence_criteria

    @convergence_criteria.setter
    def convergence_criteria(self, convergence_criteria: float):
        assert convergence_criteria > 0., f"convergence_criteria = {convergence_criteria}"
        self._convergence_criteria = convergence_criteria

    @property
    def convergence_patience(self) -> int:
        return self._convergence_patience

    @convergence_patience.setter
    def convergence_patience(self, convergence_patience: int):
        assert convergence_patience > 0, f"convergence_patience = {convergence_patience}"
        self._convergence_patience = convergence_patience

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        assert batch_size > 0, f"batch_size = {batch_size}"
        self._batch_size = batch_size

    @property
    def isTrained(self) -> bool:
        return self._model is not None


    def __init__(self,
                 input_features        : Dict[str, FeatureProcessor],
                 predicted_feature     : str,
                 layers                : List[Layer]=None,
                 initial_learning_rate : float=0.01,
                 learning_decay_rate   : float=1.,
                 epoch_limit           : int=400,
                 convergence_criteria  : float=1E-14,
                 convergence_patience  : int=100,
                 batch_size            : int=32) -> None:

        super().__init__()
        self.input_features         = input_features
        self.predicted_feature      = predicted_feature
        self.layers                 = [Dense(units=5, activation='relu')] if layers is None else layers
        self.initial_learning_rate  = initial_learning_rate
        self.learning_decay_rate    = learning_decay_rate
        self.epoch_limit            = epoch_limit
        self.convergence_criteria   = convergence_criteria
        self.convergence_patience   = convergence_patience
        self.batch_size             = batch_size

        self._model = None


    def train(self, train_data: List[StateSeries], test_data: Optional[List[StateSeries]] = None, num_procs: int = 1) -> None:
        assert test_data is None, "The Neural Network Prediction Strategy does not use test data"
        assert all(len(series) == len(train_data[0]) for series in train_data)

        X = self.preprocess_inputs(train_data, num_procs)
        y = self._get_targets(train_data)

        input_tensor = tf.keras.layers.Input(shape=(None, len(X[0][0])))

        x = self._layer_sequence.build(input_tensor)
        output = tf.keras.layers.Dense(y.shape[2])(x)

        self._model = tf.keras.Model(inputs=input_tensor, outputs=output)

        learning_rate_schedule = ExponentialDecay(initial_learning_rate = self.initial_learning_rate,
                                                  decay_steps           = self.epoch_limit,
                                                  decay_rate            = self.learning_decay_rate, staircase=True)

        self._model.compile(optimizer=Adam(learning_rate = learning_rate_schedule),
                                           loss          = MeanSquaredError(),
                                           metrics       = [MeanAbsoluteError()])

        early_stop = EarlyStopping(monitor              =  'mean_absolute_error',
                                   min_delta            = self.convergence_criteria,
                                   patience             = self.convergence_patience,
                                   verbose              = 1,
                                   mode                 = 'auto',
                                   restore_best_weights = True)
        dataset    = tf.data.Dataset.from_tensor_slices((X, y))
        dataset    = dataset.batch(self.batch_size, drop_remainder=True)
        self._model.fit(dataset, epochs=self.epoch_limit, batch_size=self.batch_size, callbacks=[early_stop])

    def _predict_one(self, state_series: StateSeries) -> List[np.ndarray]:
        return self._predict_all([state_series])[0]


    def _predict_all(self, state_series: List[StateSeries]) -> List[List[np.ndarray]]:
        assert self.isTrained

        X = self.preprocess_inputs(state_series)
        tf.convert_to_tensor(X, dtype=tf.float32)
        return [list(series) for series in self._model.predict(X)]


    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """

        self._model.save(file_name + ".keras")

        with h5py.File(file_name + ".h5", 'w') as h5_file:
            self.base_save_model(h5_file)
            h5_file.create_dataset('initial_learning_rate', data=self.initial_learning_rate)
            h5_file.create_dataset('learning_decay_rate',   data=self.learning_decay_rate)
            h5_file.create_dataset('epoch_limit',           data=self.epoch_limit)
            h5_file.create_dataset('convergence_criteria',  data=self.convergence_criteria)
            h5_file.create_dataset('batch_size',            data=self.batch_size)
            self._layer_sequence.save(h5_file.create_group('neural_network'))


    @classmethod
    def read_from_file(cls: NNStrategy, file_name: str) -> Type[NNStrategy]:
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

        new_model = cls({}, None)

        assert os.path.exists(file_name + ".keras"), f"file name = {file_name + '.keras'}"
        assert os.path.exists(file_name + ".h5"), f"file name = {file_name + '.h5'}"

        with h5py.File(file_name + ".h5", 'r') as h5_file:
            new_model.base_load_model(h5_file)
            new_model.initial_learning_rate = float( h5_file['initial_learning_rate'][()] )
            new_model.learning_decay_rate   = float( h5_file['learning_decay_rate'][()]   )
            new_model.epoch_limit           = int(   h5_file['epoch_limit'][()]           )
            new_model.convergence_criteria  = float( h5_file['convergence_criteria'][()]  )
            new_model.batch_size            = int(   h5_file['batch_size'][()]            )
            new_model._layer_sequence       = LayerSequence.from_h5(h5_file['neural_network'])

        new_model._model = load_model(file_name + ".keras")

        return new_model
