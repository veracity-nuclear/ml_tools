from __future__ import annotations
from typing import Any, List, Union
from math import isclose
from decimal import Decimal
import h5py
import numpy as np

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras import KerasTensor

from ml_tools.model.nn_strategy.layer import Layer, gather_indices


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
        assert all(index < input_tensor.shape[2] for spec in self.input_specifications for index in spec), \
            "input specification index greater than input feature vector length"
        split_inputs = [tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Lambda(gather_indices, arguments={'indices': indices}))(input_tensor)
                        for indices in self.input_specifications]

        outputs = [layer.build(split) for layer, split in zip(self._layers, split_inputs)]
        x = tf.keras.layers.Concatenate(axis=-1)(outputs)
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
