from __future__ import annotations
from math import isclose
from decimal import Decimal
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras import KerasTensor

from ml_tools.model.nn_strategy.layer import Layer, Activation

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
