from __future__ import annotations
from decimal import Decimal
from math import isclose
from typing import Any
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf

from ml_tools.model.nn_strategy.layer import Layer, Activation

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

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        x = tf.keras.layers.LSTM(units                = self.units,
                                 activation           = self.activation,
                                 return_sequences     = True,
                                 recurrent_activation = self.recurrent_activation,
                                 recurrent_dropout    = self.recurrent_dropout_rate)(input_tensor)

        if self.batch_normalize:
            x = tf.keras.layers.BatchNormalization()(x)

        if self.layer_normalize:
            x = tf.keras.layers.LayerNormalization()(x)

        if self.dropout_rate > 0.:
            x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

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

    @classmethod
    def from_dict(cls, data: dict) -> "LSTM":
        params = dict(data or {})
        if "units" not in params and "number_of_units" in params:
            params["units"] = params.pop("number_of_units")
        if "activation" not in params and "activation_function" in params:
            params["activation"] = params.pop("activation_function")
        if "recurrent_activation" not in params and "recurrent_activation_function" in params:
            params["recurrent_activation"] = params.pop("recurrent_activation_function")
        if "units" not in params and "neurons" in params:
            params["units"] = params.pop("neurons")
        if "dropout_rate" not in params and "dropout" in params:
            params["dropout_rate"] = params.pop("dropout")
        if "recurrent_dropout_rate" not in params and "recurrent_dropout" in params:
            params["recurrent_dropout_rate"] = params.pop("recurrent_dropout")
        return cls(**params)

    def to_dict(self) -> dict:
        return {"type":                          "LSTM",
                "number_of_units":               self.units,
                "activation_function":           self.activation,
                "recurrent_activation_function": self.recurrent_activation,
                "recurrent_dropout_rate":        self.recurrent_dropout_rate,
                "dropout_rate":                  self.dropout_rate,
                "batch_normalize":               self.batch_normalize,
                "layer_normalize":               self.layer_normalize}
