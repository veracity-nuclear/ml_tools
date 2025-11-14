from __future__ import annotations
from typing import Any
from math import isclose
from decimal import Decimal
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf

from ml_tools.model.nn_strategy.layer import Layer, Activation


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
                 dropout_rate:     float = 0.,):
        super().__init__(dropout_rate, batch_normalize=False, layer_normalize=False)
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
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9))
        )

    def __hash__(self) -> int:
        return hash(tuple(self.num_heads,
                          self.model_dim,
                          self.ff_dim,
                          Decimal(self.dropout_rate).quantize(Decimal('1e-9'))),
                          self.activation
                   )

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        # Project input to model_dim if needed
        if input_tensor.shape[-1] != self.model_dim:
            input_tensor = tf.keras.layers.Dense(self.model_dim)(input_tensor)

        # Multi-head self-attention
        attention = tf.keras.layers.MultiHeadAttention(num_heads = self.num_heads,
                                                       key_dim   = self.model_dim)(input_tensor, input_tensor)
        if self.dropout_rate > 0.:
            attention = tf.keras.layers.Dropout(self.dropout_rate)(attention)

        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input_tensor + attention)

        # Feedforward network (MLP block)
        feedforward = tf.keras.layers.Dense(self.ff_dim, activation=self.activation)(attention)

        if self.dropout_rate > 0.:
            feedforward = tf.keras.layers.Dropout(self.dropout_rate)(feedforward)

        feedforward = tf.keras.layers.Dense(self.model_dim)(feedforward)

        if self.dropout_rate > 0.:
            feedforward = tf.keras.layers.Dropout(self.dropout_rate)(feedforward)

        # Residual connection + LayerNorm (second norm)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + feedforward)

        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type',                    data='Transformer', dtype=h5py.string_dtype())
        group.create_dataset('number_of_heads',         data=self.num_heads)
        group.create_dataset('model_dimensions',        data=self.model_dim)
        group.create_dataset('feed_forward_dimensions', data=self.ff_dim)
        group.create_dataset('activation_function',     data=self.activation, dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate',            data=self.dropout_rate)


    @classmethod
    def from_h5(cls, group: h5py.Group) -> Transformer:
        return cls(num_heads        =   int(group["number_of_heads"        ][()]),
                   model_dim        =   int(group["model_dimensions"       ][()]),
                   ff_dim           =   int(group["feed_forward_dimensions"][()]),
                   activation       =       group["activation_function"    ][()].decode('utf-8'),
                   dropout_rate     = float(group["dropout_rate"           ][()]))

    @classmethod
    def from_dict(cls, data: dict) -> "Transformer":
        params = dict(data or {})
        if "dropout_rate" not in params and "dropout" in params:
            params["dropout_rate"] = params.pop("dropout")
        if "model_dim" not in params and "d_model" in params:
            params["model_dim"] = params.pop("d_model")
        if "ff_dim" not in params and "ffn_dim" in params:
            params["ff_dim"] = params.pop("ffn_dim")
        if "num_heads" not in params and "number_of_heads" in params:
            params["num_heads"] = params.pop("number_of_heads")
        if "model_dim" not in params and "model_dimensions" in params:
            params["model_dim"] = params.pop("model_dimensions")
        if "ff_dim" not in params and "feed_forward_dimensions" in params:
            params["ff_dim"] = params.pop("feed_forward_dimensions")
        if "activation" not in params and "activation_function" in params:
            params["activation"] = params.pop("activation_function")

        # Ignore normalization flags if present; Transformer does not use them
        if "batch_normalize" in params:
            params.pop("batch_normalize")
        if "layer_normalize" in params:
            params.pop("layer_normalize")
        return cls(**params)

    def to_dict(self) -> dict:
        return {"type":                    "Transformer",
                "number_of_heads":         self.num_heads,
                "model_dimensions":        self.model_dim,
                "feed_forward_dimensions": self.ff_dim,
                "activation_function":     self.activation,
                "dropout_rate":            self.dropout_rate}
