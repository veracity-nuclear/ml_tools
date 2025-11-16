from __future__ import annotations
from typing import Any
from math import isclose
from decimal import Decimal
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

from ml_tools.model.nn_strategy.layer import Layer, Activation


# Keras-serializable helpers used inside Lambda layers
@register_keras_serializable(package="ml_tools")
def positional_encoding_for_lambda(x: tf.Tensor, model_dim: int) -> tf.Tensor:
    """Compute sinusoidal positional encodings with sequence length inferred from input.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, seq_len, dim).
    model_dim : int
        Embedding dimension; last dimension of the model inputs.

    Returns
    -------
    tf.Tensor
        Positional encoding tensor of shape (1, seq_len, model_dim) broadcastable
        to the batch.
    """
    # pylint: disable=protected-access
    seq_len = tf.shape(x)[1]
    return Transformer._positional_encoding(seq_len, model_dim)


@register_keras_serializable(package="ml_tools")
def attention_mask_from_padding(inputs, use_causal: bool) -> tf.Tensor:
    """Combine padding and optional causal masking for attention.

    Parameters
    ----------
    inputs : Tuple[tf.Tensor, tf.Tensor]
        Tuple of (padding_mask, tensor) where padding_mask has shape (batch, seq_len)
        and tensor has shape (batch, seq_len, dim).
    use_causal : bool
        Whether to apply a causal lower-triangular mask in addition to padding.

    Returns
    -------
    tf.Tensor
        Attention mask of shape (batch, seq_len, seq_len).
    """
    # pylint: disable=protected-access
    padding_mask, tensor = inputs
    return Transformer._build_attention_mask(padding_mask, tensor, tf.constant(use_causal))


@register_keras_serializable(package="ml_tools")
def padding_mask_from_tensor(x: tf.Tensor) -> tf.Tensor:
    """Create a padding mask (True where non-zero timesteps).

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, seq_len, dim).

    Returns
    -------
    tf.Tensor
        Boolean mask of shape (batch, seq_len) marking non-zero timesteps.
    """
    return tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)


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
    use_causal_mask : bool, optional
        Whether to apply a causal (look-back-only) mask in self-attention. Default True.

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
    use_causal_mask : bool
        Whether causal masking is applied during attention.
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

    @property
    def use_causal_mask(self) -> bool:
        return self._use_causal_mask

    @use_causal_mask.setter
    def use_causal_mask(self, use_causal_mask: bool) -> None:
        self._use_causal_mask = bool(use_causal_mask)


    def __init__(self,
                 num_heads:        int,
                 model_dim:        int,
                 ff_dim:           int,
                 activation:       Activation = 'relu',
                 dropout_rate:     float = 0.,
                 use_causal_mask:  bool  = True):
        super().__init__(dropout_rate, batch_normalize=False, layer_normalize=False)
        self.num_heads        = num_heads
        self.model_dim        = model_dim
        self.ff_dim           = ff_dim
        self.activation       = activation
        self.use_causal_mask  = use_causal_mask

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, Transformer) and
                  self.num_heads        == other.num_heads and
                  self.model_dim        == other.model_dim and
                  self.ff_dim           == other.ff_dim and
                  self.activation       == other.activation and
                  self.use_causal_mask  == other.use_causal_mask and
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9))
        )

    def __hash__(self) -> int:
        return hash(tuple(self.num_heads,
                          self.model_dim,
                          self.ff_dim,
                          Decimal(self.dropout_rate).quantize(Decimal('1e-9'))),
                          self.activation,
                          self.use_causal_mask
                   )

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        original_input = input_tensor

        # Project input to model_dim if needed
        if input_tensor.shape[-1] != self.model_dim:
            input_tensor = tf.keras.layers.Dense(self.model_dim)(input_tensor)

        # Positional encoding (sin/cos) added to inputs
        pos_encoding = tf.keras.layers.Lambda(
            positional_encoding_for_lambda,
            arguments={"model_dim": self.model_dim},
            name="transformer_positional_encoding"
        )(input_tensor)
        input_tensor = tf.keras.layers.Add()([input_tensor, pos_encoding])

        # Mask padded timesteps (assumed zero-padded) and optionally apply causal mask
        padding_mask = tf.keras.layers.Lambda(
            padding_mask_from_tensor,
            name="transformer_padding_mask"
        )(original_input)

        attention_mask = tf.keras.layers.Lambda(
            attention_mask_from_padding,
            arguments={"use_causal": bool(self.use_causal_mask)},
            name="transformer_attention_mask"
        )([padding_mask, input_tensor])

        # Multi-head self-attention
        attention = tf.keras.layers.MultiHeadAttention(num_heads = self.num_heads,
                                                       key_dim   = self.model_dim)(
                                                           input_tensor,
                                                           input_tensor,
                                                           attention_mask=attention_mask)
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
        group.create_dataset('use_causal_mask',         data=self.use_causal_mask)


    @classmethod
    def from_h5(cls, group: h5py.Group) -> Transformer:
        return cls(num_heads        =   int(group["number_of_heads"        ][()]),
                   model_dim        =   int(group["model_dimensions"       ][()]),
                   ff_dim           =   int(group["feed_forward_dimensions"][()]),
                   activation       =       group["activation_function"    ][()].decode('utf-8'),
                   dropout_rate     = float(group["dropout_rate"           ][()]),
                   use_causal_mask  = bool(group.get("use_causal_mask", True)))

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
        if "use_causal_mask" not in params and "causal_mask" in params:
            params["use_causal_mask"] = params.pop("causal_mask")

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
                "dropout_rate":            self.dropout_rate,
                "use_causal_mask":         self.use_causal_mask}

    @staticmethod
    def _positional_encoding(seq_len: tf.Tensor, model_dim: int) -> tf.Tensor:
        """Sinusoidal positional encoding broadcastable to (batch, seq_len, model_dim).

        Parameters
        ----------
        seq_len : tf.Tensor
            Scalar tensor representing sequence length for the current batch.
        model_dim : int
            Embedding dimension; the last dimension of the model inputs.

        Returns
        -------
        tf.Tensor
            Tensor of shape (1, seq_len, model_dim) containing sine/cosine
            positional encodings suitable for adding to the input embeddings.
        """
        positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)  # (seq_len, 1)
        dims = tf.cast(tf.range(model_dim)[tf.newaxis, :], tf.float32)      # (1, model_dim)
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(dims / 2.0)) / tf.cast(model_dim, tf.float32))
        angle_rads = positions * angle_rates

        even_mask = tf.cast(tf.equal(tf.math.floordiv(tf.range(model_dim), 1) % 2, 0), tf.float32)
        odd_mask = 1.0 - even_mask

        pos_encoding = tf.sin(angle_rads) * even_mask + tf.cos(angle_rads) * odd_mask
        return pos_encoding[tf.newaxis, ...]

    @staticmethod
    def _build_attention_mask(padding_mask: tf.Tensor, tensor: tf.Tensor, use_causal: tf.Tensor) -> tf.Tensor:
        """Build attention mask combining padding and optional causal constraint.

        Parameters
        ----------
        padding_mask : tf.Tensor
            Boolean tensor of shape (batch, seq_len) indicating valid timesteps.
        tensor : tf.Tensor
            Target tensor to infer sequence length, shape (batch, seq_len, dim).
        use_causal : tf.Tensor
            Scalar boolean tensor; if True, apply lower-triangular causal mask.

        Returns
        -------
        tf.Tensor
            Attention mask of shape (batch, seq_len, seq_len).
        """
        seq_len = tf.shape(tensor)[1]
        mask = tf.cast(padding_mask[:, tf.newaxis, :], tf.float32)  # (batch, 1, seq)
        def apply_causal():
            causal = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # (seq, seq)
            return mask * causal
        return tf.cond(use_causal, apply_causal, lambda: mask)
