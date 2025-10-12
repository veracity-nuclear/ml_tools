from __future__ import annotations
from typing import Optional, Any, Dict, Tuple, Union
from decimal import Decimal
import numpy as np

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

from ml_tools.model.nn_strategy.graph.graph import Graph
from ml_tools.model.nn_strategy.layer_sequence import LayerSequence


@register_keras_serializable()
class GraphAttentionConv(tf.keras.layers.Layer):
    """Simplified Graph Attention (GAT) layer with fixed adjacency.

    Single-head attention that computes attention coefficients over neighbors
    and aggregates neighbor features accordingly.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (including virtual globals).
    units : int
        Output feature dimension per node.
    alpha : float, optional
        LeakyReLU negative slope for attention logits, default 0.2.
    temperature : float, optional
        Softmax temperature applied to attention logits (e/temperature) to
        control sharpness; 1.0 leaves logits unchanged.
    use_bias : bool, optional
        Whether to include a bias term, default True.
    adj_init : numpy.ndarray | None, optional
        Initial adjacency to load into a non-trainable weight. If None, ones.
    """

    def __init__(self,
                 num_nodes:   int,
                 units:       int,
                 alpha:       float = 0.2,
                 temperature: float = 1.0,
                 use_bias:    bool = True,
                 adj_init:    Optional[np.ndarray] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_nodes   = int(num_nodes)
        self.units       = int(units)
        self.alpha       = float(alpha)
        self.temperature = float(temperature)
        self.use_bias    = bool(use_bias)
        self._adj_init   = adj_init

        self._adjacency: Optional[tf.Variable] = None
        self._kernel:    Optional[tf.Variable] = None
        self._attn_src:  Optional[tf.Variable] = None
        self._attn_dst:  Optional[tf.Variable] = None
        self._bias:      Optional[tf.Variable] = None

    def build(self, input_shape: Union[Tuple[int, int], Tuple[int, int, int]]) -> None:
        feat_dim = int(input_shape[-1])

        self._adjacency = self.add_weight(
            name        = 'adjacency',
            shape       = (self.num_nodes, self.num_nodes),
            initializer = tf.keras.initializers.Constant(1.0 if self._adj_init is None else self._adj_init),
            trainable   = False)

        self._kernel = self.add_weight(
            name        = 'kernel',
            shape       = (feat_dim, self.units),
            initializer = 'glorot_uniform',
            trainable   = True)

        self._attn_src = self.add_weight(
            name        = 'attn_src',
            shape       = (self.units, 1),
            initializer = 'glorot_uniform',
            trainable   = True)

        self._attn_dst = self.add_weight(
            name        = 'attn_dst',
            shape       = (self.units, 1),
            initializer = 'glorot_uniform',
            trainable   = True)

        if self.use_bias:
            self._bias = self.add_weight(
                name        = 'bias',
                shape       = (self.units,),
                initializer = 'zeros',
                trainable   = True)
        else:
            self._bias = None

        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (B, N, F); adjacency: (N, N)
        Wh = tf.tensordot(inputs, self._kernel, axes=[[-1], [0]])  # (B, N, U)

        # Compute attention logits e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
        e_src = tf.tensordot(Wh, self._attn_src, axes=[[-1], [0]])      # (B, N, 1)
        e_dst = tf.tensordot(Wh, self._attn_dst, axes=[[-1], [0]])      # (B, N, 1)
        e     = e_src + tf.transpose(e_dst, perm=[0, 2, 1])             # (B, N, N)
        e     = tf.nn.leaky_relu(e, alpha=self.alpha)

        # Mask by adjacency: set logits to large negative where no edge
        mask     = tf.cast(self._adjacency > 0, dtype=tf.bool)      # (N, N)
        neg_inf  = tf.constant(-1e9, dtype=e.dtype)
        e_masked = tf.where(mask[tf.newaxis, :, :], e, neg_inf)
        temp     = tf.cast(self.temperature, dtype=e_masked.dtype)
        alpha    = tf.nn.softmax(e_masked / temp, axis=-1)          # (B, N, N)

        h_prime = tf.matmul(alpha, Wh)                              # (B, N, U)
        if self._bias is not None:
            h_prime = h_prime + self._bias
        return h_prime

    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> tf.TensorShape:
        return tf.TensorShape((input_shape[0], input_shape[1], self.units))

    def get_config(self) -> Dict[str, Any]:
        base = super().get_config()
        if self._adj_init is None:
            adj_init_ser = None
        else:
            try:
                adj_init_ser = self._adj_init.tolist()
            except Exception:
                adj_init_ser = self._adj_init
        base.update({'num_nodes':   self.num_nodes,
                     'units':       self.units,
                     'alpha':       self.alpha,
                     'temperature': self.temperature,
                     'use_bias':    self.use_bias,
                     'adj_init':    adj_init_ser})
        return base


class GAT(Graph):
    """Graph Attention Network variant implementing ``Graph.make_conv_layer``.

    Parameters are the same as ``Graph`` with the following additions:

    alpha : float, optional
        LeakyReLU negative slope for attention, default 0.2.
    temperature : float, optional
        Softmax temperature applied to attention logits (e/temperature) to
        control sharpness; 1.0 leaves logits unchanged.
    use_bias : bool, optional
        Include bias in attention layer, default True.
    """

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    def __init__(self,
                 input_shape:              tuple[int, int, int],
                 units:                    int,
                 ordering:                 str = 'feature_major',
                 pre_node_layers:          Optional[Union[LayerSequence, list]] = None,
                 spatial_feature_size:     Optional[int] = None,
                 global_feature_count:     int = 0,
                 connectivity:             str = '2d-4',
                 self_loops:               bool = True,
                 normalize:                bool = True,
                 distance_weighted:        bool = False,
                 connect_global_to_all:    bool = True,
                 connect_global_to_global: bool = False,
                 global_edge_weight:       float = 1.0,
                 alpha:                    float = 0.2,
                 temperature:              float = 1.0,
                 use_bias:                 bool = True) -> None:

        super().__init__(input_shape              = input_shape,
                         units                    = units,
                         ordering                 = ordering,
                         pre_node_layers          = pre_node_layers,
                         spatial_feature_size     = spatial_feature_size,
                         global_feature_count     = global_feature_count,
                         connectivity             = connectivity,
                         self_loops               = self_loops,
                         normalize                = normalize,
                         distance_weighted        = distance_weighted,
                         connect_global_to_all    = connect_global_to_all,
                         connect_global_to_global = connect_global_to_global,
                         global_edge_weight       = global_edge_weight)

        assert alpha > 0.0, "alpha must be positive"
        assert temperature > 0.0, "temperature must be positive"

        self._alpha       = float(alpha)
        self._temperature = float(temperature)
        self._use_bias    = bool(use_bias)

    def variant_name(self) -> str:
        return 'GAT'

    def make_conv_layer(self, num_nodes: int, units: int, **kwargs):
        adj_init = kwargs.get('adj_init', None)
        return GraphAttentionConv(num_nodes   = num_nodes,
                                  units       = units,
                                  alpha       = self.alpha,
                                  temperature = self.temperature,
                                  use_bias    = self.use_bias,
                                  adj_init    = adj_init)

    def _save_variant(self, group) -> None:
        group.create_dataset('alpha',       data=float(self.alpha))
        group.create_dataset('temperature', data=float(self.temperature))
        group.create_dataset('use_bias',    data=bool(self.use_bias))

    def _variant_to_dict(self) -> dict:
        return {'alpha':       float(self.alpha),
                'temperature': float(self.temperature),
                'use_bias':    bool(self.use_bias)}

    @classmethod
    def from_h5(cls, group) -> GAT:
        # Base fields
        input_shape              = tuple(int(x) for x in group['input_shape'][()])
        ordering                 = group['ordering'][()].decode('utf-8')
        units                    = int(group['units'][()])
        sfs                      = int(group['spatial_feature_size'][()])
        spatial_feature_size     = None if sfs == -1 else sfs
        global_feature_count     = int(group['global_feature_count'][()])
        connectivity             = group['connectivity'][()].decode('utf-8')
        self_loops               = bool(group['self_loops'][()])
        normalize                = bool(group['normalize'][()])
        distance_weighted        = bool(group['distance_weighted'][()])
        connect_global_to_all    = bool(group['connect_global_to_all'][()])
        connect_global_to_global = bool(group['connect_global_to_global'][()])
        global_edge_weight       = float(group['global_edge_weight'][()])

        # Pre-node sequence
        pre_node = group.get('pre_node', None)
        if pre_node is not None:
            pre_node_layers = LayerSequence.from_h5(pre_node)
        else:
            pre_node_layers = None

        # Variant subgroup
        vgroup      = group.get('variant', None)
        if vgroup is not None:
            alpha       = float(vgroup.get('alpha', 0.2)[()])
            temperature = float(vgroup.get('temperature', 1.0)[()])
            use_bias    = bool(vgroup.get('use_bias', True)[()])
        else:
            alpha       = 0.2
            temperature = 1.0
            use_bias    = True

        return cls(input_shape              = input_shape,
                   units                    = units,
                   ordering                 = ordering,
                   pre_node_layers          = pre_node_layers,
                   spatial_feature_size     = spatial_feature_size,
                   global_feature_count     = global_feature_count,
                   connectivity             = connectivity,
                   self_loops               = self_loops,
                   normalize                = normalize,
                   distance_weighted        = distance_weighted,
                   connect_global_to_all    = connect_global_to_all,
                   connect_global_to_global = connect_global_to_global,
                   global_edge_weight       = global_edge_weight,
                   alpha                    = alpha,
                   temperature              = temperature,
                   use_bias                 = use_bias)

    @classmethod
    def from_dict(cls, data: dict) -> GAT:

        if 'input_shape' not in data:
            raise KeyError("GAT configuration must include 'input_shape' key")
        input_shape = tuple(int(x) for x in data['input_shape'])
        if 'units' not in data:
            raise KeyError("GAT configuration must include 'units' key")

        # Base fields
        units                    = int(data['units'])
        ordering                 = data.get('ordering', 'feature_major')
        sfs                      = int(data.get('spatial_feature_size', -1))
        spatial_feature_size     = None if sfs == -1 else sfs
        global_feature_count     = int(data.get('global_feature_count', 0))
        connectivity             = data.get('connectivity', '2d-4')
        self_loops               = bool(data.get('self_loops', True))
        normalize                = bool(data.get('normalize', True))
        distance_weighted        = bool(data.get('distance_weighted', False))
        connect_global_to_all    = bool(data.get('connect_global_to_all', True))
        connect_global_to_global = bool(data.get('connect_global_to_global', False))
        global_edge_weight       = float(data.get('global_edge_weight', 1.0))

        # Pre-node sequence
        pre_node_cfg = data.get('pre_node', None)
        if pre_node_cfg is not None:
            pre_node_layers = LayerSequence.from_dict(pre_node_cfg)
        else:
            pre_node_layers = None

        # Variant fields
        alpha       = float(data.get('alpha', 0.2))
        temperature = float(data.get('temperature', 1.0))
        use_bias    = bool(data.get('use_bias', True))

        return cls(input_shape              = input_shape,
                   units                    = units,
                   ordering                 = ordering,
                   pre_node_layers          = pre_node_layers,
                   spatial_feature_size     = spatial_feature_size,
                   global_feature_count     = global_feature_count,
                   connectivity             = connectivity,
                   self_loops               = self_loops,
                   normalize                = normalize,
                   distance_weighted        = distance_weighted,
                   connect_global_to_all    = connect_global_to_all,
                   connect_global_to_global = connect_global_to_global,
                   global_edge_weight       = global_edge_weight,
                   alpha                    = alpha,
                   temperature              = temperature,
                   use_bias                 = use_bias)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GAT) and
            self._base_eq_fields(other) and
            abs(self.alpha - other.alpha) <= 1e-9 and
            abs(self.temperature - other.temperature) <= 1e-9 and
            self.use_bias == other.use_bias
        )

    def __hash__(self) -> int:
        return hash((
            self._base_hash_fields(),
            Decimal(self.alpha).quantize(Decimal('1e-9')),
            Decimal(self.temperature).quantize(Decimal('1e-9')),
            self.use_bias,
        ))
