from __future__ import annotations
from typing import Optional, Any, Dict, Tuple, Union
import numpy as np
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

from .graph import Graph
from ml_tools.model.nn_strategy.layer_sequence import LayerSequence


@register_keras_serializable()
class GraphSAGEConv(tf.keras.layers.Layer):
    """GraphSAGE-style convolution per time-slice with fixed adjacency.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (including any virtual globals).
    units : int
        Output feature dimension per node.
    aggregator : {'mean','sum'}, optional
        Neighbor aggregation method; default ``'mean'``.
    use_bias : bool, optional
        Whether to include a bias term; default True.
    adj_init : numpy.ndarray | None, optional
        Optional initial adjacency matrix to load into the non-trainable
        adjacency weight; if ``None`` uses ones until set by parent.
    """

    def __init__(self,
                 num_nodes: int,
                 units: int,
                 aggregator: str = 'mean',
                 use_bias: bool = True,
                 adj_init: Optional[np.ndarray] = None,
                 **kwargs):
        super().__init__(**kwargs)

        assert aggregator in ('mean', 'sum'), f"Unsupported aggregator {aggregator}"

        self.num_nodes  = int(num_nodes)
        self.units      = int(units)
        self.aggregator = aggregator
        self.use_bias   = bool(use_bias)
        self._adj_init  = adj_init

        self._adjacency:    Optional[tf.Variable] = None
        self._kernel_self:  Optional[tf.Variable] = None
        self._kernel_neigh: Optional[tf.Variable] = None
        self._bias:         Optional[tf.Variable] = None

    def build(self, input_shape: Union[Tuple[int, int], Tuple[int, int, int]]) -> None:
        feat_dim = int(input_shape[-1])

        self._adjacency = self.add_weight(
            name        = 'adjacency',
            shape       = (self.num_nodes, self.num_nodes),
            initializer = tf.keras.initializers.Constant(1.0 if self._adj_init is None else self._adj_init),
            trainable   = False)

        self._kernel_self = self.add_weight(
            name        = 'kernel_self',
            shape       = (feat_dim, self.units),
            initializer = 'glorot_uniform', trainable=True)

        self._kernel_neigh = self.add_weight(
            name        = 'kernel_neigh',
            shape       = (feat_dim, self.units),
            initializer = 'glorot_uniform', trainable=True)

        if self.use_bias:
            self._bias = self.add_weight(
                name        = 'bias',
                shape       = (self.units,),
                initializer = 'zeros',
                trainable   = True)
        else:
            self._bias = None

        super().build(input_shape)

    def set_adjacency(self, adj: tf.Tensor) -> None:
        if self._adjacency is None:
            raise RuntimeError('Layer not built yet; call build() first')
        self._adjacency.assign(adj)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass for a single time slice.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor of shape ``(B, N, F)`` containing per-node features.

        Returns
        -------
        tf.Tensor
            Output tensor of shape ``(B, N, units)``.
        """
        adj_no_self = self._adjacency - tf.linalg.diag(tf.linalg.diag_part(self._adjacency))
        neigh = tf.einsum('ij,bjf->bif', adj_no_self, inputs)
        if self.aggregator == 'mean':
            deg = tf.reduce_sum(adj_no_self, axis=1)
            deg = tf.where(deg > 0, deg, tf.ones_like(deg))
            neigh = neigh / tf.reshape(deg, (1, -1, 1))

        out = tf.tensordot(inputs, self._kernel_self, axes=[[-1], [0]])
        out += tf.tensordot(neigh, self._kernel_neigh, axes=[[-1], [0]])
        if self._bias is not None:
            out = out + self._bias
        return out

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
        base.update({'num_nodes':  self.num_nodes,
                     'units':      self.units,
                     'aggregator': self.aggregator,
                     'use_bias':   self.use_bias,
                     'adj_init':   adj_init_ser})
        return base


class SAGE(Graph):
    """GraphSAGE variant implementing ``Graph.make_conv_layer``.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Extended 3D spatial shape ``(H, W, D)`` used to construct adjacency.
    units : int
        Output feature dimension per node.
    ordering : {'feature_major','node_major'}, optional
        Feature layout; default ``'feature_major'``.
    pre_node_layers : LayerSequence | list, optional
        Per-node encoder to apply before message passing.
    spatial_feature_size : int | None, optional
        Per-spatial-node width ``S`` (required when using globals).
    global_feature_count : int, optional
        Number of virtual global nodes; default 0.
    connectivity, self_loops, normalize, distance_weighted,
    connect_global_to_all, connect_global_to_global, global_edge_weight :
        See ``Graph``.
    aggregator : {'mean','sum'}, optional
        SAGE neighbor aggregation; default ``'mean'``.
    use_bias : bool, optional
        Include bias in the SAGE conv; default True.
    """

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
                 aggregator:               str = 'mean',
                 use_bias:                 bool = True) -> None:
        assert aggregator in ('mean', 'sum'), f"aggregator = {aggregator}"
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
        self.aggregator = aggregator
        self.use_bias = bool(use_bias)

    def variant_name(self) -> str:
        return 'SAGE'

    def make_conv_layer(self, num_nodes: int, units: int, **kwargs):
        """Create the per-time-step SAGE conv layer.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the (possibly augmented) graph.
        units : int
            Output feature dimension per node.
        **kwargs : Any
            May include ``adj_init``.

        Returns
        -------
        tf.keras.layers.Layer
            Configured GraphSAGEConv instance.
        """
        adj_init = kwargs.get('adj_init', None)
        return GraphSAGEConv(num_nodes  = num_nodes,
                             units      = units,
                             aggregator = self.aggregator,
                             use_bias   = self.use_bias,
                             adj_init   = adj_init)

    def _save_variant(self, group) -> None:
        str_dtype = h5py.string_dtype()
        group.create_dataset('aggregator', data=self.aggregator, dtype=str_dtype)
        group.create_dataset('use_bias', data=bool(self.use_bias))

    @classmethod
    def from_h5(cls, group) -> 'SAGE':
        """Load a SAGE variant from an HDF5 group saved by ``Graph.save``."""
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
            from ml_tools.model.nn_strategy.layer_sequence import LayerSequence
            pre_node_layers = LayerSequence.from_h5(pre_node)
        else:
            pre_node_layers = None

        # Variant subgroup
        vgroup     = group.get('variant', None)
        aggregator = vgroup['aggregator'][()].decode('utf-8') if vgroup is not None else 'mean'
        use_bias   = bool(vgroup['use_bias'][()]) if vgroup is not None else True

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
                   aggregator               = aggregator,
                   use_bias                 = use_bias)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SAGE) and
            self._base_eq_fields(other) and
            self.aggregator == other.aggregator and
            self.use_bias   == other.use_bias
        )

    def __hash__(self) -> int:
        return hash((self._base_hash_fields(), self.aggregator, self.use_bias))
