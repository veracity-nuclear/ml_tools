from __future__ import annotations
from typing import Any, Literal, Tuple, Union
from math import isclose
from decimal import Decimal
import numpy as np
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

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


def _prod(t: Tuple[int, int, int]) -> int:
    """Computes the product of a 3D shape tuple.

    Parameters
    ----------
    t : Tuple[int, int, int]
        A 3D shape represented as integers ``(H, W, D)``.

    Returns
    -------
    int
        The product ``H * W * D``.
    """
    return int(t[0]) * int(t[1]) * int(t[2])


def _neighbors_offsets(dim: int, connectivity: Literal['1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26']):
    """Returns neighbor offsets for grid connectivity in 1D/2D/3D.

    Parameters
    ----------
    dim : int
        The grid dimensionality, one of {1, 2, 3}.
    connectivity : {'1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26'}
        Neighborhood definition:
        - '1d-2': left/right
        - '2d-4': faces (up, down, left, right)
        - '2d-8': face + corners
        - '3d-6': faces
        - '3d-18': faces + edges
        - '3d-26': faces + edges + corners

    Returns
    -------
    List[Tuple[int, int, int]]
        Neighbor offsets ``(dx, dy, dz)`` to apply to a coordinate.
    """
    if dim == 1:
        return [(1, 0, 0), (-1, 0, 0)]  # '1d-2'
    if dim == 2:
        offs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                manhattan = abs(dx) + abs(dy)
                if connectivity == '2d-4' and manhattan == 1:
                    offs.append((dx, dy, 0))
                if connectivity == '2d-8' and 1 <= manhattan <= 2:
                    offs.append((dx, dy, 0))
        return offs
    if dim == 3:
        offs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    manhattan = abs(dx) + abs(dy) + abs(dz)
                    if connectivity == '3d-6' and manhattan == 1:
                        offs.append((dx, dy, dz))
                    elif connectivity == '3d-18' and 1 <= manhattan <= 2:
                        offs.append((dx, dy, dz))
                    elif connectivity == '3d-26':
                        offs.append((dx, dy, dz))
        return offs
    raise ValueError(f"Unsupported dimension {dim}")


def _build_adjacency(shape: Tuple[int, int, int],
                     connectivity: Literal['1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26'],
                     self_loops: bool,
                     normalize: bool) -> np.ndarray:
    """Builds a dense adjacency matrix for a grid graph.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Extended 3D spatial shape ``(H, W, D)``. Use ones to pad inactive
        dimensions when using 1D or 2D grids.
    connectivity : {'1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26'}
        Neighborhood definition for edges; see ``_neighbors_offsets``.
    self_loops : bool
        Whether to include identity connections on each node.
    normalize : bool
        If True, returns symmetric normalized adjacency ``D^{-1/2} A D^{-1/2}``.

    Returns
    -------
    numpy.ndarray
        A dense adjacency matrix of shape ``(N, N)``, where ``N = H * W * D``.
    """
    H, W, D = shape
    dim = 1 if (W == 1 and D == 1) else 2 if (D == 1) else 3
    N = _prod(shape)
    A = np.zeros((N, N), dtype=np.float32)

    def idx(x: int, y: int, z: int) -> int:
        return z * (H * W) + y * H + x

    offsets = _neighbors_offsets(dim, connectivity)

    for z in range(D):
        for y in range(W):
            for x in range(H):
                i = idx(x, y, z)
                if self_loops:
                    A[i, i] = 1.0
                for dx, dy, dz in offsets:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                        j = idx(nx, ny, nz)
                        A[i, j] = 1.0
                        A[j, i] = 1.0

    if normalize:
        deg = np.sum(A, axis=1)
        # avoid division by zero
        inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        A = D_inv_sqrt @ A @ D_inv_sqrt

    return A


@register_keras_serializable()
class GraphSAGEConv(tf.keras.layers.Layer):
    """GraphSAGE-style convolution per time-slice with fixed adjacency.

    Computes ``h' = act( X @ W_self + AGG(A, X) @ W_neigh + b )`` where
    ``AGG`` is typically mean aggregation over neighbors (excluding self).

    Parameters
    ----------
    num_nodes : int
        Number of nodes ``N`` in the graph (including any virtual nodes).
    units : int
        Output feature dimension per node.
    aggregator : {'mean', 'sum'}, optional
        Aggregation type for neighbor messages, by default 'mean'.
    use_bias : bool, optional
        Whether to include a bias term, by default True.

    Notes
    -----
    - Input tensor shape: ``(N, F)``
    - Output tensor shape: ``(N, units)``
    - The adjacency ``A`` is a non-trainable weight seeded by the parent layer.
    """

    def __init__(self, num_nodes: int, units: int, aggregator: str = 'mean', use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        assert aggregator in ('mean', 'sum'), f"Unsupported aggregator {aggregator}"
        self.num_nodes = int(num_nodes)
        self.units = int(units)
        self.aggregator = aggregator
        self.use_bias = bool(use_bias)

    def build(self, input_shape):
        # input_shape: (N, F)
        assert int(input_shape[0]) == self.num_nodes, "num_nodes mismatch in GraphSAGEConv"
        feat_dim = int(input_shape[-1])
        # Adjacency (un-normalized) is loaded externally and stored here as non-trainable
        self.adj = self.add_weight(
            name='adjacency',
            shape=(self.num_nodes, self.num_nodes),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
        )
        # Self and neighbor kernels
        self.kernel_self = self.add_weight(
            name='kernel_self',
            shape=(feat_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.kernel_neigh = self.add_weight(
            name='kernel_neigh',
            shape=(feat_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (N, F)
        # Remove self-loops for neighbor aggregation
        adj_no_self = self.adj - tf.linalg.diag(tf.linalg.diag_part(self.adj))
        neigh = tf.linalg.matmul(adj_no_self, inputs)  # (N, F)
        if self.aggregator == 'mean':
            deg = tf.reduce_sum(adj_no_self, axis=1, keepdims=True)  # (N, 1)
            deg = tf.where(deg > 0, deg, tf.ones_like(deg))
            neigh = neigh / deg
        # Self and neighbor transforms
        out = tf.tensordot(inputs, self.kernel_self, axes=[[1], [0]])
        out += tf.tensordot(neigh, self.kernel_neigh, axes=[[1], [0]])
        if self.bias is not None:
            out = out + self.bias
        return out

    def get_config(self):
        base = super().get_config()
        base.update({
            'num_nodes': self.num_nodes,
            'units': self.units,
            'aggregator': self.aggregator,
            'use_bias': self.use_bias,
        })
        return base


@Layer.register_subclass("GraphConv")
class GraphConv(Layer):
    """Graph convolution over 1D/2D/3D grids with feature-major flattening.

    This layer consumes flattened per-timestep feature vectors and performs
    a graph propagation using grid connectivity. It supports feature-major
    ordering as used in this codebase, and returns a flattened vector to
    maintain compatibility with other layers.

    Parameters
    ----------
    input_shape : Tuple[int] or Tuple[int, int] or Tuple[int, int, int]
        Spatial grid shape ``(H[, W[, D]])``. Missing dims are treated as 1.
    units : int
        Output feature dimension per node.
    activation : Activation, optional
        Activation to apply after propagation, by default 'relu'.
    connectivity : {'1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26'}, optional
        Neighborhood specification, by default '2d-4'.
    self_loops : bool, optional
        Include self connections, by default True.
    normalize : bool, optional
        Apply symmetric degree normalization, by default True.
    ordering : {'feature_major', 'node_major'}, optional
        Flattening order of the input feature vector, by default 'feature_major'.
    dropout_rate : float, optional
        Dropout probability applied after activation, by default 0.0.
    batch_normalize : bool, optional
        Apply BatchNormalization after propagation, by default False.
    layer_normalize : bool, optional
        Apply LayerNormalization after propagation, by default False.

    Attributes
    ----------
    input_shape : Tuple[int, int, int]
        The extended ``(H, W, D)`` spatial shape.
    units : int
        Output feature dimension per node.
    activation : Activation
        Activation function used.
    connectivity : str
        Neighborhood specification.
    self_loops : bool
        Whether self connections are included.
    normalize : bool
        Whether degree normalization is applied.
    ordering : str
        Flattening order used for reshape/permute.

    Notes
    -----
    - Input shape: ``(batch, time, D)`` where ``D = N * F``.
    - Output shape: ``(batch, time, N * units)``.
    - For 'feature_major' ordering, the last dimension is laid out as
      ``[feat0(all N), feat1(all N), ...]``.
    """

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: ShapeType) -> None:
        eshape = _extend_shape(input_shape)
        assert eshape[0] > 0 and eshape[1] > 0 and eshape[2] > 0
        self._input_shape = eshape

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

    def __init__(self,
                 input_shape: ShapeType,
                 units: int,
                 activation: Activation = 'relu',
                 connectivity: Literal['1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26'] = '2d-4',
                 self_loops: bool = True,
                 normalize: bool = True,
                 ordering: Literal['feature_major', 'node_major'] = 'feature_major',
                 spatial_feature_size: int | None = None,
                 global_feature_count: int = 0,
                 connect_global_to_all: bool = True,
                 connect_global_to_global: bool = False,
                 global_edge_weight: float = 1.0,
                 aggregator: str = 'mean',
                 dropout_rate: float = 0.0,
                 batch_normalize: bool = False,
                 layer_normalize: bool = False) -> None:

        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.input_shape = input_shape
        self.units = units
        self.activation = activation
        self.connectivity = connectivity
        self.self_loops = self_loops
        self.normalize = normalize
        self.ordering = ordering
        self.spatial_feature_size = spatial_feature_size
        self.global_feature_count = int(global_feature_count)
        self.connect_global_to_all = connect_global_to_all
        self.connect_global_to_global = connect_global_to_global
        self.global_edge_weight = float(global_edge_weight)
        self.aggregator = aggregator

        # Build base grid adjacency, then augment with virtual global nodes if any
        base_adj = _build_adjacency(self._input_shape, connectivity, self_loops, normalize)
        N = base_adj.shape[0]
        G = self.global_feature_count
        if G > 0:
            A = np.zeros((N + G, N + G), dtype=np.float32)
            A[:N, :N] = base_adj
            if self.connect_global_to_all:
                A[N:, :N] = self.global_edge_weight
                A[:N, N:] = self.global_edge_weight
            if self.connect_global_to_global:
                # connect all virtual nodes to each other
                for i in range(G):
                    for j in range(G):
                        if i == j:
                            continue
                        A[N + i, N + j] = self.global_edge_weight
            if self_loops:
                for k in range(G):
                    A[N + k, N + k] = 1.0
            self._adjacency_np = A
        else:
            self._adjacency_np = base_adj

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                (isinstance(other, GraphConv) and
                 self.input_shape == other.input_shape and
                 self.units == other.units and
                 self.activation == other.activation and
                 self.connectivity == other.connectivity and
                 self.self_loops == other.self_loops and
                 self.normalize == other.normalize and
                 self.ordering == other.ordering and
                 self.spatial_feature_size == other.spatial_feature_size and
                 self.global_feature_count == other.global_feature_count and
                 self.connect_global_to_all == other.connect_global_to_all and
                 self.connect_global_to_global == other.connect_global_to_global and
                 isclose(self.global_edge_weight, other.global_edge_weight, rel_tol=1e-9) and
                 self.aggregator == other.aggregator and
                 isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                 self.batch_normalize == other.batch_normalize and
                 self.layer_normalize == other.layer_normalize)
               )

    def __hash__(self) -> int:
        return hash((
            self.input_shape,
            self.units,
            self.activation,
            self.connectivity,
            self.self_loops,
            self.normalize,
            self.ordering,
            self.spatial_feature_size,
            self.global_feature_count,
            self.connect_global_to_all,
            self.connect_global_to_global,
            Decimal(self.global_edge_weight).quantize(Decimal('1e-9')),
            self.aggregator,
            Decimal(self.dropout_rate).quantize(Decimal('1e-9')),
            self.batch_normalize,
            self.layer_normalize,
        ))

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        N = _prod(self._input_shape)
        G = self.global_feature_count

        if G == 0:
            # Pure spatial case: same as before but via GraphSAGEConv
            assert input_tensor.shape[-1] % N == 0, "Input length must be divisible by number of nodes"
            feat_per_node = input_tensor.shape[-1] // N
            if self.ordering == 'feature_major':
                x = tf.keras.layers.Reshape(target_shape=(-1, int(feat_per_node), int(N)))(input_tensor)
                x = tf.keras.layers.Permute((1, 3, 2))(x)
            else:
                x = tf.keras.layers.Reshape(target_shape=(-1, int(N), int(feat_per_node)))(input_tensor)
            conv = GraphSAGEConv(num_nodes=int(N), units=self.units, aggregator=self.aggregator, use_bias=True)
            conv.build((int(N), int(feat_per_node)))
            conv.adj.assign(tf.convert_to_tensor(self._adjacency_np, dtype=tf.float32))
            y = tf.keras.layers.TimeDistributed(conv)(x)
        else:
            # Spatial + global virtual nodes
            assert self.spatial_feature_size is not None, "spatial_feature_size must be provided when using global features"
            S = int(self.spatial_feature_size)
            D_expected = N * S + G
            assert int(input_tensor.shape[-1]) == D_expected, \
                f"Input length {int(input_tensor.shape[-1])} does not match N*S+G = {D_expected}"

            # Slice spatial and global parts
            spatial_flat = tf.keras.layers.Lambda(lambda t: t[..., : N * S])(input_tensor)
            global_vec   = tf.keras.layers.Lambda(lambda t: t[..., N * S :      ])(input_tensor)

            if self.ordering == 'feature_major':
                x_spatial = tf.keras.layers.Reshape(target_shape=(-1, S, int(N)))(spatial_flat)
                x_spatial = tf.keras.layers.Permute((1, 3, 2))(x_spatial)  # (batch,time,N,S)
            else:
                x_spatial = tf.keras.layers.Reshape(target_shape=(-1, int(N), S))(spatial_flat)

            # Map G-global vector to G virtual node feature matrix (G,S)
            # Use a Dense(G*S) then reshape to (G,S) to allow distinct per-virtual-node parameters
            x_global = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(int(G * S)))(global_vec)
            x_global = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((int(G), S)))(x_global)

            # Concatenate along node axis -> (batch,time, N+G, S)
            x = tf.keras.layers.Concatenate(axis=2)([x_spatial, x_global])

            N_total = N + G
            conv = GraphSAGEConv(num_nodes=int(N_total), units=self.units, aggregator=self.aggregator, use_bias=True)
            conv.build((int(N_total), int(S)))
            conv.adj.assign(tf.convert_to_tensor(self._adjacency_np, dtype=tf.float32))
            y = tf.keras.layers.TimeDistributed(conv)(x)

        if self.batch_normalize:
            y = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(y)

        if self.layer_normalize:
            y = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(y)

        y = tf.keras.layers.Activation(self.activation)(y)

        if self.dropout_rate > 0.:
            y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=self.dropout_rate))(y)

        # Flatten back to (batch, time, N*units)
        y = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(y)
        return y

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type',                data='GraphConv', dtype=h5py.string_dtype())
        group.create_dataset('input_shape',         data=self.input_shape)
        group.create_dataset('activation_function', data=self.activation, dtype=h5py.string_dtype())
        group.create_dataset('number_of_units',     data=self.units)
        group.create_dataset('connectivity',        data=self.connectivity, dtype=h5py.string_dtype())
        group.create_dataset('self_loops',          data=self.self_loops)
        group.create_dataset('normalize',           data=self.normalize)
        group.create_dataset('ordering',            data=self.ordering, dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate',        data=self.dropout_rate)
        group.create_dataset('batch_normalize',     data=self.batch_normalize)
        group.create_dataset('layer_normalize',     data=self.layer_normalize)
        group.create_dataset('spatial_feature_size', data=self.spatial_feature_size if self.spatial_feature_size is not None else -1)
        group.create_dataset('global_feature_count', data=self.global_feature_count)
        group.create_dataset('connect_global_to_all', data=self.connect_global_to_all)
        group.create_dataset('connect_global_to_global', data=self.connect_global_to_global)
        group.create_dataset('global_edge_weight',   data=self.global_edge_weight)
        group.create_dataset('aggregator',           data=self.aggregator, dtype=h5py.string_dtype())

    @classmethod
    def from_h5(cls, group: h5py.Group) -> GraphConv:
        input_shape   = tuple(int(x) for x in group['input_shape'][()])
        activation    = group['activation_function'][()].decode('utf-8')
        units         = int(group['number_of_units'][()])
        connectivity  = group['connectivity'][()].decode('utf-8')
        self_loops    = bool(group['self_loops'][()])
        normalize     = bool(group['normalize'][()])
        ordering      = group['ordering'][()].decode('utf-8')
        dropout_rate  = float(group['dropout_rate'][()])
        batch_norm    = bool(group['batch_normalize'][()])
        layer_norm    = bool(group['layer_normalize'][()])
        spatial_feature_size = int(group.get('spatial_feature_size', -1)[()])
        if spatial_feature_size == -1:
            spatial_feature_size = None
        global_feature_count = int(group.get('global_feature_count', 0)[()])
        connect_global_to_all = bool(group.get('connect_global_to_all', True)[()])
        connect_global_to_global = bool(group.get('connect_global_to_global', False)[()])
        global_edge_weight = float(group.get('global_edge_weight', 1.0)[()])
        aggregator = group.get('aggregator', None)
        aggregator = aggregator[()].decode('utf-8') if aggregator is not None else 'mean'

        return cls(input_shape   = input_shape,
                   units         = units,
                   activation    = activation,
                   connectivity  = connectivity,
                   self_loops    = self_loops,
                   normalize     = normalize,
                   ordering      = ordering,
                   spatial_feature_size = spatial_feature_size,
                   global_feature_count = global_feature_count,
                   connect_global_to_all = connect_global_to_all,
                   connect_global_to_global = connect_global_to_global,
                   global_edge_weight = global_edge_weight,
                   aggregator    = aggregator,
                   dropout_rate  = dropout_rate,
                   batch_normalize = batch_norm,
                   layer_normalize = layer_norm)
