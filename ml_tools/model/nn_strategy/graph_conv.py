from __future__ import annotations
from typing import Any, Literal, Tuple, Union, List, Optional, Dict
from math import isclose
from decimal import Decimal
import numpy as np
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

from ml_tools.model.nn_strategy.layer import Layer, Activation
from ml_tools.model.nn_strategy.layer_sequence import LayerSequence
from ml_tools.model.nn_strategy.pass_through import PassThrough


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


def _neighbors_offsets(dim:          int,
                       connectivity: Literal['1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26']) -> List[Tuple[int, int, int]]:
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
def merge_batch_node(t: tf.Tensor) -> tf.Tensor:
    """Merges the batch and node axes of a 4D tensor.

    Parameters
    ----------
    t : tf.Tensor
        A tensor of shape ``(batch, time, nodes, features)``.

    Returns
    -------
    tf.Tensor
        A tensor of shape ``(batch*nodes, time, features)`` suitable for
        applying per-node encoders that operate independently on each node.
    """
    b  = tf.shape(t)[0]
    n  = tf.shape(t)[2]
    tm = tf.shape(t)[1]
    f  = tf.shape(t)[3]
    return tf.reshape(t, (b * n, tm, f))


@register_keras_serializable()
def unmerge_batch_node(inputs: Union[Tuple[tf.Tensor, tf.Tensor], List[tf.Tensor]]) -> tf.Tensor:
    """Restores the batch and node axes after merging.

    Parameters
    ----------
    inputs : Union[tf.Tensor, list[tf.Tensor], tuple]
        Either a tensor ``t`` of shape ``(batch*nodes, time, features)`` and a
        reference tensor ``ref`` of shape ``(batch, time, nodes, _)`` provided as
        a 2-tuple/list ``[t, ref]``, or a single tensor when used in a context
        where the reference shape is known by closure.

    Returns
    -------
    tf.Tensor
        A tensor of shape ``(batch, time, nodes, features)``.
    """
    if isinstance(inputs, (list, tuple)):
        t, ref = inputs
    else:
        raise ValueError("unmerge_batch_node expects [t, ref]")
    b  = tf.shape(ref)[0]
    n  = tf.shape(ref)[2]
    tm = tf.shape(ref)[1]
    u  = tf.shape(t)[2]
    return tf.reshape(t, (b, tm, n, u))


@register_keras_serializable()
def merge_batch_time(t: tf.Tensor) -> tf.Tensor:
    """Merges the batch and time axes of a 4D tensor.

    Parameters
    ----------
    t : tf.Tensor
        A tensor of shape ``(batch, time, nodes, features)``.

    Returns
    -------
    tf.Tensor
        A tensor of shape ``(batch*time, nodes, features)`` suitable for
        applying graph layers that expect ``(B, N, F)``.
    """
    b = tf.shape(t)[0]
    tm = tf.shape(t)[1]
    n = tf.shape(t)[2]
    f = tf.shape(t)[3]
    return tf.reshape(t, (b * tm, n, f))


@register_keras_serializable()
def unmerge_batch_time(inputs: Union[Tuple[tf.Tensor, tf.Tensor], List[tf.Tensor]]) -> tf.Tensor:
    """Restores the batch and time axes after merging.

    Parameters
    ----------
    inputs : Tuple[tf.Tensor, tf.Tensor]
        A 2-tuple/list ``[t, ref]`` where ``t`` has shape ``(batch*time, nodes, features)`` and
        ``ref`` is a reference tensor with shape ``(batch, time, nodes, _)``.

    Returns
    -------
    tf.Tensor
        A tensor of shape ``(batch, time, nodes, features)``.
    """
    if isinstance(inputs, (list, tuple)):
        t, ref = inputs
    else:
        raise ValueError("unmerge_batch_time expects [t, ref]")
    b = tf.shape(ref)[0]
    tm = tf.shape(ref)[1]
    n = tf.shape(ref)[2]
    u = tf.shape(t)[2]
    return tf.reshape(t, (b, tm, n, u))


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
    adj_init : Optional[numpy.ndarray], optional
        Optional initial value for the non-trainable adjacency weight of shape
        ``(N, N)``. When provided, the layer initializes its internal adjacency
        to this value; otherwise ones are used until set by the parent.

    Notes
    -----
    - Preferred input tensor shape: ``(B, N, F)`` (we normalize to this form).
    - Output tensor shape: ``(B, N, units)``.
    - The adjacency ``A`` is a non-trainable weight initialized via
      ``adj_init`` (or ones by default) and can be set by the parent layer.
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
        self.num_nodes = int(num_nodes)
        self.units = int(units)
        self.aggregator = aggregator
        self.use_bias = bool(use_bias)

        # Optional initializer for adjacency (non-trainable)
        self._adj_init = adj_init

        # These will be populated in build(); declared here to satisfy linters
        self._adjacency: Optional[tf.Variable] = None
        self._kernel_self: Optional[tf.Variable] = None
        self._kernel_neigh: Optional[tf.Variable] = None
        self._bias: Optional[tf.Variable] = None

    def build(self, input_shape: Union[Tuple[int, int], Tuple[int, int, int]]) -> None:
        """Creates layer weights based on the input shape.

        Parameters
        ----------
        input_shape : Union[Tuple[int, int], Tuple[int, int, int]]
            Shape tuple as ``(batch, N, F)`` or ``(N, F)``; only the final two
            dims are used to infer ``N`` and ``F``.
        """
        # input_shape: (N, F) or (batch, N, F)
        n_nodes = int(input_shape[-2])
        assert n_nodes == self.num_nodes, "num_nodes mismatch in GraphSAGEConv"
        feat_dim = int(input_shape[-1])

        # Adjacency (un-normalized) is a non-trainable weight; initialize if provided
        adj_initializer = (tf.keras.initializers.Constant(self._adj_init)
                           if self._adj_init is not None else tf.keras.initializers.Ones())
        self._adjacency = self.add_weight(
            name='adjacency',
            shape=(self.num_nodes, self.num_nodes),
            initializer=adj_initializer,
            trainable=False,
        )
        # Self and neighbor kernels
        self._kernel_self = self.add_weight(
            name='kernel_self',
            shape=(feat_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self._kernel_neigh = self.add_weight(
            name='kernel_neigh',
            shape=(feat_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        if self.use_bias:
            self._bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )
        else:
            self._bias = None
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Computes the GraphSAGE forward pass for a single timestep.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor of shape ``(B, N, F)`` containing per-node features.

        Returns
        -------
        tf.Tensor
            Output tensor of shape ``(B, N, units)`` with updated per-node embeddings.
        """
        # Remove self-loops for neighbor aggregation
        adj_no_self = self._adjacency - tf.linalg.diag(tf.linalg.diag_part(self._adjacency))  # (N, N)

        # Aggregate neighbors: (N,N) x (B,N,F) -> (B,N,F)
        neigh = tf.einsum('ij,bjf->bif', adj_no_self, inputs)
        if self.aggregator == 'mean':
            deg = tf.reduce_sum(adj_no_self, axis=1)  # (N,)
            deg = tf.where(deg > 0, deg, tf.ones_like(deg))
            neigh = neigh / tf.reshape(deg, (1, -1, 1))

        # Self and neighbor transforms across feature dim
        out = tf.tensordot(inputs, self._kernel_self, axes=[[-1], [0]])
        out += tf.tensordot(neigh, self._kernel_neigh, axes=[[-1], [0]])
        if self._bias is not None:
            out = out + self._bias
        return out

    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> tf.TensorShape:
        """Infers output shape for inputs of shape ``(batch, N, F)``.

        Parameters
        ----------
        input_shape : Tuple[int, int, int]
            Shape tuple ``(batch, N, F)``.

        Returns
        -------
        tf.TensorShape
            Output shape ``(batch, N, units)``.
        """
        return tf.TensorShape((input_shape[0], input_shape[1], self.units))

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config for Keras SavedModel compatibility.

        Returns
        -------
        Dict[str, Any]
            A dictionary including ``num_nodes``, ``units``, ``aggregator``,
            and ``use_bias`` along with base Keras config.
        """
        base = super().get_config()
        base.update({
            'num_nodes':  self.num_nodes,
            'units':      self.units,
            'aggregator': self.aggregator,
            'use_bias':   self.use_bias,
        })
        return base


@Layer.register_subclass("GraphConv")
class GraphConv(Layer):
    """Graph convolution over 1D/2D/3D grids with optional pre-node encoders.

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
    pre_node_layers : Optional[Union[LayerSequence, list[Layer]]]
        Layers applied independently to each spatial node before message passing.
        Defaults to a single PassThrough.
    spatial_feature_size : Optional[int]
        Per-spatial-node feature width S; required when using global features (G > 0).
    global_feature_count : int, optional
        Number of global scalars G appended after spatial features (treated as
        virtual nodes), by default 0.
    connect_global_to_all : bool, optional
        Whether to connect each virtual node to all spatial nodes, by default True.
    connect_global_to_global : bool, optional
        Whether to fully connect virtual nodes among themselves, by default False.
    global_edge_weight : float, optional
        Edge weight for edges incident to virtual nodes, by default 1.0.
    aggregator : {'mean', 'sum'}, optional
        GraphSAGE neighbor aggregation method, by default 'mean'.
    use_bias : bool, optional
        Whether the inner GraphSAGEConv uses a bias term, by default True.
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
    pre_node_layers : LayerSequence
        Per-node encoder sequence.
    spatial_feature_size : Optional[int]
        Per-spatial-node feature width S (if globals used).
    global_feature_count : int
        Number of global scalars G.
    connect_global_to_all : bool
        Whether virtual nodes connect to all spatial nodes.
    connect_global_to_global : bool
        Whether virtual nodes connect among themselves.
    global_edge_weight : float
        Edge weight for virtual-node edges.
    aggregator : str
        GraphSAGE aggregator ('mean' or 'sum').
    use_bias : bool
        Whether the inner GraphSAGEConv uses a bias term.

    Notes
    -----
    - Input shape: ``(batch, time, D)``. Without virtual nodes ``D=N*F``;
      with virtual nodes ``D=N*S+G``.
    - Output shape: ``(batch, time, (N[+G]) * units)``.
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

    @property
    def connectivity(self) -> str:
        return self._connectivity

    @connectivity.setter
    def connectivity(self, value: str) -> None:
        assert value in ('1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26'), f"connectivity = {value}"
        self._connectivity = value

    @property
    def self_loops(self) -> bool:
        return self._self_loops

    @self_loops.setter
    def self_loops(self, value: bool) -> None:
        self._self_loops = bool(value)

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        self._normalize = bool(value)

    @property
    def ordering(self) -> str:
        return self._ordering

    @ordering.setter
    def ordering(self, value: str) -> None:
        assert value in ('feature_major', 'node_major'), f"ordering = {value}"
        self._ordering = value

    @property
    def pre_node_layers(self) -> LayerSequence:
        return self._pre_node_sequence

    @pre_node_layers.setter
    def pre_node_layers(self, value: Union[LayerSequence, list[Layer]]) -> None:
        if isinstance(value, LayerSequence):
            self._pre_node_sequence = value
        else:
            assert isinstance(value, list) and len(value) > 0
            self._pre_node_sequence = LayerSequence(value)

    @property
    def spatial_feature_size(self) -> Optional[int]:
        return self._spatial_feature_size

    @spatial_feature_size.setter
    def spatial_feature_size(self, value: Optional[int]) -> None:
        if value is None:
            self._spatial_feature_size = None
        else:
            assert int(value) > 0, f"spatial_feature_size = {value}"
            self._spatial_feature_size = int(value)

    @property
    def global_feature_count(self) -> int:
        return self._global_feature_count

    @global_feature_count.setter
    def global_feature_count(self, value: int) -> None:
        assert int(value) >= 0, f"global_feature_count = {value}"
        self._global_feature_count = int(value)

    @property
    def connect_global_to_all(self) -> bool:
        return self._connect_global_to_all

    @connect_global_to_all.setter
    def connect_global_to_all(self, value: bool) -> None:
        self._connect_global_to_all = bool(value)

    @property
    def connect_global_to_global(self) -> bool:
        return self._connect_global_to_global

    @connect_global_to_global.setter
    def connect_global_to_global(self, value: bool) -> None:
        self._connect_global_to_global = bool(value)

    @property
    def global_edge_weight(self) -> float:
        return self._global_edge_weight

    @global_edge_weight.setter
    def global_edge_weight(self, value: float) -> None:
        self._global_edge_weight = float(value)

    @property
    def aggregator(self) -> str:
        return self._aggregator

    @aggregator.setter
    def aggregator(self, value: str) -> None:
        assert value in ('mean', 'sum'), f"aggregator = {value}"
        self._aggregator = value

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @use_bias.setter
    def use_bias(self, value: bool) -> None:
        self._use_bias = bool(value)

    def __init__(self,
                 input_shape:              ShapeType,
                 units:                    int,
                 activation:               Activation = 'relu',
                 connectivity:             Literal['1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26'] = '2d-4',
                 self_loops:               bool = True,
                 normalize:                bool = True,
                 ordering:                 Literal['feature_major', 'node_major'] = 'feature_major',
                 pre_node_layers:          Optional[Union[LayerSequence, list[Layer]]] = None,
                 spatial_feature_size:     Optional[int] = None,
                 global_feature_count:     int = 0,
                 connect_global_to_all:    bool = True,
                 connect_global_to_global: bool = False,
                 global_edge_weight:       float = 1.0,
                 aggregator:               str = 'mean',
                 use_bias:                 bool = True,
                 dropout_rate:             float = 0.0,
                 batch_normalize:          bool = False,
                 layer_normalize:          bool = False) -> None:

        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.input_shape   = input_shape
        self.units         = units
        self.activation    = activation
        self.connectivity  = connectivity
        self.self_loops    = self_loops
        self.normalize     = normalize
        self.ordering      = ordering

        # Configure pre-node sequence (applied identically to each node)
        if pre_node_layers is None:
            self._pre_node_sequence = LayerSequence([PassThrough()])
        elif isinstance(pre_node_layers, LayerSequence):
            self._pre_node_sequence = pre_node_layers
        else:
            assert len(pre_node_layers) > 0, f"len(pre_node_layers) = {len(pre_node_layers)}"
            self._pre_node_sequence = LayerSequence(pre_node_layers)

        self.spatial_feature_size     = spatial_feature_size
        self.global_feature_count     = int(global_feature_count)
        self.connect_global_to_all    = connect_global_to_all
        self.connect_global_to_global = connect_global_to_global
        self.global_edge_weight       = float(global_edge_weight)
        self.aggregator               = aggregator
        self.use_bias                 = use_bias

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
                 self._pre_node_sequence == other._pre_node_sequence and
                 self.spatial_feature_size == other.spatial_feature_size and
                 self.global_feature_count == other.global_feature_count and
                 self.connect_global_to_all == other.connect_global_to_all and
                 self.connect_global_to_global == other.connect_global_to_global and
                 isclose(self.global_edge_weight, other.global_edge_weight, rel_tol=1e-9) and
                 self.aggregator == other.aggregator and
                 self.use_bias == other.use_bias and
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
            self._pre_node_sequence,
            self.spatial_feature_size,
            self.global_feature_count,
            self.connect_global_to_all,
            self.connect_global_to_global,
            Decimal(self.global_edge_weight).quantize(Decimal('1e-9')),
            self.aggregator,
            self.use_bias,
            Decimal(self.dropout_rate).quantize(Decimal('1e-9')),
            self.batch_normalize,
            self.layer_normalize,
        ))

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        N = _prod(self._input_shape)
        G = self.global_feature_count

        if G == 0:
            y = self._build_spatial_only(input_tensor, N)
        else:
            y = self._build_with_globals(input_tensor, N, G)

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

    def _build_spatial_only(self, input_tensor: tf.Tensor, N: int) -> tf.Tensor:
        """Build the computation for purely spatial nodes (no virtual globals).

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape ``(batch, time, D)`` where ``D`` is divisible by ``N``.
        N : int
            Number of spatial nodes derived from ``input_shape``.

        Returns
        -------
        tf.Tensor
            Tensor of shape ``(batch, time, N, units)`` after GraphSAGEConv.
        """
        assert input_tensor.shape[-1] % N == 0, "Input length must be divisible by number of nodes"
        feat_per_node = input_tensor.shape[-1] // N
        if self.ordering == 'feature_major':
            x = tf.keras.layers.Reshape(target_shape=(-1, int(feat_per_node), int(N)))(input_tensor)
            x = tf.keras.layers.Permute((1, 3, 2))(x)
        else:
            x = tf.keras.layers.Reshape(target_shape=(-1, int(N), int(feat_per_node)))(input_tensor)

        x, feat_per_node_after = self._apply_pre_node_layers(x)

        conv = GraphSAGEConv(num_nodes=int(N), units=self.units, aggregator=self.aggregator, use_bias=self.use_bias,
                             adj_init=self._adjacency_np)
        conv.build((int(N), int(feat_per_node_after)))
        # Merge (batch,time) for conv, then restore
        x_bt = tf.keras.layers.Lambda(merge_batch_time)(x)
        y_bt = conv(x_bt)
        y    = tf.keras.layers.Lambda(unmerge_batch_time)([y_bt, x])
        return y

    def _build_with_globals(self, input_tensor: tf.Tensor, N: int, G: int) -> tf.Tensor:
        """Build the computation when using virtual global nodes.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape ``(batch, time, D)`` where ``D = N*S + G``.
        N : int
            Number of spatial nodes.
        G : int
            Number of virtual global nodes (global features).

        Returns
        -------
        tf.Tensor
            Tensor of shape ``(batch, time, N+G, units)`` after GraphSAGEConv.
        """
        assert self.spatial_feature_size is not None, "spatial_feature_size must be provided when using global features"
        S = int(self.spatial_feature_size)
        D_expected = N * S + G
        assert int(input_tensor.shape[-1]) == D_expected, \
            f"Input length {int(input_tensor.shape[-1])} does not match N*S+G = {D_expected}"

        spatial_flat = tf.keras.layers.Lambda(lambda t: t[..., : N * S])(input_tensor)
        global_vec   = tf.keras.layers.Lambda(lambda t: t[..., N * S :      ])(input_tensor)

        if self.ordering == 'feature_major':
            x_spatial = tf.keras.layers.Reshape(target_shape=(-1, S, int(N)))(spatial_flat)
            x_spatial = tf.keras.layers.Permute((1, 3, 2))(x_spatial)
        else:
            x_spatial = tf.keras.layers.Reshape(target_shape=(-1, int(N), S))(spatial_flat)

        x_spatial, S_prime = self._apply_pre_node_layers(x_spatial)
        x_global = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(int(G * S_prime)))(global_vec)
        x_global = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((int(G), S_prime)))(x_global)

        x = tf.keras.layers.Concatenate(axis=2)([x_spatial, x_global])

        N_total = N + G
        conv = GraphSAGEConv(num_nodes=int(N_total), units=self.units, aggregator=self.aggregator, use_bias=self.use_bias,
                             adj_init=self._adjacency_np)
        conv.build((int(N_total), int(S_prime)))
        x_bt = tf.keras.layers.Lambda(merge_batch_time)(x)
        y_bt = conv(x_bt)
        y    = tf.keras.layers.Lambda(unmerge_batch_time)([y_bt, x])
        return y

    def _apply_pre_node_layers(self, x: tf.Tensor) -> Tuple[tf.Tensor, int]:
        """Applies the configured pre-node layers to each node.

        Parameters
        ----------
        x : tf.Tensor
            Tensor of shape ``(batch, time, nodes, features)``.

        Returns
        -------
        Tuple[tf.Tensor, int]
            The transformed tensor of shape ``(batch, time, nodes, features')``
            and the new per-node feature width ``features'``.
        """
        x_merged = tf.keras.layers.Lambda(merge_batch_node)(x)
        z = self._pre_node_sequence.build(x_merged)
        x = tf.keras.layers.Lambda(unmerge_batch_node)([z, x])
        feat_after = int(z.shape[-1]) if z.shape[-1] is not None else int(x.shape[-1])
        return x, feat_after

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
        group.create_dataset('spatial_feature_size', data=self.spatial_feature_size
                             if self.spatial_feature_size is not None else -1)
        group.create_dataset('global_feature_count', data=self.global_feature_count)
        group.create_dataset('connect_global_to_all', data=self.connect_global_to_all)
        group.create_dataset('connect_global_to_global', data=self.connect_global_to_global)
        group.create_dataset('global_edge_weight',   data=self.global_edge_weight)
        group.create_dataset('aggregator',           data=self.aggregator, dtype=h5py.string_dtype())
        group.create_dataset('use_bias',             data=self.use_bias)
        # Save pre-node layer sequence
        pre_node_group = group.create_group('pre_node')
        self._pre_node_sequence.save(pre_node_group)

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
        use_bias_ds = group.get('use_bias', None)
        use_bias = bool(use_bias_ds[()]) if use_bias_ds is not None else True
        # Load pre-node sequence (optional for backward compatibility)
        pre_node = group.get('pre_node', None)
        if pre_node is not None:
            pre_node_sequence = LayerSequence.from_h5(pre_node)
        else:
            pre_node_sequence = LayerSequence([PassThrough()])

        return cls(input_shape   = input_shape,
                   units         = units,
                   activation    = activation,
                   connectivity  = connectivity,
                   self_loops    = self_loops,
                   normalize     = normalize,
                   ordering      = ordering,
                   pre_node_layers = pre_node_sequence,
                   spatial_feature_size = spatial_feature_size,
                   global_feature_count = global_feature_count,
                   connect_global_to_all = connect_global_to_all,
                   connect_global_to_global = connect_global_to_global,
                   global_edge_weight = global_edge_weight,
                   aggregator    = aggregator,
                   use_bias      = use_bias,
                   dropout_rate  = dropout_rate,
                   batch_normalize = batch_norm,
                   layer_normalize = layer_norm)
