from __future__ import annotations
from typing import Tuple, Optional, Union, List
from abc import ABC, abstractmethod
from decimal import Decimal
import numpy as np
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf

from ml_tools.model.nn_strategy.layer import Layer
from ml_tools.model.nn_strategy.layer_sequence import LayerSequence
from ml_tools.model.nn_strategy.pass_through import PassThrough
from ml_tools.model.nn_strategy.graph.utils import (
    merge_batch_node,
    unmerge_batch_node,
    merge_batch_time,
    unmerge_batch_time,
    _build_adjacency,
    _prod,
    _extend_shape,
)

class Graph(ABC):
    """Abstract graph variant used by GraphConv.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Extended 3D spatial shape ``(H, W, D)`` used to construct adjacency.
    units : int
        Output feature dimension per node.
    ordering : {'feature_major','node_major'}, optional
        Layout of per-timestep feature vectors; default ``'feature_major'``.
    pre_node_layers : LayerSequence | list[Layer] | None, optional
        Per-node encoder applied identically to each node before propagation.
    spatial_feature_size : int | None, optional
        Per-spatial-node feature width ``S`` when using global nodes.
    global_feature_count : int, optional
        Number of virtual global nodes appended after spatial nodes; default 0.
    connectivity : {'1d-2','2d-4','2d-8','3d-6','3d-18','3d-26'}, optional
        Grid neighborhood for adjacency; default ``'2d-4'``.
    self_loops : bool, optional
        Whether to include self-loops in adjacency; default True.
    normalize : bool, optional
        Apply symmetric degree normalization to adjacency; default True.
    distance_weighted : bool, optional
        Use inverse Manhattan neighbor weights; default False.
    connect_global_to_all : bool, optional
        Connect globals to all spatial nodes; default True.
    connect_global_to_global : bool, optional
        Fully connect global nodes among themselves; default False.
    global_edge_weight : float, optional
        Weight for edges incident to globals; default 1.0.
    """

    _adjacency_np: Optional[np.ndarray] = None

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    @property
    def units(self) -> int:
        return self._units

    @property
    def ordering(self) -> str:
        return self._ordering

    @property
    def pre_node_sequence(self) -> LayerSequence:
        return self._pre_node_sequence

    @property
    def spatial_feature_size(self) -> Optional[int]:
        return self._spatial_feature_size

    @property
    def global_feature_count(self) -> int:
        return self._global_feature_count

    @property
    def connectivity(self) -> str:
        return self._connectivity

    @property
    def self_loops(self) -> bool:
        return self._self_loops

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def distance_weighted(self) -> bool:
        return self._distance_weighted

    @property
    def connect_global_to_all(self) -> bool:
        return self._connect_global_to_all

    @property
    def connect_global_to_global(self) -> bool:
        return self._connect_global_to_global

    @property
    def global_edge_weight(self) -> float:
        return self._global_edge_weight

    def __init__(self,
                 input_shape:              Tuple[int, int, int],
                 units:                    int,
                 ordering:                 str = 'feature_major',
                 pre_node_layers:          Optional[Union[LayerSequence, List[Layer]]] = None,
                 spatial_feature_size:     Optional[int] = None,
                 global_feature_count:     int = 0,
                 connectivity:             str = '2d-4',
                 self_loops:               bool = True,
                 normalize:                bool = True,
                 distance_weighted:        bool = False,
                 connect_global_to_all:    bool = True,
                 connect_global_to_global: bool = False,
                 global_edge_weight:       float = 1.0) -> None:

        assert ordering in ('feature_major', 'node_major')

        if global_feature_count > 0:
            assert spatial_feature_size is not None and spatial_feature_size > 0, (
                "spatial_feature_size must be provided and positive when using global nodes")

        if pre_node_layers is None:
            pre_node_seq = LayerSequence([PassThrough()])
        elif isinstance(pre_node_layers, LayerSequence):
            pre_node_seq = pre_node_layers
        else:
            assert isinstance(pre_node_layers, list) and len(pre_node_layers) > 0
            pre_node_seq = LayerSequence(pre_node_layers)

        eshape = _extend_shape(tuple(input_shape))
        assert all(int(d) > 0 for d in eshape), f"input_shape must be positive, got {eshape}"
        assert int(units) > 0, f"units must be > 0, got {units}"
        assert connectivity in ('1d-2','2d-4','2d-8','3d-6','3d-18','3d-26'), f"invalid connectivity {connectivity}"
        assert int(global_feature_count) >= 0, f"global_feature_count must be >= 0, got {global_feature_count}"
        if spatial_feature_size is not None:
            assert int(spatial_feature_size) > 0, f"spatial_feature_size must be > 0, got {spatial_feature_size}"

        self._input_shape              = eshape
        self._units                    = int(units)
        self._ordering                 = ordering
        self._pre_node_sequence        = pre_node_seq
        if spatial_feature_size is None:
            self._spatial_feature_size = None
        else:
            self._spatial_feature_size = int(spatial_feature_size)
        self._global_feature_count     = int(global_feature_count)
        self._connectivity             = connectivity
        self._self_loops               = bool(self_loops)
        self._normalize                = bool(normalize)
        self._distance_weighted        = bool(distance_weighted)
        self._connect_global_to_all    = bool(connect_global_to_all)
        self._connect_global_to_global = bool(connect_global_to_global)
        self._global_edge_weight       = float(global_edge_weight)

    def variant_name(self) -> str:
        """
        Returns a short identifier for the variant.

        This method should be implemented by subclasses to provide a concise
        string that identifies the specific variant of the model or strategy,
        such as 'SAGE', 'GCN', etc.

        Returns
        -------
        str
            A short string identifier for the variant.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError


    def _base_eq_fields(self, other: 'Graph') -> bool:
        """
        Compare fields common to all Graph variants.

        Parameters
        ----------
        other : Graph
            The other Graph instance to compare against.

        Returns
        -------
        bool
            True if all base fields are equal (within a tolerance for floating-point
            values), False otherwise.
        """
        return (
            isinstance(other, Graph) and
            self.units                    == other.units and
            self.ordering                 == other.ordering and
            self.pre_node_sequence        == other.pre_node_sequence and
            self.spatial_feature_size     == other.spatial_feature_size and
            self.global_feature_count     == other.global_feature_count and
            self.connectivity             == other.connectivity and
            self.self_loops               == other.self_loops and
            self.normalize                == other.normalize and
            self.distance_weighted        == other.distance_weighted and
            self.connect_global_to_all    == other.connect_global_to_all and
            self.connect_global_to_global == other.connect_global_to_global and
            abs(self.global_edge_weight - other.global_edge_weight) <= 1e-9
        )

    def _base_hash_fields(self) -> tuple:
        """Return base fields for hashing in subclasses.

        Returns
        -------
        tuple
            Tuple of hashable core configuration values.
        """
        return (
            self.units,
            self.ordering,
            self.pre_node_sequence,
            self.spatial_feature_size,
            self.global_feature_count,
            self.connectivity,
            self.self_loops,
            self.normalize,
            self.distance_weighted,
            self.connect_global_to_all,
            self.connect_global_to_global,
            Decimal(self.global_edge_weight).quantize(Decimal('1e-9')),
        )

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    def prepare(self) -> None:
        """Prepare and cache the adjacency matrix for the graph based on the current configuration.
        """

        base_adj = _build_adjacency(self._input_shape,
                                    self._connectivity,
                                    self._self_loops,
                                    self._normalize,
                                    self._distance_weighted)
        N = base_adj.shape[0]
        G = self._global_feature_count
        if G > 0:
            A = np.zeros((N + G, N + G), dtype=np.float32)
            A[:N, :N] = base_adj
            if self._connect_global_to_all:
                A[N:, :N] = self._global_edge_weight
                A[:N, N:] = self._global_edge_weight
            if self._connect_global_to_global:
                for i in range(G):
                    for j in range(G):
                        if i == j:
                            continue
                        A[N + i, N + j] = self._global_edge_weight
                        A[N + j, N + i] = self._global_edge_weight
            if self._self_loops:
                for k in range(G):
                    A[N + k, N + k] = 1.0
            self._adjacency_np = A
        else:
            self._adjacency_np = base_adj

    @abstractmethod
    def make_conv_layer(self, num_nodes: int, units: int, **kwargs):
        """Return a tf.keras layer implementing variant-specific propagation.

        Subclasses must implement and return a layer with signature
        (B,N,F)->(B,N,units). ``kwargs`` is variant-specific.

        Parameters
        ----------
        num_nodes : int
            Number of nodes ``N`` in the graph.
        units : int
            Output feature dimension per node.
        **kwargs
            Additional variant-specific parameters.
        """
        raise NotImplementedError

    def _apply_pre_node_layers(self, x: tf.Tensor) -> tuple[tf.Tensor, int]:
        """Apply the pre-node encoder sequence to each node independently.

        Parameters
        ----------
        x : tf.Tensor
            Tensor of shape ``(batch, time, nodes, features)``.

        Returns
        -------
        tuple[tf.Tensor, int]
            The transformed tensor and the new per-node feature width.
        """
        x_merged   = tf.keras.layers.Lambda(merge_batch_node)(x)
        z          = self._pre_node_sequence.build(x_merged)
        x          = tf.keras.layers.Lambda(unmerge_batch_node)([z, x])
        feat_after = int(z.shape[-1]) if z.shape[-1] is not None else int(x.shape[-1])
        return x, feat_after

    def build_spatial_only(self, input_tensor: tf.Tensor, N: int) -> tf.Tensor:
        """Build computation without global nodes.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Tensor of shape ``(batch, time, D)`` with ``D`` divisible by ``N``.
        N : int
            Number of spatial nodes.

        Returns
        -------
        tf.Tensor
            Tensor of shape ``(batch, time, N, units)``.
        """
        assert int(input_tensor.shape[-1]) % N == 0, "Input length must be divisible by number of nodes"

        feat_per_node = int(input_tensor.shape[-1]) // int(N)
        if self._ordering == 'feature_major':
            x = tf.keras.layers.Reshape(target_shape=(-1, int(feat_per_node), int(N)))(input_tensor)
            x = tf.keras.layers.Permute((1, 3, 2))(x)
        else:
            x = tf.keras.layers.Reshape(target_shape=(-1, int(N), int(feat_per_node)))(input_tensor)

        x, feat_per_node_after = self._apply_pre_node_layers(x)

        conv = self.make_conv_layer(num_nodes=int(N), units=int(self._units), adj_init=self._adjacency_np)
        conv.build((int(N), int(feat_per_node_after)))

        x_bt = tf.keras.layers.Lambda(merge_batch_time)(x)
        y_bt = conv(x_bt)
        y    = tf.keras.layers.Lambda(unmerge_batch_time)([y_bt, x])

        return y

    def build_with_globals(self, input_tensor: tf.Tensor, N: int, G: int) -> tf.Tensor:
        """Build computation with appended virtual global nodes.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Tensor of shape ``(batch, time, D)`` where ``D = N*S + G``.
        N : int
            Number of spatial nodes.
        G : int
            Number of global nodes.

        Returns
        -------
        tf.Tensor
            Tensor of shape ``(batch, time, N+G, units)``.
        """
        assert self._spatial_feature_size is not None, "spatial_feature_size must be provided when using global features"

        S = int(self._spatial_feature_size)
        D_expected = int(N) * S + int(G)
        assert int(input_tensor.shape[-1]) == D_expected, (
            f"Input length {int(input_tensor.shape[-1])} does not match N*S+G = {D_expected}")

        spatial_flat = tf.keras.layers.Lambda(lambda t: t[..., : N * S])(input_tensor)
        global_vec   = tf.keras.layers.Lambda(lambda t: t[..., N * S :      ])(input_tensor)

        if self._ordering == 'feature_major':
            x_spatial = tf.keras.layers.Reshape(target_shape=(-1, S, int(N)))(spatial_flat)
            x_spatial = tf.keras.layers.Permute((1, 3, 2))(x_spatial)
        else:
            x_spatial = tf.keras.layers.Reshape(target_shape=(-1, int(N), S))(spatial_flat)

        x_spatial, S_prime = self._apply_pre_node_layers(x_spatial)
        x_global = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(int(G * S_prime)))(global_vec)
        x_global = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((int(G), int(S_prime))))(x_global)

        x = tf.keras.layers.Concatenate(axis=2)([x_spatial, x_global])

        N_total = int(N) + int(G)
        conv = self.make_conv_layer(num_nodes=N_total, units=int(self._units), adj_init=self._adjacency_np)
        conv.build((int(N_total), int(S_prime)))

        x_bt = tf.keras.layers.Lambda(merge_batch_time)(x)
        y_bt = conv(x_bt)
        y    = tf.keras.layers.Lambda(unmerge_batch_time)([y_bt, x])

        return y

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Build the per-time-step graph computation for the given input.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Tensor of shape ``(batch, time, D)`` following the variant's
            configured ordering and global-node layout.

        Returns
        -------
        tf.Tensor
            Tensor of shape ``(batch, time, N[+G], units)``.
        """
        if self._adjacency_np is None:
            self.prepare()
        N = _prod(self._input_shape)
        G = self._global_feature_count
        if G == 0:
            return self.build_spatial_only(input_tensor, N)
        return self.build_with_globals(input_tensor, N, G)

    def save(self, group) -> None:
        """Persist this Graph variant into an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            Group to populate with variant settings and layers.
        """

        str_dtype = h5py.string_dtype()
        group.create_dataset('graph_type', data=self.variant_name(), dtype=str_dtype)
        group.create_dataset('input_shape', data=self._input_shape)
        group.create_dataset('units', data=int(self._units))
        group.create_dataset('ordering', data=self._ordering, dtype=str_dtype)
        group.create_dataset('spatial_feature_size', data=(self._spatial_feature_size if self._spatial_feature_size is not None
                                                           else -1))
        group.create_dataset('global_feature_count', data=int(self._global_feature_count))
        group.create_dataset('connectivity', data=self._connectivity, dtype=str_dtype)
        group.create_dataset('self_loops', data=bool(self._self_loops))
        group.create_dataset('normalize', data=bool(self._normalize))
        group.create_dataset('distance_weighted', data=bool(self._distance_weighted))
        group.create_dataset('connect_global_to_all', data=bool(self._connect_global_to_all))
        group.create_dataset('connect_global_to_global', data=bool(self._connect_global_to_global))
        group.create_dataset('global_edge_weight', data=float(self._global_edge_weight))

        # Save pre-node layer sequence under sub-group
        pre_node_group = group.create_group('pre_node')
        self._pre_node_sequence.save(pre_node_group)

        # Allow variant to persist extra fields
        vgroup = group.create_group('variant')
        self._save_variant(vgroup)

    def to_dict(self) -> dict:
        """Serialize this Graph configuration to a dict suitable for from_dict.

        Includes variant name under key 'variant' and all constructor parameters
        including pre-node layers under 'pre_node'.
        """
        base = {
            'variant':                  self.variant_name(),
            'input_shape':              tuple(self._input_shape),
            'units':                    int(self._units),
            'ordering':                 self._ordering,
            'spatial_feature_size':     (self._spatial_feature_size if self._spatial_feature_size is not None else -1),
            'global_feature_count':     int(self._global_feature_count),
            'connectivity':             self._connectivity,
            'self_loops':               bool(self._self_loops),
            'normalize':                bool(self._normalize),
            'distance_weighted':        bool(self._distance_weighted),
            'connect_global_to_all':    bool(self._connect_global_to_all),
            'connect_global_to_global': bool(self._connect_global_to_global),
            'global_edge_weight':       float(self._global_edge_weight),
        }
        if self._spatial_feature_size is None:
            base['spatial_feature_size'] = -1

        if isinstance(self._pre_node_sequence, LayerSequence):
            base['pre_node'] = self._pre_node_sequence.to_dict()

        try:
            extra = self._variant_to_params()
            if isinstance(extra, dict):
                base.update(extra)
        except AttributeError:
            pass
        return base

    def _variant_to_dict(self) -> dict:
        """Return variant-specific parameters for to_dict; override in subclasses."""
        return {}

    def _save_variant(self, group) -> None:
        """
        Persist variant-specific fields to the given storage group.

        This hook is invoked by the save routine to allow subclasses to record any
        additional attributes, metadata, or configuration that distinguish a particular
        variant. The base implementation performs no action.

        Parameters
        ----------
        group : object
            A writable storage handle where data should be persisted (e.g., an
            h5py.Group, a zarr Group, or a dict-like/mapping object).
        """
