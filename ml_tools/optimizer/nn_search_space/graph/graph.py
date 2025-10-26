from ml_tools.optimizer.search_space import (
    StructDimension,
    IntDimension,
    FloatDimension,
    CategoricalDimension,
    BoolDimension,
)
from ml_tools.optimizer.nn_search_space.layer_sequence import LayerSequence


class Graph(StructDimension):
    """Abstract base search-space for graph variants used by GraphConv.

    Owns common graph configuration fields; variants (e.g., SAGE) add only
    their specific parameters (such as ``aggregator`` or ``use_bias``).

    Parameters
    ----------
    input_shape : CategoricalDimension
        Choice of spatial input shape tuples (H,), (H, W), or (H, W, D).
    units : IntDimension
        Per-node output feature dimension.
    ordering : CategoricalDimension, optional
        Feature layout; default ['feature_major'].
        Acceptable values must mirror model graph ordering:
        {'feature_major', 'node_major'}. See
        ml_tools.model.nn_strategy.graph.Graph for details.
    pre_node_layers : LayerSequence | None, optional
        Per-node encoder applied identically to each node before propagation.
    spatial_feature_size : IntDimension | None, optional
        Per-spatial-node width S (required when using globals). Use None for absent.
    global_feature_count : IntDimension, optional
        Number of virtual global nodes; default IntDimension(0, 0).
    connectivity : CategoricalDimension, optional
        Grid neighborhood; default ['2d-4'].
        Acceptable values must mirror model graph connectivity:
        {'1d-2','2d-4','2d-8','3d-6','3d-18','3d-26'}. See
        ml_tools.model.nn_strategy.graph.Graph for details.
    self_loops : BoolDimension, optional
        Whether to include self-loops; default [True].
    normalize : BoolDimension, optional
        Apply symmetric degree normalization; default [True].
    distance_weighted : BoolDimension, optional
        Use inverse Manhattan neighbor weights; default [False].
    connect_global_to_all : BoolDimension, optional
        Connect globals to all spatial nodes; default [True].
    connect_global_to_global : BoolDimension, optional
        Fully connect global nodes among themselves; default [False].
    global_edge_weight : FloatDimension, optional
        Weight for edges incident to globals; default FloatDimension(1.0, 1.0).

    Notes
    -----
    Variants should set a ``variant`` field (CategoricalDimension) used by the
    model factory for dispatch.
    """

    @property
    def input_shape(self) -> CategoricalDimension:
        return self.fields["input_shape"]

    @property
    def units(self) -> IntDimension:
        return self.fields["units"]

    @property
    def ordering(self) -> CategoricalDimension:
        return self.fields["ordering"]

    @property
    def spatial_feature_size(self):
        return self.fields.get("spatial_feature_size")

    @property
    def global_feature_count(self) -> IntDimension:
        return self.fields["global_feature_count"]

    @property
    def connectivity(self) -> CategoricalDimension:
        return self.fields["connectivity"]

    @property
    def self_loops(self) -> BoolDimension:
        return self.fields["self_loops"]

    @property
    def normalize(self) -> BoolDimension:
        return self.fields["normalize"]

    @property
    def distance_weighted(self) -> BoolDimension:
        return self.fields["distance_weighted"]

    @property
    def connect_global_to_all(self) -> BoolDimension:
        return self.fields["connect_global_to_all"]

    @property
    def connect_global_to_global(self) -> BoolDimension:
        return self.fields["connect_global_to_global"]

    @property
    def global_edge_weight(self) -> FloatDimension:
        return self.fields["global_edge_weight"]

    def __init__(self,
                 input_shape:              CategoricalDimension,
                 units:                    IntDimension,
                 ordering:                 CategoricalDimension = CategoricalDimension(['feature_major']),
                 pre_node_layers:          LayerSequence | None = None,
                 spatial_feature_size:     IntDimension | None = None,
                 global_feature_count:     IntDimension = IntDimension(0, 0),
                 connectivity:             CategoricalDimension = CategoricalDimension(['2d-4']),
                 self_loops:               BoolDimension = BoolDimension([True]),
                 normalize:                BoolDimension = BoolDimension([True]),
                 distance_weighted:        BoolDimension = BoolDimension([False]),
                 connect_global_to_all:    BoolDimension = BoolDimension([True]),
                 connect_global_to_global: BoolDimension = BoolDimension([False]),
                 global_edge_weight:       FloatDimension = FloatDimension(1.0, 1.0)) -> None:

        fields: dict[str, object] = {'input_shape':              input_shape,
                                     'units':                    units,
                                     'ordering':                 ordering,
                                     'global_feature_count':     global_feature_count,
                                     'connectivity':             connectivity,
                                     'self_loops':               self_loops,
                                     'normalize':                normalize,
                                     'distance_weighted':        distance_weighted,
                                     'connect_global_to_all':    connect_global_to_all,
                                     'connect_global_to_global': connect_global_to_global,
                                     'global_edge_weight':       global_edge_weight}
        if pre_node_layers is not None:
            fields['pre_node'] = pre_node_layers
        if spatial_feature_size is not None:
            fields['spatial_feature_size'] = spatial_feature_size

        super().__init__(fields)
