from ml_tools.optimizer.search_space import (
    StructDimension,
    IntDimension,
    FloatDimension,
    CategoricalDimension,
    BoolDimension,
)
from ml_tools.optimizer.nn_search_space.layer_sequence import LayerSequence


class Graph(StructDimension):
    """Abstract base search-space for GraphConv graph variants (domains, not values).

    Owns common graph configuration domains; variants (e.g., SAGE, GAT) add only
    their specific domains (such as ``aggregator`` or ``use_bias``).

    Parameters
    ----------
    input_shape : CategoricalDimension
        Choices of spatial input shape tuples (H,), (H, W), or (H, W, D).
    units : IntDimension
        Inclusive range for per-node output feature dimension.
    ordering : CategoricalDimension, optional
        Choices for feature layout; default ['feature_major'].
        Acceptable values must mirror model graph ordering:
        {'feature_major', 'node_major'}. See
        ml_tools.model.nn_strategy.graph.Graph for details.
    pre_node_layers : LayerSequence | None, optional
        LayerSequence dimension applied identically per node before propagation (or None).
    spatial_feature_size : IntDimension | None, optional
        Inclusive range for per-spatial-node width S when using globals (or None if absent).
    global_feature_count : IntDimension, optional
        Inclusive range for the number of virtual global nodes; default IntDimension(0, 0).
    connectivity : CategoricalDimension, optional
        Choices for grid neighborhood; default ['2d-4'].
        Acceptable values must mirror model graph connectivity:
        {'1d-2','2d-4','2d-8','3d-6','3d-18','3d-26'}.
    self_loops : BoolDimension, optional
        Domain for including self-loops; default [True].
    normalize : BoolDimension, optional
        Domain for applying symmetric degree normalization; default [True].
    distance_weighted : BoolDimension, optional
        Domain for using inverse Manhattan neighbor weights; default [False].
    connect_global_to_all : BoolDimension, optional
        Domain for connecting globals to all spatial nodes; default [True].
    connect_global_to_global : BoolDimension, optional
        Domain for fully connecting global nodes among themselves; default [False].
    global_edge_weight : FloatDimension, optional
        Inclusive range for weight on edges incident to globals; default FloatDimension(1.0, 1.0).

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
