from ml_tools.optimizer.nn_search_space.graph.graph import Graph
from ml_tools.optimizer.nn_search_space.layer_sequence import LayerSequence as LayerSequenceDim
from ml_tools.optimizer.search_space import IntDimension, CategoricalDimension, BoolDimension, FloatDimension


class GAT(Graph):
    """Search-space struct for the GAT (Graph Attention) variant (domains, not values).

    Parameters
    ----------
    input_shape : CategoricalDimension
        Choices of spatial input shape tuples (H,), (H, W), or (H, W, D).
    units : IntDimension
        Inclusive range for per-node output feature dimension.
    ordering : CategoricalDimension, optional
        Choices for feature layout; default ['feature_major'].
    pre_node_layers : LayerSequenceDim | None, optional
        LayerSequence dimension applied identically per node before propagation (or None).
    spatial_feature_size : IntDimension | None, optional
        Inclusive range for per-spatial-node width S (or None if absent).
    global_feature_count : IntDimension, optional
        Inclusive range for number of virtual global nodes; default IntDimension(0, 0).
    connectivity : CategoricalDimension, optional
        Choices for grid neighborhood (e.g., '2d-4'); default ['2d-4'].
    self_loops : BoolDimension, optional
        Domain for including self-loops; default [True].
    normalize : BoolDimension, optional
        Domain for symmetric degree normalization; default [True].
    distance_weighted : BoolDimension, optional
        Domain for inverse Manhattan neighbor weights; default [False].
    connect_global_to_all : BoolDimension, optional
        Domain for connecting globals to all spatial nodes; default [True].
    connect_global_to_global : BoolDimension, optional
        Domain for fully connecting global nodes among themselves; default [False].
    global_edge_weight : FloatDimension | None, optional
        Inclusive range for edges incident to globals (or None); default FloatDimension(1.0, 1.0).
    alpha : FloatDimension, optional
        Inclusive range for LeakyReLU negative slope for attention logits; default FloatDimension(0.2, 0.2).
    temperature : FloatDimension, optional
        Inclusive range for softmax temperature applied to attention logits; default FloatDimension(1.0, 1.0).
    use_bias : BoolDimension, optional
        Domain for including a bias term; default [True].

    Attributes
    ----------
    alpha : FloatDimension
        Domain for LeakyReLU negative slope for attention logits.
    temperature : FloatDimension
        Domain for softmax temperature for attention logits.
    use_bias : BoolDimension
        Domain for bias inclusion flag for attention layer.
    """

    @property
    def alpha(self) -> FloatDimension:
        return self.fields["alpha"]

    @alpha.setter
    def alpha(self, value: FloatDimension) -> None:
        assert 0.0 <= value.low <= 1.0 and 0.0 <= value.high <= 1.0, \
            f"alpha (LeakyReLU slope) must be within [0,1], got [{value.low}, {value.high}]"
        self.fields["alpha"] = value

    @property
    def temperature(self) -> FloatDimension:
        return self.fields["temperature"]

    @temperature.setter
    def temperature(self, value: FloatDimension) -> None:
        self.fields["temperature"] = value

    @property
    def use_bias(self) -> BoolDimension:
        return self.fields["use_bias"]

    @use_bias.setter
    def use_bias(self, value: BoolDimension) -> None:
        self.fields["use_bias"] = value

    def __init__(self,
                 input_shape:              CategoricalDimension,
                 units:                    IntDimension,
                 ordering:                 CategoricalDimension = CategoricalDimension(['feature_major']),
                 pre_node_layers:          LayerSequenceDim | None = None,
                 spatial_feature_size:     IntDimension | None = None,
                 global_feature_count:     IntDimension = IntDimension(0, 0),
                 connectivity:             CategoricalDimension = CategoricalDimension(['2d-4']),
                 self_loops:               BoolDimension = BoolDimension([True]),
                 normalize:                BoolDimension = BoolDimension([True]),
                 distance_weighted:        BoolDimension = BoolDimension([False]),
                 connect_global_to_all:    BoolDimension = BoolDimension([True]),
                 connect_global_to_global: BoolDimension = BoolDimension([False]),
                 global_edge_weight:       FloatDimension | None = None,
                 alpha:                    FloatDimension = FloatDimension(0.2, 0.2),
                 temperature:              FloatDimension = FloatDimension(1.0, 1.0),
                 use_bias:                 BoolDimension = BoolDimension([True])) -> None:

        # Initialize common graph fields via base Graph
        super().__init__(input_shape            = input_shape,
                         units                  = units,
                         ordering               = ordering,
                         pre_node_layers        = pre_node_layers,
                         spatial_feature_size   = spatial_feature_size,
                         global_feature_count   = global_feature_count,
                         connectivity           = connectivity,
                         self_loops             = self_loops,
                         normalize              = normalize,
                         distance_weighted      = distance_weighted,
                         connect_global_to_all  = connect_global_to_all,
                         connect_global_to_global = connect_global_to_global,
                         global_edge_weight     = global_edge_weight or FloatDimension(1.0, 1.0))

        # Add variant-specific fields
        self.fields['variant']     = CategoricalDimension(['GAT'])
        self.fields['alpha']       = alpha
        self.fields['temperature'] = temperature
        self.fields['use_bias']    = use_bias
