from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.nn_search_space.graph.graph import Graph
from ml_tools.optimizer.search_space import (
    FloatDimension,
    CategoricalDimension,
    BoolDimension,
)


class GraphConv(Layer):
    """Search-space dimension for a GraphConv layer (domains, not values).

    Parameters
    ----------
    graph : Graph
        Graph search-space struct describing graph connectivity/encoding domains
        (e.g., an instance of graph.SAGE or graph.GAT).
    activation : CategoricalDimension, optional
        Domain of activation choices applied after propagation (default ['relu']).
    dropout_rate : FloatDimension, optional
        Domain for dropout after activation (default 0.0).
    batch_normalize : BoolDimension, optional
        Domain for whether to apply TimeDistributed(BatchNormalization) (default False).
    layer_normalize : BoolDimension, optional
        Domain for whether to apply TimeDistributed(LayerNormalization) (default False).

    Attributes
    ----------
    graph : Graph
        Domain struct defining graph-related sub-dimensions.
    activation : CategoricalDimension
        Domain for activation choices.
    """

    @property
    def graph(self) -> Graph:
        return self.fields["graph"]

    @graph.setter
    def graph(self, value: Graph) -> None:
        assert isinstance(value, Graph)
        self.fields["graph"] = value

    @property
    def activation(self) -> CategoricalDimension:
        return self.fields["activation"]

    @activation.setter
    def activation(self, value: CategoricalDimension) -> None:
        self.fields["activation"] = value

    def __init__(self,
                 graph:           Graph,
                 activation:      CategoricalDimension = CategoricalDimension(['relu']),
                 dropout_rate:    FloatDimension       = FloatDimension(0.0, 0.0),
                 batch_normalize: BoolDimension        = BoolDimension([False]),
                 layer_normalize: BoolDimension        = BoolDimension([False])) -> None:
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.graph      = graph
        self.activation = activation
