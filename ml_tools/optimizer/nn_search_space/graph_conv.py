from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.nn_search_space.graph.graph import Graph
from ml_tools.optimizer.search_space import (
    FloatDimension,
    CategoricalDimension,
    BoolDimension,
)


class GraphConv(Layer):
    """Search-space dimension for GraphConv layer.

    Parameters
    ----------
    graph : Graph
        Graph configuration struct (e.g., an instance of graph.SAGE).
    activation : CategoricalDimension, optional
        Activation applied after graph propagation; default ['relu'].
    dropout_rate : FloatDimension, optional
        Dropout after activation; default 0.0.
    batch_normalize : BoolDimension, optional
        Apply TimeDistributed(BatchNormalization); default False.
    layer_normalize : BoolDimension, optional
        Apply TimeDistributed(LayerNormalization); default False.

    Attributes
    ----------
    graph : Graph
        Graph configuration.
    activation : CategoricalDimension
        Activation choices.
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
