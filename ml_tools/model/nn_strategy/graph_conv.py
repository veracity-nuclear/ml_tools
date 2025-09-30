from __future__ import annotations
from typing import Any
from math import isclose
from decimal import Decimal
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf

from ml_tools.model.nn_strategy.layer import Layer, Activation
from ml_tools.model.nn_strategy.graph.graph import Graph
from ml_tools.model.nn_strategy.graph import load_graph_from_h5


@Layer.register_subclass("GraphConv")
class GraphConv(Layer):
    """GraphConv orchestrator that delegates to a Graph variant.

    Parameters
    ----------
    graph : Graph
        Fully-configured Graph variant (e.g., ``SAGE``) that owns spatial
        shape, units, connectivity, ordering, pre-node layers, and globals.
    activation : Activation, optional
        Post-propagation activation, by default ``'relu'``.
    dropout_rate : float, optional
        Dropout after activation, by default ``0.0``.
    batch_normalize : bool, optional
        Apply ``TimeDistributed(BatchNormalization)``; by default ``False``.
    layer_normalize : bool, optional
        Apply ``TimeDistributed(LayerNormalization)``; by default ``False``.

    Attributes
    ----------
    graph : Graph
        The current Graph variant used to perform message passing.
    activation : Activation
        Activation applied after the graph propagation.
    dropout_rate : float
        Dropout probability applied after activation.
    batch_normalize : bool
        Whether ``BatchNormalization`` is applied per time step.
    layer_normalize : bool
        Whether ``LayerNormalization`` is applied per time step.
    """

    @property
    def activation(self) -> Activation:
        return self._activation

    @activation.setter
    def activation(self, activation: Activation) -> None:
        self._activation = activation

    @property
    def graph(self) -> Graph:
        return self._graph

    @graph.setter
    def graph(self, value: Graph) -> None:
        assert value is not None, "graph cannot be None"
        assert isinstance(value, Graph), f"graph must be a Graph, got {type(value)}"
        self._graph = value

    def __init__(self,
                 graph:                    Graph,
                 activation:               Activation = 'relu',
                 dropout_rate:             float = 0.0,
                 batch_normalize:          bool = False,
                 layer_normalize:          bool = False) -> None:

        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.activation    = activation
        self.graph         = graph

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                (isinstance(other, GraphConv) and
                 isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                 self.activation      == other.activation and
                 self.batch_normalize == other.batch_normalize and
                 self.layer_normalize == other.layer_normalize and
                 self.graph           == other.graph))

    def __hash__(self) -> int:
        return hash((
            self.activation,
            Decimal(self.dropout_rate).quantize(Decimal('1e-9')),
            self.batch_normalize,
            self.layer_normalize,
            self.graph,
        ))

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        y = self.graph.build(input_tensor)

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
        group.create_dataset('activation_function', data=self.activation, dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate',        data=self.dropout_rate)
        group.create_dataset('batch_normalize',     data=self.batch_normalize)
        group.create_dataset('layer_normalize',     data=self.layer_normalize)
        g = group.create_group('graph')
        self.graph.save(g)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> GraphConv:
        activation    = group['activation_function'][()].decode('utf-8')
        dropout_rate  = float(group['dropout_rate'][()])
        batch_norm    = bool(group['batch_normalize'][()])
        layer_norm    = bool(group['layer_normalize'][()])
        graph         = load_graph_from_h5(group['graph'])
        return cls(activation=activation,
                   dropout_rate=dropout_rate,
                   batch_normalize=batch_norm,
                   layer_normalize=layer_norm,
                   graph=graph)
