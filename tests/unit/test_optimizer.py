import pytest

from ml_tools.model import build_prediction_strategy
from ml_tools import NNStrategy
from ml_tools.model.feature_processor import NoProcessing
from ml_tools.model.nn_strategy.dense import Dense
from ml_tools.model.nn_strategy.lstm import LSTM
from ml_tools.model.nn_strategy.transformer import Transformer
from ml_tools.model.nn_strategy.spatial_conv import SpatialConv, SpatialMaxPool
from ml_tools.model.nn_strategy.compound_layer import CompoundLayer
from ml_tools.model.nn_strategy.graph import SAGE, GAT
from ml_tools.model.nn_strategy.graph_conv import GraphConv

from ml_tools.optimizer.optuna_strategy import OptunaStrategy
from ml_tools.optimizer.search_space import (
    StructDimension,
    FloatDimension,
    IntDimension,
    CategoricalDimension,
    ListDimension,
    BoolDimension,
)
from ml_tools.optimizer.nn_search_space.nn_search_space import NNSearchSpace
from ml_tools.optimizer.nn_search_space.dense import Dense as DenseDim
from ml_tools.optimizer.nn_search_space.lstm import LSTM as LSTMDim
from ml_tools.optimizer.nn_search_space.transformer import Transformer as TransformerDim
from ml_tools.optimizer.nn_search_space.spatial_conv import SpatialConv as SpatialConvDim, SpatialMaxPool as SpatialMaxPoolDim
from ml_tools.optimizer.nn_search_space.compound_layer import CompoundLayer as CompoundLayerDim
from ml_tools.optimizer.nn_search_space.graph_conv import GraphConv as GraphConvDim
from ml_tools.optimizer.nn_search_space.graph.sage import SAGE as SAGEDim
from ml_tools.optimizer.nn_search_space.graph.gat import GAT as GATDim
from ml_tools.optimizer.search_space import StructDimension, BoolDimension
from ml_tools.optimizer.nn_search_space.layer_sequence import LayerSequence as LayerSequenceDim

class MockOptunaTrial:
    """ A mock optuna.trial.Trial.
    """

    def suggest_int(self, name, low, high, log=False, step=None):  # noqa: ARG002
        return low

    def suggest_float(self, name, low, high, log=False, step=None):  # noqa: ARG002
        return low

    def suggest_categorical(self, name, choices):  # noqa: ARG002
        return choices[0]


def test_nn_optimizer_LayerSequence():
    dense_layer = DenseDim(units      = IntDimension(8, 20),
                           activation = CategoricalDimension(["relu", "tanh", "sigmoid"]))

    search_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [dense_layer, dense_layer],
                                                         initial_learning_rate = FloatDimension(0.001, 0.1, log=True),
                                                         learning_decay_rate   = FloatDimension(1.0, 2.0),
                                                         epoch_limit           = IntDimension(100, 10000, log=True),
                                                         convergence_criteria  = FloatDimension(1e-6, 1e-5, log=True),
                                                         convergence_patience  = IntDimension(10, 20),
                                                         batch_size_log2       = IntDimension(4, 8)))

    strategy = OptunaStrategy()
    params   = strategy._get_sample(MockOptunaTrial(), search_space.dimensions)
    model    = build_prediction_strategy(strategy_type     = "NNStrategy",
                                         dict              = params,
                                         input_features    = {"x": NoProcessing()},
                                         predicted_feature = "y",
                                         biasing_model     = None)

    expected = NNStrategy(input_features        = {"x": NoProcessing()},
                          predicted_feature     = "y",
                          layers                = [Dense(units=8, activation="relu"), Dense(units=8, activation="relu")],
                          initial_learning_rate = 0.001,
                          learning_decay_rate   = 1.0,
                          epoch_limit           = 100,
                          convergence_criteria  = 1e-6,
                          convergence_patience  = 10,
                          batch_size            = 2 ** 4)
    assert model == expected


def test_nn_optimizer_Dense():
    dense_layer = DenseDim(units      = IntDimension(8, 20),
                           activation = CategoricalDimension(["relu", "tanh", "sigmoid"]))

    search_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [dense_layer],
                                                         initial_learning_rate = FloatDimension(0.001, 0.1, log=True),
                                                         learning_decay_rate   = FloatDimension(1.0, 2.0),
                                                         epoch_limit           = IntDimension(100, 10000, log=True),
                                                         convergence_criteria  = FloatDimension(1e-6, 1e-5, log=True),
                                                         convergence_patience  = IntDimension(10, 20),
                                                         batch_size_log2       = IntDimension(4, 8)))

    strategy = OptunaStrategy()
    params   = strategy._get_sample(MockOptunaTrial(), search_space.dimensions)
    model    = build_prediction_strategy(strategy_type     = "NNStrategy",
                                         dict              = params,
                                         input_features    = {"x": NoProcessing()},
                                         predicted_feature = "y",
                                         biasing_model     = None)

    expected = NNStrategy(input_features        = {"x": NoProcessing()},
                          predicted_feature     = "y",
                          layers                = [Dense(units=8, activation="relu")],
                          initial_learning_rate = 0.001,
                          learning_decay_rate   = 1.0,
                          epoch_limit           = 100,
                          convergence_criteria  = 1e-6,
                          convergence_patience  = 10,
                          batch_size            = 2 ** 4)
    assert model == expected


def test_nn_optimizer_LSTM():
    lstm_layer = LSTMDim(units      = IntDimension(5, 10),
                         activation = CategoricalDimension(["relu", "tanh"]))

    search_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [lstm_layer],
                                                         initial_learning_rate = FloatDimension(0.001, 0.1, log=True),
                                                         learning_decay_rate   = FloatDimension(1.0, 2.0),
                                                         epoch_limit           = IntDimension(100, 10000, log=True),
                                                         convergence_criteria  = FloatDimension(1e-6, 1e-5, log=True),
                                                         convergence_patience  = IntDimension(10, 20),
                                                         batch_size_log2       = IntDimension(4, 8)))

    strategy = OptunaStrategy()
    params   = strategy._get_sample(MockOptunaTrial(), search_space.dimensions)
    model    = build_prediction_strategy(strategy_type     = "NNStrategy",
                                         dict              = params,
                                         input_features    = {"x": NoProcessing()},
                                         predicted_feature = "y",
                                         biasing_model     = None)

    expected = NNStrategy(input_features        = {"x": NoProcessing()},
                          predicted_feature     = "y",
                          layers                = [LSTM(units=5, activation='relu')],
                          initial_learning_rate = 0.001,
                          learning_decay_rate   = 1.0,
                          epoch_limit           = 100,
                          convergence_criteria  = 1e-6,
                          convergence_patience  = 10,
                          batch_size            = 2 ** 4)
    assert model == expected


def test_nn_optimizer_Transformer():
    transformer = TransformerDim(num_heads  = IntDimension(2, 4),
                                 model_dim  = IntDimension(27, 64),
                                 ff_dim     = IntDimension(50, 128),
                                 activation = CategoricalDimension(["relu", "tanh"]))

    search_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [transformer],
                                                         initial_learning_rate = FloatDimension(0.001, 0.1, log=True),
                                                         learning_decay_rate   = FloatDimension(1.0, 2.0),
                                                         epoch_limit           = IntDimension(100, 10000, log=True),
                                                         convergence_criteria  = FloatDimension(1e-6, 1e-5, log=True),
                                                         convergence_patience  = IntDimension(10, 20),
                                                         batch_size_log2       = IntDimension(4, 8)))

    strategy = OptunaStrategy()
    params   = strategy._get_sample(MockOptunaTrial(), search_space.dimensions)
    model    = build_prediction_strategy(strategy_type     = "NNStrategy",
                                         dict              = params,
                                         input_features    = {"x": NoProcessing()},
                                         predicted_feature = "y",
                                         biasing_model     = None)

    expected = NNStrategy(input_features        = {"x": NoProcessing()},
                          predicted_feature     = "y",
                          layers                = [Transformer(num_heads=2, model_dim=27, ff_dim=50, activation='relu')],
                          initial_learning_rate = 0.001,
                          learning_decay_rate   = 1.0,
                          epoch_limit           = 100,
                          convergence_criteria  = 1e-6,
                          convergence_patience  = 10,
                          batch_size            = 2 ** 4)
    assert model == expected


def test_nn_optimizer_CNN():
    conv = SpatialConvDim(input_shape = CategoricalDimension([(3, 3)]),
                          activation  = CategoricalDimension(["relu", "tanh"]),
                          filters     = IntDimension(4, 8),
                          kernel_size = CategoricalDimension([(2, 2)]),
                          strides     = CategoricalDimension([(1, 1)]),
                          padding     = BoolDimension([False]))

    pool = SpatialMaxPoolDim(input_shape = CategoricalDimension([(3, 3)]),
                             pool_size   = CategoricalDimension([(2, 2)]),
                             strides     = CategoricalDimension([(1, 1)]),
                             padding     = BoolDimension([False]))

    search_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [conv, pool],
                                                         initial_learning_rate = FloatDimension(0.001, 0.1, log=True),
                                                         learning_decay_rate   = FloatDimension(1.0, 2.0),
                                                         epoch_limit           = IntDimension(100, 10000, log=True),
                                                         convergence_criteria  = FloatDimension(1e-6, 1e-5, log=True),
                                                         convergence_patience  = IntDimension(10, 20),
                                                         batch_size_log2       = IntDimension(4, 8)))

    strategy = OptunaStrategy()
    params   = strategy._get_sample(MockOptunaTrial(), search_space.dimensions)
    model    = build_prediction_strategy(strategy_type     = "NNStrategy",
                                         dict              = params,
                                         input_features    = {"x": NoProcessing()},
                                         predicted_feature = "y",
                                         biasing_model     = None)

    expected = NNStrategy(input_features        = {"x": NoProcessing()},
                          predicted_feature     = "y",
                          layers                = [SpatialConv(input_shape=(3,3), kernel_size=(2,2), filters=4, activation='relu', padding=False),
                                                   SpatialMaxPool(input_shape=(3,3), pool_size=(2,2), padding=False)],
                          initial_learning_rate = 0.001,
                          learning_decay_rate   = 1.0,
                          epoch_limit           = 100,
                          convergence_criteria  = 1e-6,
                          convergence_patience  = 10,
                          batch_size            = 2 ** 4)
    assert model == expected


def test_nn_optimizer_CompoundLayer():
    branch1 = DenseDim(units=IntDimension(5, 10), activation=CategoricalDimension(["relu"]))
    branch2 = DenseDim(units=IntDimension(10, 20), activation=CategoricalDimension(["relu"]))

    compound_layer = CompoundLayerDim(layers=[branch1, branch2],
                                      input_specifications=[slice(0, 9), slice(9, 19)])

    search_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [compound_layer],
                                                         initial_learning_rate = FloatDimension(0.001, 0.1, log=True),
                                                         learning_decay_rate   = FloatDimension(1.0, 2.0),
                                                         epoch_limit           = IntDimension(100, 10000, log=True),
                                                         convergence_criteria  = FloatDimension(1e-6, 1e-5, log=True),
                                                         convergence_patience  = IntDimension(10, 20),
                                                         batch_size_log2       = IntDimension(4, 8)))

    strategy = OptunaStrategy()
    params   = strategy._get_sample(MockOptunaTrial(), search_space.dimensions)
    model    = build_prediction_strategy(strategy_type     = "NNStrategy",
                                         dict              = params,
                                         input_features    = {"x": NoProcessing()},
                                         predicted_feature = "y",
                                         biasing_model     = None)

    expected = NNStrategy(input_features        = {"x": NoProcessing()},
                          predicted_feature     = "y",
                          layers                = [CompoundLayer(layers=[Dense(units=5, activation='relu'),
                                                                         Dense(units=10, activation='relu')],
                                                                 input_specifications=[slice(0, 9), slice(9, 19)])],
                          initial_learning_rate = 0.001,
                          learning_decay_rate   = 1.0,
                          epoch_limit           = 100,
                          convergence_criteria  = 1e-6,
                          convergence_patience  = 10,
                          batch_size            = 2 ** 4)
    assert model == expected


def test_nn_optimizer_GNN_SAGE():
    graph = SAGEDim(input_shape  = CategoricalDimension([(3, 3)]),
                    units        = IntDimension(4, 8),
                    connectivity = CategoricalDimension(['2d-4']),
                    aggregator   = CategoricalDimension(['mean']))

    layer = GraphConvDim(graph=graph, activation=CategoricalDimension(['relu']))

    search_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [layer],
                                                         initial_learning_rate = FloatDimension(0.001, 0.1, log=True),
                                                         learning_decay_rate   = FloatDimension(1.0, 2.0),
                                                         epoch_limit           = IntDimension(100, 10000, log=True),
                                                         convergence_criteria  = FloatDimension(1e-6, 1e-5, log=True),
                                                         convergence_patience  = IntDimension(10, 20),
                                                         batch_size_log2       = IntDimension(4, 8)))

    strategy = OptunaStrategy()
    params   = strategy._get_sample(MockOptunaTrial(), search_space.dimensions)
    model    = build_prediction_strategy(strategy_type     = "NNStrategy",
                                         dict              = params,
                                         input_features    = {"x": NoProcessing()},
                                         predicted_feature = "y",
                                         biasing_model     = None)

    expected_graph = SAGE(input_shape              = (3, 3),
                          units                    = 4,
                          ordering                 = 'feature_major',
                          spatial_feature_size     = None,
                          global_feature_count     = 0,
                          connectivity             = '2d-4',
                          self_loops               = False,
                          normalize                = False,
                          distance_weighted        = False,
                          connect_global_to_all    = False,
                          connect_global_to_global = False,
                          global_edge_weight       = 1.0,
                          aggregator               = 'mean',
                          use_bias                 = False)

    expected = NNStrategy(input_features        = {"x": NoProcessing()},
                          predicted_feature     = "y",
                          layers                = [GraphConv(graph      = expected_graph,
                                                             activation = 'relu')],
                          initial_learning_rate = 0.001,
                          learning_decay_rate   = 1.0,
                          epoch_limit           = 100,
                          convergence_criteria  = 1e-6,
                          convergence_patience  = 10,
                          batch_size            = 2 ** 4)
    assert model == expected


def test_nn_optimizer_GNN_GAT():
    graph = GATDim(input_shape   = CategoricalDimension([(3, 3)]),
                   units         = IntDimension(4, 8),
                   connectivity  = CategoricalDimension(['2d-4']),
                   alpha         = FloatDimension(0.2, 0.2),
                   temperature   = FloatDimension(1.0, 1.0),
                   use_bias      = BoolDimension([False]))

    layer = GraphConvDim(graph=graph, activation=CategoricalDimension(['relu']))

    search_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [layer],
                                                         initial_learning_rate = FloatDimension(0.001, 0.1, log=True),
                                                         learning_decay_rate   = FloatDimension(1.0, 2.0),
                                                         epoch_limit           = IntDimension(100, 10000, log=True),
                                                         convergence_criteria  = FloatDimension(1e-6, 1e-5, log=True),
                                                         convergence_patience  = IntDimension(10, 20),
                                                         batch_size_log2       = IntDimension(4, 8)))

    strategy = OptunaStrategy()
    params   = strategy._get_sample(MockOptunaTrial(), search_space.dimensions)
    model    = build_prediction_strategy(strategy_type     = "NNStrategy",
                                         dict              = params,
                                         input_features    = {"x": NoProcessing()},
                                         predicted_feature = "y",
                                         biasing_model     = None)

    expected_graph = GAT(input_shape              = (3, 3),
                         units                    = 4,
                         ordering                 = 'feature_major',
                         spatial_feature_size     = None,
                         global_feature_count     = 0,
                         connectivity             = '2d-4',
                         self_loops               = False,
                         normalize                = False,
                         distance_weighted        = False,
                         connect_global_to_all    = False,
                         connect_global_to_global = False,
                         global_edge_weight       = 1.0,
                         alpha                    = 0.2,
                         temperature              = 1.0,
                         use_bias                 = False)

    expected = NNStrategy(input_features        = {"x": NoProcessing()},
                          predicted_feature     = "y",
                          layers                = [GraphConv(graph=expected_graph, activation='relu')],
                          initial_learning_rate = 0.001,
                          learning_decay_rate   = 1.0,
                          epoch_limit           = 100,
                          convergence_criteria  = 1e-6,
                          convergence_patience  = 10,
                          batch_size            = 2 ** 4)
    assert model == expected
