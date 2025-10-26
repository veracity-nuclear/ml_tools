import pytest

from ml_tools.model import build_prediction_strategy
from ml_tools.model.feature_processor import NoProcessing
from ml_tools.model.nn_strategy.dense import Dense

from ml_tools.optimizer.optuna_strategy import OptunaStrategy
from ml_tools.optimizer.search_space import (
    StructDimension,
    FloatDimension,
    IntDimension,
    CategoricalDimension,
    ListDimension,
)
from ml_tools.optimizer.nn_search_space.nn_search_space import NNSearchSpace
from ml_tools.optimizer.nn_search_space.dense import Dense as DenseDim
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


def test_nn_optimization():
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

    # Assert hyperparameters
    assert model.initial_learning_rate == 0.001
    assert model.learning_decay_rate   == 1.0
    assert model.epoch_limit           == 100
    assert model.convergence_criteria  == pytest.approx(1e-6)
    assert model.convergence_patience  == 10
    assert model.batch_size            == 2 ** 4

    # Assert layers
    assert len(model.layers) == 2
    for layer in model.layers:
        assert isinstance(layer, Dense)
        assert layer.units      == 8
        assert layer.activation == "relu"
