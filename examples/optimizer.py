from __future__ import annotations
from typing import Dict, Optional

from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureProcessor
from ml_tools.optimizer.optimizer import Optimizer
from ml_tools.optimizer.optuna_strategy import OptunaStrategy
from ml_tools.optimizer.search_space import FloatDimension, IntDimension, CategoricalDimension, BoolDimension, ChoiceDimension
from ml_tools.optimizer.nn_search_space.nn_search_space import NNSearchSpace
from ml_tools.optimizer.nn_search_space.dense import Dense as DenseDim
from ml_tools.optimizer.nn_search_space.layer_sequence import LayerSequence as LayerSeqDim
from ml_tools.optimizer.nn_search_space.spatial_conv import SpatialConv as SpatialConvDim, \
                                                            SpatialMaxPool as SpatialMaxPoolDim


def build_dnn_optimizer(input_features: Dict[str, FeatureProcessor],
                        predicted_feature: str,
                        biasing_model: Optional[PredictionStrategy] = None) -> Optimizer:
    """Build an Optuna-backed optimizer for a DNN search space with variable depth.

    Chooses among LayerSequence lengths 1..5, each composed of Dense layers.
    """

    def dense_dim() -> DenseDim:
        return DenseDim(units      = IntDimension(16, 128),
                        activation = CategoricalDimension(["relu", "tanh"]))

    def make_dense_layer_sequence(n: int) -> LayerSeqDim:
        return LayerSeqDim(layers=[dense_dim() for _ in range(n)])

    dense_layers = ChoiceDimension({"layers_1": make_dense_layer_sequence(1),
                                    "layers_2": make_dense_layer_sequence(2),
                                    "layers_3": make_dense_layer_sequence(3),
                                    "layers_4": make_dense_layer_sequence(4),
                                    "layers_5": make_dense_layer_sequence(5),})

    dims = NNSearchSpace.Dimension(layers                = [dense_layers],
                                   initial_learning_rate = FloatDimension(1e-4, 1e-1, log=10),
                                   learning_decay_rate   = FloatDimension(0.1,   1.0          ),
                                   epoch_limit           = IntDimension(   200, 3000, log=2),
                                   convergence_criteria  = FloatDimension(1e-8, 1e-4, log=10),
                                   convergence_patience  = IntDimension(    50,  200          ),
                                   batch_size_log2       = IntDimension(     8,   11          ))

    search_space = NNSearchSpace(dims,
                                 input_features    = input_features,
                                 predicted_feature = predicted_feature,
                                 biasing_model     = biasing_model)

    return Optimizer(search_space=search_space, search_strategy=OptunaStrategy())


def build_cnn_optimizer(input_features: Dict[str, FeatureProcessor],
                        predicted_feature: str,
                        biasing_model: Optional[PredictionStrategy] = None) -> Optimizer:
    """Build an Optuna-backed optimizer for the CNN search space used in the example.
    """

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

    dims = NNSearchSpace.Dimension(layers                = [conv, pool],
                                   initial_learning_rate = FloatDimension(1e-4, 1e-1, log=10),
                                   learning_decay_rate   = FloatDimension(0.1,  1.0           ),
                                   epoch_limit           = IntDimension(   200, 3000, log=2),
                                   convergence_criteria  = FloatDimension(1e-8, 1e-4, log=10),
                                   convergence_patience  = IntDimension(    50,  200          ),
                                   batch_size_log2       = IntDimension(     8,   11          ))

    search_space = NNSearchSpace(dims,
                                 input_features    = input_features,
                                 predicted_feature = predicted_feature,
                                 biasing_model     = biasing_model)

    return Optimizer(search_space=search_space, search_strategy=OptunaStrategy())

