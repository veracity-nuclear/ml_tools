import pytest
import os
import glob
import h5py
from numpy.testing import assert_allclose
import numpy as np
from ml_tools.model import build_prediction_strategy

from ml_tools.model.nn_strategy import Dense, LSTM, Transformer, SpatialConv, SpatialMaxPool, PassThrough, LayerSequence, CompoundLayer, GraphConv
from ml_tools.model.nn_strategy.graph import SAGE, GAT
from ml_tools import State, NNStrategy, GBMStrategy, PODStrategy, MinMaxNormalize, NoProcessing

input_features = {'average_exposure' : MinMaxNormalize(0., 45.),
                  'is_refl'          : NoProcessing(),
                  'num_gad_rods'     : MinMaxNormalize(0., 12.)}
output_feature = "cips_index"

data_file                          = os.path.dirname(__file__)+"/test_data.h5"
state                              = State.from_hdf5(data_file, 'set_000001', ["2d_assembly_exposure", "num_gad_rods", "measured_rh_detector", "outputs/cips_index", "outputs/fine_detector"])
state.features["average_exposure"] = np.nan_to_num(state["2d_assembly_exposure"], nan=0.)
state.features["num_gad_rods"]     = np.where(state["num_gad_rods"] == -1, 0., state["num_gad_rods"])
state.features["is_refl"]          = np.where(np.isnan(state["2d_assembly_exposure"]), 1., 0.)

with h5py.File(data_file, 'r') as h5_file:
    fine_mesh   = h5_file["mesh"]["fine_mesh"][()]
    coarse_mesh = h5_file["mesh"]["rhodium_mesh"][()]

fine_to_coarse_map = []
for cm_min, cm_max in coarse_mesh:
    row = np.zeros(len(fine_mesh) - 1)
    for i in range(len(fine_mesh) - 1):
        dx = fine_mesh[i+1] - fine_mesh[i]
        if fine_mesh[i] >= cm_min and fine_mesh[i + 1] <= cm_max:
            row[i] = dx
    assert np.sum(row) > 0.0
    row /= (cm_max - cm_min)
    fine_to_coarse_map.append(row)


def assert_nn_equal(lhs: NNStrategy, rhs: NNStrategy) -> None:
    """Reusable assertions for NNStrategy objects across read_from_file and dict round-trips."""
    assert len(rhs.layers) == len(lhs.layers)
    assert all(a                     == b for a, b in zip(lhs.layers, rhs.layers))
    assert lhs.input_features.keys() == rhs.input_features.keys()
    assert all(rhs.input_features[f] == lhs.input_features[f] for f in lhs.input_features.keys())
    assert rhs.epoch_limit           == lhs.epoch_limit
    assert rhs.batch_size            == lhs.batch_size
    assert_allclose(rhs.initial_learning_rate, lhs.initial_learning_rate)
    assert_allclose(rhs.learning_decay_rate, lhs.learning_decay_rate)
    assert_allclose(rhs.convergence_criteria, lhs.convergence_criteria)



def test_preprocess_inputs():

    cips_calculator = NNStrategy(input_features, output_feature)

    actual_values   = cips_calculator.preprocess_inputs([[state]])[0][0].tolist()
    expected_values = [0.0, 0.0, 0.0, 0.0, 0.8229136277763227, 0.6038608249885462, 0.9359057444294857, 0.0,                0.0,
                       1.0, 1.0, 1.0, 1.0, 0.0,                0.0,                0.0,                0.0,                0.0,
                       0.0, 0.0, 0.0, 0.0, 0.6666666666666666, 0.6666666666666666, 0.0,                0.6666666666666666, 1.0]
    assert_allclose(actual_values, expected_values)


def test_gbm_strategy():

    cips_calculator = GBMStrategy(input_features, output_feature)
    cips_calculator.train([[state]]*5, [[state]]*5)

    assert_allclose(state["cips_index"], cips_calculator.predict([[state]])[0][0], atol=1E-5)

    cips_calculator.save_model('test_gbm_model.h5')

    def assert_gbm_equal(lhs: GBMStrategy, rhs: GBMStrategy) -> None:
        assert rhs.boosting_type          == lhs.boosting_type
        assert rhs.objective              == lhs.objective
        assert rhs.metric                 == lhs.metric
        assert rhs.num_leaves             == lhs.num_leaves
        assert rhs.n_estimators           == lhs.n_estimators
        assert rhs.max_depth              == lhs.max_depth
        assert rhs.min_child_samples      == lhs.min_child_samples
        assert rhs.verbose                == lhs.verbose
        assert rhs.num_boost_round        == lhs.num_boost_round
        assert rhs.stopping_rounds        == lhs.stopping_rounds
        assert_allclose(rhs.learning_rate,    lhs.learning_rate)
        assert_allclose(rhs.subsample,        lhs.subsample)
        assert_allclose(rhs.colsample_bytree, lhs.colsample_bytree)
        assert_allclose(rhs.reg_alpha,        lhs.reg_alpha)
        assert_allclose(rhs.reg_lambda,       lhs.reg_lambda)

    new_cips_calculator = GBMStrategy.read_from_file('test_gbm_model.h5')
    assert_gbm_equal(cips_calculator, new_cips_calculator)
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]])[0][0], atol=1E-5)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'GBMStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_gbm_equal(cips_calculator, new_cips_calculator)

    os.remove('test_gbm_model.h5')
    os.remove('test_gbm_model.lgbm')


def test_pod_strategy():

    detector_predictor = PODStrategy("measured_rh_detector", "fine_detector", np.asarray(fine_to_coarse_map))
    detector_predictor.train([[state]]*100)

    assert np.allclose(state["fine_detector"], detector_predictor.predict([[state]])[0][0], atol=1E-2)

    new_detector_predictor = build_prediction_strategy(strategy_type     = 'PODStrategy',
                                                       dict              = detector_predictor.to_dict(),
                                                       input_features    = {'measured_rh_detector': NoProcessing()},
                                                       predicted_feature ='fine_detector',
                                                       biasing_model     = None)
    assert new_detector_predictor.input_feature == detector_predictor.input_feature
    assert new_detector_predictor.nclusters     == detector_predictor.nclusters
    assert new_detector_predictor.max_svd_size  == detector_predictor.max_svd_size
    assert new_detector_predictor.ndims         == detector_predictor.ndims
    assert np.allclose(new_detector_predictor.fine_to_coarse_map, detector_predictor.fine_to_coarse_map)



def test_nn_strategy_Dense():

    cips_calculator = NNStrategy(input_features, output_feature)
    cips_calculator.train([[state]]*1000)
    assert_allclose(state["cips_index"], cips_calculator.predict([[state]])[0][0], atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert_nn_equal(cips_calculator, new_cips_calculator)
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]])[0][0], atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_nn_equal(cips_calculator, new_cips_calculator)


def test_nn_strategy_LSTM():

    layers = [LSTM(units=5, activation='relu')]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]*100]*1000)
    assert_allclose(state["cips_index"], cips_calculator.predict([[state]*100])[0][-1], atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert_nn_equal(cips_calculator, new_cips_calculator)
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]*100])[0][-1], atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_nn_equal(cips_calculator, new_cips_calculator)


def test_nn_strategy_Transformer():

    layers = [Transformer(num_heads=2, model_dim=27, ff_dim=50)]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]*100]*1000)
    assert_allclose(state["cips_index"], cips_calculator.predict([[state]*100])[0][-1], atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert_nn_equal(cips_calculator, new_cips_calculator)
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]*100])[0][-1], atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_nn_equal(cips_calculator, new_cips_calculator)


def test_nn_strategy_CNN():

    layers = [SpatialConv(input_shape=(3,3), kernel_size=(2,2), filters=4),
              SpatialMaxPool(input_shape=(3,3), pool_size=(2,2), padding=False)]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*100)
    assert_allclose(state["cips_index"], cips_calculator.predict([[state]])[0][0], atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert_nn_equal(cips_calculator, new_cips_calculator)
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]])[0][0], atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_nn_equal(cips_calculator, new_cips_calculator)


def test_nn_strategy_LayerSequence():

    layers          = [PassThrough(), Dense(units=10, activation='relu'), LayerSequence(layers=[Dense(units=5, activation='relu'), Dense(units=10, activation='relu')])]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*1000)
    assert_allclose(state["cips_index"], cips_calculator.predict([[state]])[0][0], atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert_nn_equal(cips_calculator, new_cips_calculator)
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]])[0][0], atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_nn_equal(cips_calculator, new_cips_calculator)


def test_nn_strategy_CompoundLayer():

    layers          = [CompoundLayer(layers=[Dense(units=5, activation='relu'), Dense(units=10, activation='relu')], input_specifications=[slice(0, 9), slice(9, 19)])]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*1000)
    assert_allclose(state["cips_index"], cips_calculator.predict([[state]])[0][0], atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert_nn_equal(cips_calculator, new_cips_calculator)
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]])[0][0], atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_nn_equal(cips_calculator, new_cips_calculator)

def test_nn_strategy_GNN_SAGE():

    graph = SAGE(input_shape=(3, 3), units=4, connectivity='2d-4', aggregator='mean')
    layers = [GraphConv(graph=graph, activation='relu')]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*100)
    assert_allclose(state["cips_index"], cips_calculator.predict([[state]])[0][0], atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert all(old_layer == new_layer for old_layer, new_layer in zip(cips_calculator.layers, new_cips_calculator.layers))
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]])[0][0], atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_nn_equal(cips_calculator, new_cips_calculator)


def test_nn_strategy_GNN_GAT():

    graph = GAT(input_shape=(3, 3), units=4, connectivity='2d-4', alpha=0.2, temperature=1.0)
    layers = [GraphConv(graph=graph, activation='relu')]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*100)
    assert_allclose(state["cips_index"], cips_calculator.predict([[state]])[0][0], atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert all(old_layer == new_layer for old_layer, new_layer in zip(cips_calculator.layers, new_cips_calculator.layers))
    assert_allclose(state["cips_index"], new_cips_calculator.predict([[state]])[0][0], atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    dict              = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_feature = output_feature,
                                                    biasing_model     = None)
    assert_nn_equal(cips_calculator, new_cips_calculator)

    for file in glob.glob('test_nn_model.*'):
        os.remove(file)
