import pytest
import os
import h5py
from math import isclose
import numpy as np

from ml_tools.model.state import State
from ml_tools.model.nn_strategy import NNStrategy, Dense, LayerSequence, CompoundLayer
from ml_tools.model.gbm_strategy import GBMStrategy
from ml_tools.model.pod_strategy import PODStrategy
from ml_tools.model.feature_processor import MinMaxNormalize, NoProcessing

input_features = {'average_exposure' : MinMaxNormalize(0., 45.),
                  'num_gad_rods'     : MinMaxNormalize(0., 12.),
                  'is_refl'          : NoProcessing()}
output_feature = "cips_index"

data_file                          = os.path.dirname(__file__)+"/test_data.h5"
state                              = State.read_state_from_hdf5(data_file, 'set_000001', ["2d_assembly_exposure", "num_gad_rods", "measured_rh_detector", "outputs/cips_index", "outputs/fine_detector"])
state.features["average_exposure"] = np.nan_to_num(state.feature("2d_assembly_exposure"), nan=0.)
state.features["num_gad_rods"]     = np.where(state.feature("num_gad_rods") == -1, 0., state.feature("num_gad_rods"))
state.features["is_refl"]          = np.where(np.isnan(state.feature("2d_assembly_exposure")), 1., 0.)

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
    assert(np.sum(row) > 0.0)
    row /= (cm_max - cm_min)
    fine_to_coarse_map.append(row)



def test_preprocess_inputs():

    cips_calculator = NNStrategy(input_features, output_feature)

    actual_values   = cips_calculator.preprocess_inputs([[state]])[0][0].tolist()
    expected_values = [0.0, 0.0, 0.0, 0.0, 0.8229136277763227, 0.6038608249885462, 0.9359057444294857, 0.0,                0.0,
                       1.0, 1.0, 1.0, 1.0, 0.0,                0.0,                0.0,                0.0,                0.0,
                       0.0, 0.0, 0.0, 0.0, 0.6666666666666666, 0.6666666666666666, 0.0,                0.6666666666666666, 1.0]

    assert(all(isclose(actual, expected) for actual, expected in zip(actual_values, expected_values)))


def test_gbm_strategy():

    cips_calculator = GBMStrategy(input_features, output_feature)
    cips_calculator.train([[state]]*5, [[state]]*5)

    assert(isclose(state.feature("cips_index")[0], cips_calculator.predict([[state]])[0], abs_tol=1E-5))

    cips_calculator.save_model('test_gbm_model.h5')

    new_cips_calculator = GBMStrategy.read_from_hdf5('test_gbm_model.h5')

    assert new_cips_calculator.boosting_type          == cips_calculator.boosting_type
    assert new_cips_calculator.objective              == cips_calculator.objective
    assert new_cips_calculator.metric                 == cips_calculator.metric
    assert new_cips_calculator.num_leaves             == cips_calculator.num_leaves
    assert new_cips_calculator.n_estimators           == cips_calculator.n_estimators
    assert new_cips_calculator.max_depth              == cips_calculator.max_depth
    assert new_cips_calculator.min_child_samples      == cips_calculator.min_child_samples
    assert new_cips_calculator.verbose                == cips_calculator.verbose
    assert new_cips_calculator.num_boost_round        == cips_calculator.num_boost_round
    assert new_cips_calculator.stopping_rounds        == cips_calculator.stopping_rounds
    assert isclose(new_cips_calculator.learning_rate,    cips_calculator.learning_rate)
    assert isclose(new_cips_calculator.subsample,        cips_calculator.subsample)
    assert isclose(new_cips_calculator.colsample_bytree, cips_calculator.colsample_bytree)
    assert isclose(new_cips_calculator.reg_alpha,        cips_calculator.reg_alpha)
    assert isclose(new_cips_calculator.reg_lambda,       cips_calculator.reg_lambda)

    assert isclose(state.feature("cips_index")[0], new_cips_calculator.predict([[state]])[0], abs_tol=1E-5)

    os.remove('test_gbm_model.h5')
    os.remove('test_gbm_model.lgbm')


def test_pod_strategy():

    cips_calculator = PODStrategy("measured_rh_detector", "fine_detector", np.asarray(fine_to_coarse_map))
    cips_calculator.train([[state]]*100)

    assert np.allclose(state.feature("fine_detector"), cips_calculator.predict([[state]])[0])



def test_nn_strategy_Dense():

    cips_calculator = NNStrategy(input_features, output_feature)
    cips_calculator.train([[state]]*1000)
    assert(isclose(state.feature("cips_index")[0], cips_calculator.predict([[state]])[0], abs_tol=1E-5))

    cips_calculator.save_model('test_nn_model.h5')
    new_cips_calculator = NNStrategy.read_from_hdf5('test_nn_model.h5')
    assert all(old_layer == new_layer for old_layer, new_layer in zip(cips_calculator.layers, new_cips_calculator.layers))
    assert cips_calculator.input_features.keys() == new_cips_calculator.input_features.keys()
    assert all(new_cips_calculator.input_features[feature] == cips_calculator.input_features[feature] for feature in cips_calculator.input_features.keys())
    assert new_cips_calculator.epoch_limit == cips_calculator.epoch_limit
    assert new_cips_calculator.batch_size  == cips_calculator.batch_size
    assert isclose(new_cips_calculator.initial_learning_rate, cips_calculator.initial_learning_rate)
    assert isclose(new_cips_calculator.learning_decay_rate,   cips_calculator.learning_decay_rate)
    assert isclose(new_cips_calculator.convergence_criteria,  cips_calculator.convergence_criteria)
    assert isclose(state.feature("cips_index")[0], new_cips_calculator.predict([[state]])[0], abs_tol=1E-5)


def test_nn_strategy_LayerSequence():

    layers          = [Dense(units=10, activation='relu'), LayerSequence(layers=[Dense(units=5, activation='relu'), Dense(units=10, activation='relu')])]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*1000)
    assert(isclose(state.feature("cips_index")[0], cips_calculator.predict([[state]])[0], abs_tol=1E-5))

    cips_calculator.save_model('test_nn_model.h5')
    new_cips_calculator = NNStrategy.read_from_hdf5('test_nn_model.h5')
    assert all(old_layer == new_layer for old_layer, new_layer in zip(cips_calculator.layers, new_cips_calculator.layers))
    assert isclose(state.feature("cips_index")[0], new_cips_calculator.predict([[state]])[0], abs_tol=1E-5)


def test_nn_strategy_CompoundLayer():

    layers          = [CompoundLayer(layers=[Dense(units=5, activation='relu'), Dense(units=10, activation='relu')], input_specifications=[slice(0, 9), slice(9, 19)])]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*1000)
    assert(isclose(state.feature("cips_index")[0], cips_calculator.predict([[state]])[0], abs_tol=1E-5))

    cips_calculator.save_model('test_nn_model.h5')
    new_cips_calculator = NNStrategy.read_from_hdf5('test_nn_model.h5')
    assert all(old_layer == new_layer for old_layer, new_layer in zip(cips_calculator.layers, new_cips_calculator.layers))
    assert isclose(state.feature("cips_index")[0], new_cips_calculator.predict([[state]])[0], abs_tol=1E-5)

    os.remove('test_nn_model.h5')
