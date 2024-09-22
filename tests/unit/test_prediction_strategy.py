import pytest
import os
import h5py
from math import isclose
import numpy as np

from ml_tools.model.state import State
from ml_tools.model.nn_strategy import NNStrategy
from ml_tools.model.feature_processor import MinMaxNormalize, NoProcessing

input_features = {'average_exposure' : MinMaxNormalize(0., 45.),
                  'num_gad_rods'         : MinMaxNormalize(0., 12.),
                  'is_refl'              : NoProcessing()}
output_feature = "cips_index"

data_file                          = os.path.dirname(__file__)+"/test_data.h5"
state                              = State.read_state_from_hdf5(data_file, 'set_000001', ["2d_assembly_exposure", "num_gad_rods", "outputs/cips_index"])
state.features["average_exposure"] = np.nan_to_num(state.feature("2d_assembly_exposure"), nan=0.)
state.features["num_gad_rods"]     = np.where(state.feature("num_gad_rods") == -1, 0., state.feature("num_gad_rods"))
state.features["is_refl"]          = np.where(np.isnan(state.feature("2d_assembly_exposure")), 1., 0.)


def test_preprocess_inputs():

    cips_calculator = NNStrategy(input_features, output_feature)

    actual_values   = cips_calculator.preprocess_inputs([state])[0].tolist()
    expected_values = [0.0, 0.0, 0.0, 0.0, 0.8229136277763227, 0.6038608249885462, 0.9359057444294857, 0.0,                0.0,
                       1.0, 1.0, 1.0, 1.0, 0.0,                0.0,                0.0,                0.0,                0.0,
                       0.0, 0.0, 0.0, 0.0, 0.6666666666666666, 0.6666666666666666, 0.0,                0.6666666666666666, 1.0]

    assert(all(isclose(actual, expected) for actual, expected in zip(actual_values, expected_values)))


def test_nn_strategy():

    cips_calculator = NNStrategy(input_features, output_feature)

    cips_calculator.epoch_limit = 400
    cips_calculator.convergence_criteria = 1E-14
    cips_calculator.train([state]*5)

    assert(isclose(state.feature("cips_index"), cips_calculator.predict([state])[0], abs_tol=1E-5))

    cips_calculator.save_model('test_nn_model.h5')

    new_cips_calculator = NNStrategy.read_from_hdf5('test_nn_model.h5')

    assert(all(old_layer == new_layer for old_layer, new_layer in zip(cips_calculator.hidden_layers, new_cips_calculator.hidden_layers)))
    assert(cips_calculator.input_features.keys() == new_cips_calculator.input_features.keys())
    assert(all(new_cips_calculator.input_features[feature] == cips_calculator.input_features[feature] for feature in cips_calculator.input_features.keys()))
    assert(new_cips_calculator.epoch_limit == cips_calculator.epoch_limit)
    assert(new_cips_calculator.batch_size  == cips_calculator.batch_size)
    assert(isclose(new_cips_calculator.initial_learning_rate, cips_calculator.initial_learning_rate))
    assert(isclose(new_cips_calculator.learning_decay_rate,   cips_calculator.learning_decay_rate))
    assert(isclose(new_cips_calculator.convergence_criteria,  cips_calculator.convergence_criteria))

    assert(isclose(state.feature("cips_index"), new_cips_calculator.predict([state])[0], abs_tol=1E-5))

    os.remove('test_nn_model.h5')