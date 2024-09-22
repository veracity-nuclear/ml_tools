import pytest
import os
import h5py
from math import isclose
import numpy as np
from ml_tools.model.state import State
from ml_tools.utils.h5_utils import get_groups_with_prefix

def test_state():

    data_file    = os.path.dirname(__file__)+"/test_data.h5"
    state_groups = get_groups_with_prefix(data_file, "set_", 1)
    features     = ["average_enrichment", "boron_concentration", "measured_fixed_detector"]
    states       = State.read_states_from_hdf5(data_file, features, state_groups)

    assert len(states) == 1

    state = states[0]

    assert len(state.feature("average_enrichment"))      == 9
    assert len(state.feature("boron_concentration"))     == 1
    assert len(state.feature("measured_fixed_detector")) == 7

    actual_average_enrichment      = np.nan_to_num(state.feature("average_enrichment"), nan=0.)
    actual_boron_concentration     = state.feature("boron_concentration")
    actual_measured_fixed_detector = state.feature("measured_fixed_detector")

    expected_average_enrichment      = [0., 0., 0., 0., 4.37, 4.59, 3.87, 3.95, 4.64]
    expected_boron_concentration     = [1439.72086910379]
    expected_measured_fixed_detector = [0.729616659550453, 1.08741813179624, 1.15960534546973, 1.16407884412748, 1.13889237591912, 1.05279742857571, 0.617740889411089]

    assert all(isclose(actual, expected) for actual, expected in zip(actual_average_enrichment,      expected_average_enrichment))
    assert all(isclose(actual, expected) for actual, expected in zip(actual_boron_concentration,     expected_boron_concentration))
    assert all(isclose(actual, expected) for actual, expected in zip(actual_measured_fixed_detector, expected_measured_fixed_detector))
