import pytest
import os
import h5py
from numpy.testing import assert_allclose
import numpy as np
import pandas as pd
import pandas.testing as pdt
from ml_tools.model.state import series_to_pandas
from ml_tools.utils.h5_utils import get_groups_with_prefix
from ml_tools import State, RelativeNormalPerturbator

def test_state():

    data_file    = os.path.dirname(__file__)+"/test_data.h5"
    state_groups = get_groups_with_prefix(data_file, "set_", 1)
    features     = ["average_enrichment", "boron_concentration", "measured_fixed_detector"]
    states       = State.read_states_from_hdf5(data_file, features, state_groups)

    assert len(states) == 1

    state = states[0]
    state["average_enrichment"] = np.nan_to_num(state["average_enrichment"], nan=0.)

    assert len(state["average_enrichment"])      == 9
    assert len(state["boron_concentration"])     == 1
    assert len(state["measured_fixed_detector"]) == 7

    actual_average_enrichment      = state["average_enrichment"]
    actual_boron_concentration     = state["boron_concentration"]
    actual_measured_fixed_detector = state["measured_fixed_detector"]

    expected_average_enrichment      = [0., 0., 0., 0., 4.37, 4.59, 3.87, 3.95, 4.64]
    expected_boron_concentration     = [1439.7208691037854]
    expected_measured_fixed_detector = [0.729616659550453, 1.08741813179624, 1.15960534546973, 1.16407884412748, 1.13889237591912, 1.05279742857571, 0.617740889411089]

    assert_allclose(actual_average_enrichment,      expected_average_enrichment)
    assert_allclose(actual_boron_concentration,     expected_boron_concentration)
    assert_allclose(actual_measured_fixed_detector, expected_measured_fixed_detector)

    actual_df   = series_to_pandas([[state]], ["average_enrichment", "boron_concentration"])
    expected_df = pd.DataFrame({"series_index": [0], "state_index": [0],
                                "average_enrichment_0": [   0], "average_enrichment_1": [   0], "average_enrichment_2": [   0],
                                "average_enrichment_3": [   0], "average_enrichment_4": [4.37], "average_enrichment_5": [4.59],
                                "average_enrichment_6": [3.87], "average_enrichment_7": [3.95], "average_enrichment_8": [4.64],
                                "boron_concentration": [1439.7208691037854]
    }).set_index(["series_index", "state_index"])
    pdt.assert_frame_equal(actual_df, expected_df, check_dtype=False)

    perturbators = {"measured_fixed_detector": RelativeNormalPerturbator(0.5),
                    "boron_concentration":     RelativeNormalPerturbator(0.2)}

    perturbed_states = State.perturb_states(perturbators, states)

    perturbed_boron_concentration     = perturbed_states[0]["boron_concentration"]
    perturbed_measured_fixed_detector = perturbed_states[0]["measured_fixed_detector"]

    assert not(np.allclose(actual_boron_concentration, perturbed_boron_concentration))
    assert not(np.allclose(actual_measured_fixed_detector, perturbed_measured_fixed_detector))
