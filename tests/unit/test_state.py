import pytest
import os
import h5py
from numpy.testing import assert_allclose, assert_almost_equal
import numpy as np
import pandas as pd
import pandas.testing as pdt
from ml_tools import State, StateSeries, SeriesCollection
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

    actual_df   = SeriesCollection([StateSeries([state])]).to_dataframe(["average_enrichment", "boron_concentration"])
    expected_df = pd.DataFrame({"series_index": [0], "state_index": [0],
                                "average_enrichment_0": [   0], "average_enrichment_1": [   0], "average_enrichment_2": [   0],
                                "average_enrichment_3": [   0], "average_enrichment_4": [4.37], "average_enrichment_5": [4.59],
                                "average_enrichment_6": [3.87], "average_enrichment_7": [3.95], "average_enrichment_8": [4.64],
                                "boron_concentration": [1439.7208691037854]
    }).set_index(["series_index", "state_index"])
    pdt.assert_frame_equal(actual_df, expected_df, check_dtype=False)

    state = SeriesCollection.from_dataframe(SeriesCollection([StateSeries([state])]).to_dataframe(features))[0][0]
    actual_average_enrichment      = state["average_enrichment"]
    actual_boron_concentration     = state["boron_concentration"]
    actual_measured_fixed_detector = state["measured_fixed_detector"]
    assert_allclose(actual_average_enrichment,      expected_average_enrichment)
    assert_allclose(actual_boron_concentration,     expected_boron_concentration)
    assert_allclose(actual_measured_fixed_detector, expected_measured_fixed_detector)

    perturbators = {"measured_fixed_detector": RelativeNormalPerturbator(0.5),
                    "boron_concentration":     RelativeNormalPerturbator(0.2)}

    perturbed_states = State.perturb_states(perturbators, states)

    perturbed_boron_concentration     = perturbed_states[0]["boron_concentration"]
    perturbed_measured_fixed_detector = perturbed_states[0]["measured_fixed_detector"]

    assert not(np.allclose(actual_boron_concentration, perturbed_boron_concentration))
    assert not(np.allclose(actual_measured_fixed_detector, perturbed_measured_fixed_detector))

def test_state_mean_std_max_min():

    series = StateSeries([
        State({"feature_1": 10.0, "feature_2": 120.0}),
        State({"feature_1": 9.0, "feature_2": 115.0}),
        State({"feature_1": 12.0, "feature_2": 130.0}),
    ])

    # Test mean
    assert_almost_equal(series.mean(["feature_1"]), 10.333333333333334)
    assert_almost_equal(series.mean(["feature_2"]), 121.666666666666667)
    assert_almost_equal(series.mean(), [10.33333333333334, 121.666666666666667])

    # Test std
    assert_almost_equal(series.std(["feature_1"]), 1.247219129)
    assert_almost_equal(series.std(["feature_2"]), 6.236095645)
    assert_almost_equal(series.std(), [1.247219129, 6.236095645])

    # Test max
    assert_almost_equal(series.max(["feature_1"]), 12.0)
    assert_almost_equal(series.max(["feature_2"]), 130.0)
    assert_almost_equal(series.max(), [12.0, 130.0])

    # Test min
    assert_almost_equal(series.min(["feature_1"]), 9.0)
    assert_almost_equal(series.min(["feature_2"]), 115.0)
    assert_almost_equal(series.min(), [9.0, 115.0])

def test_state_series():
    def compare_series(series1, series2):
        assert len(series1) == len(series2)
        for i in range(len(series1)):
            assert series1[i].features == series2[i].features
            for feat in series1[i].features:
                assert_almost_equal(series1[i][feat], series2[i][feat])

    test_dir = os.path.join(os.path.dirname(__file__), "test_state")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    series = StateSeries([
        State({"feature1": 10.0, "feature2": 120.0}),
        State({"feature1": 9.0, "feature2": 115.0}),
        State({"feature1": 12.0, "feature2": 130.0}),
    ])

    # Test csv
    series.to_csv(os.path.join(test_dir, "test_state_series.csv"))
    loaded_series = StateSeries.from_csv(os.path.join(test_dir, "test_state_series.csv"), features=series.features)
    compare_series(series, loaded_series)

    # Test hdf5
    series.to_hdf5(os.path.join(test_dir, "test_state_series.h5"))
    loaded_series = StateSeries(State.read_states_from_hdf5(os.path.join(test_dir, "test_state_series.h5"), features=series.features))
    compare_series(series, loaded_series)

    # Test to_dataframe
    df = series.to_dataframe(features=series.features)
    expected_df = pd.DataFrame({
        "feature1": [10.0, 9.0, 12.0],
        "feature2": [120.0, 115.0, 130.0]
    })
    pdt.assert_frame_equal(df, expected_df, check_dtype=False)

    # Test from_dataframe
    loaded_series = StateSeries.from_dataframe(expected_df, features=series.features)
    compare_series(series, loaded_series)

    # Test to numpy
    np_array = series.to_numpy()
    expected_array = np.array([[10.0, 120.0],
                                [9.0, 115.0],
                                [12.0, 130.0]])
    assert np.array_equal(np_array, expected_array)

    # Cleanup
    os.remove(os.path.join(test_dir, "test_state_series.csv"))
    os.remove(os.path.join(test_dir, "test_state_series.h5"))
    os.rmdir(test_dir)

def test_state_vector_features():
    series = StateSeries([
        State({"feature1": [10.0, 20.0, 30.0, 40.0, 50.0], "feature2": [120.0, 121.0, 126.0, 132.0, 140.0]}),
        State({"feature1": [9.0, 19.0, 29.0, 39.0, 49.0], "feature2": [115.0, 116.0, 117.0, 118.0, 119.0]}),
        State({"feature1": [12.0, 22.0, 32.0, 42.0, 52.0], "feature2": [130.0, 131.0, 132.0, 133.0, 134.0]})
    ])

    expected_array = np.array([
        [
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [120.0, 121.0, 126.0, 132.0, 140.0]
        ],
        [
            [9.0, 19.0, 29.0, 39.0, 49.0],
            [115.0, 116.0, 117.0, 118.0, 119.0]
        ],
        [
            [12.0, 22.0, 32.0, 42.0, 52.0],
            [130.0, 131.0, 132.0, 133.0, 134.0]
        ]
    ])

    series_array = series.to_numpy()
    assert series_array.shape == expected_array.shape
    assert np.array_equal(series_array, expected_array)

def test_series_collection_random_sample():
    series1 = StateSeries([State({"x": 1.0})])
    series2 = StateSeries([State({"x": 2.0})])
    series3 = StateSeries([State({"x": 3.0})])
    series4 = StateSeries([State({"x": 4.0})])
    series5 = StateSeries([State({"x": 5.0})])

    full_collection = SeriesCollection([series1, series2, series3, series4, series5])
    sample_size = 3

    sampled = full_collection.random_sample(sample_size, seed=42)

    # Check sample size
    assert len(sampled) == sample_size

    # Check sampled elements are from original
    for s in sampled:
        assert s in full_collection.state_series_list

    # Check deterministic behavior with seed
    sampled_again = full_collection.random_sample(sample_size, seed=42)
    assert [s[0]["x"] for s in sampled] == [s[0]["x"] for s in sampled_again]