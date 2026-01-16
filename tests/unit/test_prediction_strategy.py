import pytest
import os
import glob
import tempfile
import h5py
from numpy.testing import assert_allclose
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from ml_tools.model import build_prediction_strategy

from ml_tools.model.nn_strategy import Dense, LSTM, Transformer, SpatialConv, SpatialMaxPool, PassThrough, LayerSequence, CompoundLayer, GraphConv
from ml_tools.model.nn_strategy.graph import SAGE, GAT
from ml_tools import State, NNStrategy, GBMStrategy, PODStrategy, MinMaxNormalize, NoProcessing, StateSeries, SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.sklearn_strategy import SklearnStrategy

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



def test_preprocess_features():

    actual_values   = PredictionStrategy.preprocess_features([[state]], input_features)[0][0].tolist()
    expected_values = [0.0, 0.0, 0.0, 0.0, 0.8229136277763227, 0.6038608249885462, 0.9359057444294857, 0.0,                0.0,
                       1.0, 1.0, 1.0, 1.0, 0.0,                0.0,                0.0,                0.0,                0.0,
                       0.0, 0.0, 0.0, 0.0, 0.6666666666666666, 0.6666666666666666, 0.0,                0.6666666666666666, 1.0]
    assert_allclose(actual_values, expected_values)


def test_create_feature_processor_map():
    feature_map = PredictionStrategy.create_feature_processor_map(input_features)
    assert feature_map == input_features

    feature_map = PredictionStrategy.create_feature_processor_map("cips_index")
    assert feature_map == {"cips_index": NoProcessing()}

    feature_map = PredictionStrategy.create_feature_processor_map(["a", "b"])
    assert feature_map == {"a": NoProcessing(), "b": NoProcessing()}


def test_postprocess_features():
    data_array = np.array([
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
    ])
    series_lengths = [2, 3]
    feature_order = ["a", "b"]
    feature_sizes = {"a": 1, "b": 1}
    features = {"a": NoProcessing(), "b": NoProcessing()}

    series_collection = PredictionStrategy.postprocess_features(
        data_array=data_array,
        series_lengths=series_lengths,
        feature_order=feature_order,
        feature_sizes=feature_sizes,
        features=features,
    )

    assert len(series_collection) == 2
    assert len(series_collection[0]) == 2
    assert len(series_collection[1]) == 3
    assert_allclose(series_collection[0][0]["a"], np.array([1.0]))
    assert_allclose(series_collection[0][1]["a"], np.array([2.0]))
    assert_allclose(series_collection[0][0]["b"], np.array([10.0]))
    assert_allclose(series_collection[1][2]["a"], np.array([6.0]))
    assert_allclose(series_collection[1][2]["b"], np.array([60.0]))


def test_gbm_strategy():

    cips_calculator = GBMStrategy(input_features, output_feature)
    cips_calculator.train([[state]]*5, [[state]]*5)

    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-5)

    cips_calculator.save_model('test_gbm_model.h5')

    new_cips_calculator = GBMStrategy.read_from_file('test_gbm_model.h5')
    assert new_cips_calculator == cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-5)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'GBMStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert new_cips_calculator == cips_calculator

    os.remove('test_gbm_model.h5')
    os.remove('test_gbm_model.lgbm')


def test_pod_strategy():

    detector_predictor = PODStrategy("measured_rh_detector", "fine_detector", np.asarray(fine_to_coarse_map))
    detector_predictor.train([[state]]*100)

    assert np.allclose(state["fine_detector"],
                       detector_predictor.predict([[state]])[0][0]["fine_detector"],
                       atol=1E-2)

    new_detector_predictor = build_prediction_strategy(strategy_type     = 'PODStrategy',
                                                       params            = detector_predictor.to_dict(),
                                                       input_features    = {'measured_rh_detector': NoProcessing()},
                                                       predicted_features ='fine_detector',
                                                       biasing_model     = None)
    assert new_detector_predictor == detector_predictor
    assert new_detector_predictor.input_feature == detector_predictor.input_feature
    assert new_detector_predictor.nclusters     == detector_predictor.nclusters
    assert new_detector_predictor.max_svd_size  == detector_predictor.max_svd_size
    assert new_detector_predictor.ndims         == detector_predictor.ndims
    assert np.allclose(new_detector_predictor.fine_to_coarse_map, detector_predictor.fine_to_coarse_map)



def test_nn_strategy_Dense():

    cips_calculator = NNStrategy(input_features, output_feature)
    cips_calculator.train([[state]]*1000)
    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert cips_calculator == new_cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert cips_calculator == new_cips_calculator


def test_nn_strategy_LSTM():

    layers = [LSTM(units=5, activation='relu')]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]*100]*1000)
    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]*100])[0][-1][output_feature],
                    atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert cips_calculator == new_cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]*100])[0][-1][output_feature],
                    atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert cips_calculator == new_cips_calculator


def test_nn_strategy_Transformer():

    layers = [Transformer(num_heads=2, model_dim=27, ff_dim=50)]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]*100]*1000)
    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]*100])[0][-1][output_feature],
                    atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert cips_calculator == new_cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]*100])[0][-1][output_feature],
                    atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert cips_calculator == new_cips_calculator


def test_nn_strategy_CNN():

    layers = [SpatialConv(input_shape=(3,3), kernel_size=(2,2), filters=4),
              SpatialMaxPool(input_shape=(3,3), pool_size=(2,2), padding=False)]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*100)
    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert cips_calculator == new_cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert cips_calculator == new_cips_calculator


def test_nn_strategy_LayerSequence():

    layers          = [PassThrough(), Dense(units=10, activation='relu'), LayerSequence(layers=[Dense(units=5, activation='relu'), Dense(units=10, activation='relu')])]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*1000)
    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert cips_calculator == new_cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert cips_calculator == new_cips_calculator
    assert cips_calculator == new_cips_calculator


def test_nn_strategy_CompoundLayer():

    layers          = [CompoundLayer(layers=[Dense(units=5, activation='relu'), Dense(units=10, activation='relu')], input_specifications=[slice(0, 9), slice(9, 19)])]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*1000)
    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert cips_calculator == new_cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert cips_calculator == new_cips_calculator

def test_nn_strategy_GNN_SAGE():

    graph = SAGE(input_shape=(3, 3), units=4, connectivity='2d-4', aggregator='mean')
    layers = [GraphConv(graph=graph, activation='relu')]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*100)
    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert all(old_layer == new_layer for old_layer, new_layer in zip(cips_calculator.layers, new_cips_calculator.layers))
    assert cips_calculator == new_cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert cips_calculator == new_cips_calculator


def test_nn_strategy_GNN_GAT():

    graph = GAT(input_shape=(3, 3), units=4, connectivity='2d-4', alpha=0.2, temperature=1.0)
    layers = [GraphConv(graph=graph, activation='relu')]
    cips_calculator = NNStrategy(input_features, output_feature, layers)
    cips_calculator.train([[state]]*100)
    assert_allclose(state["cips_index"],
                    cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    cips_calculator.save_model('test_nn_model')
    new_cips_calculator = NNStrategy.read_from_file('test_nn_model')
    assert all(old_layer == new_layer for old_layer, new_layer in zip(cips_calculator.layers, new_cips_calculator.layers))
    assert cips_calculator == new_cips_calculator
    assert_allclose(state["cips_index"],
                    new_cips_calculator.predict([[state]])[0][0][output_feature],
                    atol=1E-2)

    new_cips_calculator = build_prediction_strategy(strategy_type     = 'NNStrategy',
                                                    params            = cips_calculator.to_dict(),
                                                    input_features    = input_features,
                                                    predicted_features = output_feature,
                                                    biasing_model     = None)
    assert cips_calculator == new_cips_calculator

    for file in glob.glob('test_nn_model.*'):
        os.remove(file)


# SklearnStrategy Tests


def create_sklearn_data(n_series: int) -> SeriesCollection:
    """Create synthetic data for testing."""
    series_list = []
    for _ in range(n_series):
        x1 = np.random.randn()
        x2 = np.random.randn()
        y = 2 * x1 + 3 * x2 + np.random.randn() * 0.1

        # Ensure y is a numpy array (not just a scalar)
        state = State({'x1': x1, 'x2': x2, 'y': np.array([y])})
        series_list.append(StateSeries([state]))

    return SeriesCollection(series_list)


@pytest.fixture
def sklearn_train_data():
    """Fixture for training data."""
    return create_sklearn_data(n_series=50)


@pytest.fixture
def sklearn_test_data():
    """Fixture for test data."""
    return create_sklearn_data(n_series=10)


def test_sklearn_initialization():
    """Test strategy initialization."""
    strategy = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=LinearRegression
    )

    assert strategy.estimator is not None
    assert not strategy.isTrained
    assert len(strategy.input_features) == 2
    assert len(strategy.predicted_features) == 1


def test_sklearn_train_and_predict(sklearn_train_data, sklearn_test_data):
    """Test training and prediction."""
    strategy = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=LinearRegression
    )

    # Train
    strategy.train(sklearn_train_data)
    assert strategy.isTrained

    # Predict
    predictions = strategy.predict(sklearn_test_data)
    assert len(predictions) == len(sklearn_test_data)

    # Check predictions are reasonable
    for series in predictions:
        for state in series:
            assert 'y' in state.features
            # Check that y is either a scalar or numpy array
            y_val = state['y']
            if isinstance(y_val, np.ndarray):
                assert y_val.size > 0
            else:
                assert isinstance(y_val, (int, float, np.number))


def test_sklearn_save_and_load(sklearn_train_data, sklearn_test_data):
    """Test saving and loading a trained model."""
    strategy = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=RandomForestRegressor,
        estimator_args={'n_estimators': 10, 'random_state': 42}
    )

    # Train
    strategy.train(sklearn_train_data)
    predictions_before = strategy.predict(sklearn_test_data)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        strategy.save_model(tmp_path)

        # Load
        loaded_strategy = SklearnStrategy.read_from_file(tmp_path)
        assert loaded_strategy.isTrained

        # Predict with loaded model
        predictions_after = loaded_strategy.predict(sklearn_test_data)

        # Compare predictions
        for series_before, series_after in zip(predictions_before, predictions_after):
            for state_before, state_after in zip(series_before, series_after):
                np.testing.assert_almost_equal(
                    state_before['y'],
                    state_after['y'],
                    decimal=6
                )
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.parametrize("estimator_class,estimator_args", [
    (LinearRegression, {}),
    (pytest.importorskip("sklearn.linear_model").Ridge, {'alpha': 1.0}),
    (pytest.importorskip("sklearn.tree").DecisionTreeRegressor, {'max_depth': 3, 'random_state': 42}),
    (RandomForestRegressor, {'n_estimators': 5, 'max_depth': 3, 'random_state': 42}),
])
def test_sklearn_multiple_estimators(sklearn_train_data, sklearn_test_data, estimator_class, estimator_args):
    """Test that different sklearn estimators work."""
    strategy = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=estimator_class,
        estimator_args=estimator_args
    )

    strategy.train(sklearn_train_data)
    assert strategy.isTrained

    predictions = strategy.predict(sklearn_test_data)
    assert len(predictions) == len(sklearn_test_data)


def test_sklearn_predict_before_train_error(sklearn_test_data):
    """Test that predicting before training raises an error."""
    strategy = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=LinearRegression
    )

    with pytest.raises(AssertionError):
        strategy.predict(sklearn_test_data)


def test_sklearn_equality():
    """Test equality comparison."""
    strategy1 = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=LinearRegression
    )

    strategy2 = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=LinearRegression
    )

    # Should be equal (same configuration)
    assert strategy1 == strategy2

    # Different estimator type
    strategy3 = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=Ridge
    )
    assert strategy1 != strategy3
    
    # Same class but different args
    strategy4 = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=Ridge,
        estimator_args={'alpha': 2.0}
    )
    assert strategy3 != strategy4
