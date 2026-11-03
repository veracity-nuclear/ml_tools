import glob
import os
import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression

from ml_tools import State, StateSeries, SeriesCollection, DMDPODStrategy, MinMaxNormalize, NoProcessing
from ml_tools.model.prediction_strategy import PredictionStrategy

OUTPUT_DIM, R, M = 12, 3, 2

_rng    = np.random.default_rng(0)
U_TRUE  = np.linalg.qr(_rng.normal(size=(OUTPUT_DIM, R)))[0]
_Q      = np.linalg.qr(_rng.normal(size=(R, R)))[0] + 0.5 * np.eye(R)
B_TRUE  = _rng.normal(size=(R, M))
C_TRUE  = 0.1 * _rng.normal(size=R)


def make_step_operator(eigenvalues) -> np.ndarray:
    return _Q @ np.diag(eigenvalues) @ np.linalg.inv(_Q)


A_TRUE = make_step_operator([0.8, 0.5, -0.3])


def make_collection(lengths, seed=1, A=A_TRUE, c=C_TRUE) -> SeriesCollection:
    """ Series following w_t = A w_{t-1} + B u_t + c, y_t = U w_t, with w_0 = W0 u_0 """
    rng = np.random.default_rng(seed)
    collection = []
    for length in lengths:
        u = rng.uniform(0., 1., size=(length, M))
        w = rng.normal(size=R)
        states = []
        for t in range(length):
            if t > 0:
                w = A @ w + B_TRUE @ u[t] + c
            states.append(State({'u': u[t], 'y': U_TRUE @ w}))
        collection.append(StateSeries(states))
    return SeriesCollection(collection)


def build(**kwargs) -> DMDPODStrategy:
    kwargs.setdefault('num_moments', R)
    kwargs.setdefault('initial_theta_model_type', None)
    predicted = kwargs.pop('predicted_features', 'y')
    return DMDPODStrategy({'u': NoProcessing()}, predicted, **kwargs)


def values(series: StateSeries) -> np.ndarray:
    return np.asarray([state['y'] for state in series])


def rollout(model: DMDPODStrategy, series: StateSeries, initialization: str = 'project') -> np.ndarray:
    trajectory = model.create_trajectory(series[0], initialization)
    states = [trajectory.state] + [trajectory.predict_next(state) for state in series[1:]]
    return np.asarray([state['y'] for state in states])


def test_recovers_linear_system():
    model = build()
    model.train(make_collection([10] * 20))

    assert model.isTrained
    assert model.moments.shape == (OUTPUT_DIM, R)
    assert model.step_operator.shape == (R, R)
    assert model.input_operator.shape == (R, M)
    assert model.offset.shape == (R,)
    assert_allclose(np.sort(np.linalg.eigvals(model.step_operator).real), [-0.3, 0.5, 0.8], atol=1E-8)
    assert model.spectral_radius == pytest.approx(0.8, abs=1E-8)

    for series in make_collection([15] * 3, seed=2):
        assert_allclose(rollout(model, series), values(series), atol=1E-8)


def test_uneven_series_lengths():
    # Padding must not create transition pairs across or after the end of a series
    model = build()
    model.train(make_collection([1, 2, 7, 3, 12, 1, 5, 9, 4, 6]))
    for series in make_collection([8] * 3, seed=2):
        assert_allclose(rollout(model, series), values(series), atol=1E-8)


def test_requires_transitions():
    with pytest.raises(ValueError):
        build().train(make_collection([1] * 10))


def test_fit_intercept():
    no_offset = make_collection([10] * 20, c=np.zeros(R))
    model = build(fit_intercept=False)
    model.train(no_offset)
    assert_allclose(model.offset, np.zeros(R))
    for series in make_collection([10] * 2, seed=2, c=np.zeros(R)):
        assert_allclose(rollout(model, series), values(series), atol=1E-8)

    # Without the intercept, a constant forcing cannot be represented
    model.train(make_collection([10] * 20))
    series = make_collection([10], seed=2)[0]
    assert np.abs(rollout(model, series) - values(series)).max() > 1E-3


def test_processed_predicted_features():
    # MinMaxNormalize is affine, so processed space needs one extra moment for the constant direction
    model = build(predicted_features={'y': MinMaxNormalize(-3., 3.)}, num_moments=R + 1)
    model.train(make_collection([10] * 20))
    for series in make_collection([10] * 2, seed=2):
        assert_allclose(rollout(model, series), values(series), atol=1E-7)


def test_regularization():
    data = make_collection([10] * 20)
    exact = build()
    exact.train(data)
    ridge = build(regularization=10.)
    ridge.train(data)

    assert ridge.regularization == 10.
    assert np.linalg.norm(ridge.input_operator) < np.linalg.norm(exact.input_operator)
    series = make_collection([10], seed=2)[0]
    assert np.abs(rollout(ridge, series) - values(series)).max() > 1E-6


def test_unstable_warning():
    data = make_collection([6] * 20, A=make_step_operator([1.05, 0.5, -0.3]))
    with pytest.warns(RuntimeWarning, match="spectral radius"):
        build().train(data)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        build().train(make_collection([6] * 20))


def test_training_does_not_mutate_data():
    data = make_collection([5] * 10)
    build(initial_theta_model_type='sklearn', initial_theta_model_settings={'estimator': LinearRegression}).train(data)
    assert all(set(state.features) == {'u', 'y'} for series in data for state in series)


def test_predict_matches_trajectory():
    model = build(initial_theta_model_type='sklearn', initial_theta_model_settings={'estimator': LinearRegression})
    model.train(make_collection([10] * 20))
    assert model.initial_theta_model_type == 'SKLEARN'

    collection = make_collection([7, 3, 5], seed=2)
    predicted  = model.predict(collection)
    weights    = model.predict_weights(collection)
    processed  = PredictionStrategy.preprocess_features(collection, model.input_features)

    assert weights.shape == (3, 7, R)
    assert_allclose(weights @ model.moments.T, model.predict_processed_inputs(processed), rtol=1E-5, atol=1E-6)
    for i, series in enumerate(collection):
        assert len(predicted[i]) == len(series)
        assert_allclose(values(predicted[i]), rollout(model, series, 'infer'), rtol=1E-5, atol=1E-6)

        trajectory = model.create_trajectory(series[0], 'infer')
        assert_allclose(trajectory.weights, weights[i, 0], rtol=1E-10)
        for t in range(1, len(series)):
            trajectory.predict_next(series[t])
            assert_allclose(trajectory.weights, weights[i, t], rtol=1E-10)

    assert_allclose(model.predict(collection, num_procs=2).to_array(['y']), predicted.to_array(['y']))


def test_trajectory():
    model = build()
    model.train(make_collection([10] * 20))
    series = make_collection([4], seed=2)[0]

    trajectory = model.create_trajectory(series[0])
    assert trajectory.strategy is model
    assert trajectory.step == 0
    assert_allclose(trajectory.weights, model.moments.T @ series[0]['y'])

    trajectory.predict_next(series[1])
    branch = trajectory.copy()
    trajectory.predict_next(series[2])
    assert (trajectory.step, branch.step) == (2, 1)
    assert not np.allclose(trajectory.weights, branch.weights)
    assert_allclose(branch.predict_next(series[2])['y'], trajectory.state['y'])

    # weights are returned as copies
    trajectory.weights[:] = 0.
    assert not np.allclose(trajectory.weights, 0.)

    processed = trajectory.predict_next_processed_inputs(series[3]['u'])
    assert_allclose(processed, series[3]['y'], atol=1E-8)


def test_initialization_errors():
    model = build()
    model.train(make_collection([5] * 10))
    series = make_collection([3], seed=2)
    with pytest.raises(RuntimeError, match="initial theta model"):
        model.create_trajectory(series[0][0], 'infer')
    with pytest.raises(RuntimeError, match="initial theta model"):
        model.predict(series)
    with pytest.raises(ValueError):
        model.create_trajectory(series[0][0], 'bogus')


@pytest.mark.parametrize("initial_theta_model_type", [None, 'sklearn', 'GBM'])
def test_save_load(initial_theta_model_type, tmp_path):
    settings = {'estimator': LinearRegression} if initial_theta_model_type == 'sklearn' else None
    model = build(initial_theta_model_type=initial_theta_model_type, initial_theta_model_settings=settings,
                  regularization=1E-3, max_svd_size=150)
    model.train(make_collection([10] * 20), make_collection([10] * 5, seed=3))

    file_name = str(tmp_path / 'dmd_pod_model.h5')
    model.save_model(file_name)
    loaded = DMDPODStrategy.read_from_file(file_name)

    assert loaded == model
    assert loaded.predicted_feature_names == model.predicted_feature_names
    for attr in ('moments', 'step_operator', 'input_operator', 'offset'):
        assert_allclose(getattr(loaded, attr), getattr(model, attr))

    series = make_collection([6], seed=2)
    assert_allclose(rollout(loaded, series[0]), rollout(model, series[0]))
    if initial_theta_model_type is not None:
        assert_allclose(loaded.predict(series).to_array(['y']), model.predict(series).to_array(['y']))

    # A loaded model can be saved again
    loaded.save_model(str(tmp_path / 'dmd_pod_model_resaved.h5'))
    assert DMDPODStrategy.read_from_file(str(tmp_path / 'dmd_pod_model_resaved.h5')) == model


@pytest.mark.parametrize("initial_theta_model_type", [None, 'GBM'])
def test_to_from_dict(initial_theta_model_type):
    model = build(initial_theta_model_type=initial_theta_model_type, regularization=0.5, fit_intercept=False)
    payload = model.to_dict()
    assert payload["strategy_type"] == "DMDPODStrategy"

    rebuilt = PredictionStrategy.from_dict(payload)
    assert isinstance(rebuilt, DMDPODStrategy)
    assert rebuilt == model
    assert not rebuilt.isTrained
    assert rebuilt.initial_theta_model_type == model.initial_theta_model_type
