import os
import pytest
import h5py
import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression, Ridge

from ml_tools import State, StateSeries, SeriesCollection, MinMaxNormalize
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.pod_theta import (
    THETA_SCALINGS,
    ThetaModel,
    compute_pod_basis,
    project_theta,
    theta_scaling_vector,
)

input_features = {'x': MinMaxNormalize(0., 1.)}


def make_targets(num_samples: int = 30, output_dim: int = 6, rank: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(num_samples, rank)) @ rng.normal(size=(rank, output_dim))


def make_collection(num_series: int = 20, series_len: int = 1) -> SeriesCollection:
    rng = np.random.default_rng(1)
    return SeriesCollection([StateSeries([State({'x': rng.uniform(size=1)}) for _ in range(series_len)])
                             for _ in range(num_series)])


def test_compute_pod_basis():
    targets = make_targets()
    U, L = compute_pod_basis(targets, num_moments=2)

    assert U.shape == (6, 2)
    assert L.shape == (2,)
    assert_allclose(U.T @ U, np.eye(2), atol=1E-12)
    assert_allclose(L, np.linalg.svd(targets, compute_uv=False)[:2])

    # Rank-3 data is exactly reconstructed by 3 moments
    U, _ = compute_pod_basis(targets, num_moments=3)
    assert_allclose(targets @ U @ U.T, targets, atol=1E-10)

    # Subsampling only applies when there are more samples than max_svd_size
    np.random.seed(0)
    _, L_sub = compute_pod_basis(targets, num_moments=3, max_svd_size=10)
    assert not np.allclose(L_sub, np.linalg.svd(targets, compute_uv=False)[:3])
    _, L_all = compute_pod_basis(targets, num_moments=3, max_svd_size=targets.shape[0])
    assert_allclose(L_all, np.linalg.svd(targets, compute_uv=False)[:3])


@pytest.mark.parametrize("theta_scaling", THETA_SCALINGS)
def test_theta_scaling_vector(theta_scaling):
    singular_values = np.array([9., 4., 1.])
    expected = {'singular_values':         singular_values,
                'sqrt_singular_values':    np.sqrt(singular_values),
                'max_singular_value':      np.full(3, 9.),
                'max_sqrt_singular_value': np.full(3, 3.),
                'ones':                    np.ones(3)}[theta_scaling]
    assert_allclose(theta_scaling_vector(theta_scaling, singular_values), expected, rtol=1E-10)


def test_theta_scaling_vector_invalid():
    with pytest.raises(ValueError):
        theta_scaling_vector('bogus', np.ones(2))


def test_project_theta():
    targets = make_targets()
    U, L = compute_pod_basis(targets, num_moments=2)
    scaling = np.sqrt(L)

    thetas = project_theta(list(targets), U, [], scaling)
    assert_allclose(np.asarray(thetas), (targets @ U) / scaling, atol=1E-12)

    # Constrained projection is the stacked least-squares solution
    gamma, W = 0.5, np.random.default_rng(2).normal(size=(4, 6))
    thetas = project_theta(list(targets), U, [(gamma, W)], scaling)
    A = np.vstack([U, gamma * W @ U])
    for target, theta in zip(targets, thetas):
        expected, *_ = np.linalg.lstsq(A, np.concatenate([target, gamma * W @ target]), rcond=None)
        assert_allclose(theta, expected / scaling, atol=1E-12)

    parallel = project_theta(list(targets), U, [(gamma, W)], scaling, num_procs=2)
    assert_allclose(np.asarray(parallel), np.asarray(thetas))


def test_theta_model_construction():
    assert len(ThetaModel(input_features, 3, 'GBM').models) == 3
    assert len(ThetaModel(input_features, 3, 'NN').models) == 1
    assert ThetaModel(input_features, 3, None).model_type == 'GBM'
    assert ThetaModel(input_features, 3, 'sklearn').model_type == 'SKLEARN'
    with pytest.raises(ValueError):
        ThetaModel(input_features, 3, 'bogus')

    # Caller-supplied settings are not mutated
    settings = {'estimator': Ridge}
    ThetaModel(input_features, 2, 'sklearn', settings)
    assert settings == {'estimator': Ridge}


def test_theta_model_set_targets():
    collection = make_collection(num_series=2, series_len=2)
    thetas = [np.array([i, -i], dtype=float) for i in range(4)]
    ThetaModel.set_targets(collection, thetas)

    states = [state for series in collection for state in series]
    for state, theta in zip(states, thetas):
        assert_allclose(state['theta'], theta)
        assert_allclose(state['theta-1'], theta[:1])
        assert_allclose(state['theta-2'], theta[1:])

    with pytest.raises(AssertionError):
        ThetaModel.set_targets(collection, thetas[:3])


def make_theta_collection(num_series: int) -> SeriesCollection:
    collection = make_collection(num_series)
    ThetaModel.set_targets(collection, [np.array([2. * series[0]['x'][0], 1. - series[0]['x'][0]])
                                        for series in collection])
    return collection


@pytest.mark.parametrize("model_type", ['sklearn', 'GBM'])
def test_theta_model_train_predict_save_load(model_type, tmp_path):
    collection = make_theta_collection(20)
    thetas = [series[0]['theta'] for series in collection]

    settings = {'estimator': LinearRegression} if model_type == 'sklearn' else None
    model = ThetaModel(input_features, 2, model_type, settings)
    model.train(collection, make_theta_collection(5))

    processed = PredictionStrategy.preprocess_features(collection, input_features)
    predicted = np.asarray([model.predict(series[0]) for series in processed])
    assert predicted.shape == (len(collection), 2)
    if model_type == 'sklearn':
        assert_allclose(predicted, np.asarray(thetas), atol=1E-8)

    file_name = str(tmp_path / 'theta.h5')
    with h5py.File(file_name, 'w') as h5_file:
        model.write_to_hdf5(h5_file)
    with h5py.File(file_name, 'r') as h5_file:
        loaded = ThetaModel.read_from_hdf5(h5_file, input_features, 2)
    assert loaded.model_type == model.model_type
    assert_allclose(np.asarray([loaded.predict(series[0]) for series in processed]), predicted)

    rebuilt = ThetaModel.from_dict(model.to_dict(), input_features, 2, model.model_type)
    assert rebuilt.model_type == model.model_type
    assert len(rebuilt.models) == len(model.models)
    assert all(not m.isTrained for m in rebuilt.models)


def test_theta_model_from_dict_validates_payload():
    payload = ThetaModel(input_features, 2, 'GBM').to_dict()
    with pytest.raises(AssertionError):
        ThetaModel.from_dict([], input_features, 2, 'GBM')
    with pytest.raises(AssertionError):
        ThetaModel.from_dict(payload[:1], input_features, 2, 'GBM')
    with pytest.raises(AssertionError):
        ThetaModel.from_dict(payload, input_features, 2, 'sklearn')
