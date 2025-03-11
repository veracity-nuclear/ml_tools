import pytest
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np

from ml_tools import NonPerturbator, NormalPerturbator, RelativeNormalPerturbator

orig_data = np.random.uniform(0.0, 10.0, size=100000)

def test_non_perturbator():
    perturbator = NonPerturbator()
    pert_data   = perturbator.perturb(orig_data)
    assert_array_equal(pert_data, orig_data)

def test_normal_perturbator():
    perturbator   = NormalPerturbator(0.2)
    pert_data     = perturbator.perturb(orig_data)
    perturbations = pert_data - orig_data
    assert not(np.array_equal(pert_data, orig_data))
    assert_allclose(np.mean(perturbations), 0.0, atol=1E-2)
    assert_allclose(np.std(perturbations), 0.2, atol=1E-2)

def test_relative_normal_perturbator():
    perturbator   = RelativeNormalPerturbator(0.5)
    pert_data     = perturbator.perturb(orig_data)
    perturbations = pert_data / orig_data
    assert not(np.array_equal(pert_data, orig_data))
    assert_allclose(np.mean(perturbations - 1.0), 0.0, atol=1E-2)
    assert_allclose(np.std(perturbations - 1.0), 0.5, atol=1E-2)