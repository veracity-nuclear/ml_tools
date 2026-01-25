import pytest
from numpy.testing import assert_allclose
import numpy as np
import os
import h5py

from ml_tools import MinMaxNormalize, NoProcessing

def test_minmax_normalize():
    orig_data = np.array([2., 5., 6., 8.])

    processor = MinMaxNormalize(2., 8.)

    # Test preprocess
    preprocessed_data  = processor.preprocess(orig_data)
    expected_values = np.array([0., 0.5, 2./3., 1.])
    assert_allclose(preprocessed_data, expected_values)

    # Test postprocess
    postprocessed_data = processor.postprocess(preprocessed_data)
    assert_allclose(postprocessed_data, orig_data)



def test_no_processing():
    orig_data = np.array([2., 5., 6., 8.])

    processor = NoProcessing()

    # Test preprocess
    preprocessed_data  = processor.preprocess(orig_data)
    assert_allclose(preprocessed_data, orig_data)

    # Test postprocess
    postprocessed_data = processor.postprocess(preprocessed_data)
    assert_allclose(postprocessed_data, orig_data)



def test_read_write_functions():
    min_max_normalize   = MinMaxNormalize(0., 1.)
    no_processing       = NoProcessing()

    with h5py.File('read_write_processor.h5', 'w') as h5_file:
        input_features = h5_file.create_group('input_features')

        min_max_normalize_group = input_features.create_group('min_max_normalize')
        min_max_normalize.to_hdf5(min_max_normalize_group)

        no_processing_group = input_features.create_group('no_processing')
        no_processing.to_hdf5(no_processing_group)

    with h5py.File('read_write_processor.h5', 'r') as h5_file:
        processor = MinMaxNormalize.from_hdf5(h5_file['input_features']['min_max_normalize'])
        assert(processor == min_max_normalize)

        processor = NoProcessing.from_hdf5(h5_file['input_features']['no_processing'])
        assert(processor == no_processing)

    os.system('rm read_write_processor.h5')