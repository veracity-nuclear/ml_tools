from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy
from math import isclose
import numpy as np
import h5py

class FeatureProcessor(ABC):
    """ An abstract class for pre-processing and post-processing input / output features
    """

    @abstractmethod
    def preprocess(self, orig_data: np.ndarray) -> np.ndarray:
       """ a method for pre-processing feature data

       Parameters
       ----------
       orig_data : np.ndarray
           The data in its original form

       Returns
       -------
       np.ndarray
           The preprocessed form of the data
       """
       pass


    @abstractmethod
    def postprocess(self, processed_data: np.ndarray) -> np.ndarray:
       """ a method for post-processing feature data

       Post-processing here means the inverse operation of pre-processing

       Parameters
       ----------
       processed_data : np.ndarray
           The data in its processed form that must be post-processed

       Returns
       -------
       np.ndarray
           The post-processed form of the data
       """
       pass


    @abstractmethod
    def __eq__(self, other: FeatureProcessor) -> bool:
       pass


class MinMaxNormalize(FeatureProcessor):
    """ A feature processor that performs Min-Max normalization

    Attributes
    ----------
    min_value : float
        The minimum value of the value range
    max_value : float
        The maximum value of the value range
    """

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max


    def __init__(self, min_value: float, max_value: float):
        assert(min_value < max_value)
        self._min = min_value
        self._max = max_value

    def preprocess(self, orig_data: np.ndarray) -> np.ndarray:
        return (orig_data - self.min)/(self.max - self.min)

    def postprocess(self, processed_data: np.ndarray) -> np.ndarray:
        return processed_data * (self.max - self.min) + self.min

    def __eq__(self, other: FeatureProcessor) -> bool:
        if(not(type(self) == type(other))): return False
        if(not(isclose(self.min, other.min))): return False
        if(not(isclose(self.max, other.max))): return False
        return True


class NoProcessing(FeatureProcessor):
    """ A feature processor that performs no processing operations
    """

    def __init__(self):
        pass

    def preprocess(self, orig_data: np.ndarray) -> np.ndarray:
        return deepcopy(orig_data)

    def postprocess(self, processed_data: np.ndarray) -> np.ndarray:
        return deepcopy(processed_data)

    def __eq__(self, other: FeatureProcessor) -> bool:
        if(not(type(self) == type(other))): return False
        return True


def write_feature_processor(group: h5py.Group, processor: FeatureProcessor) -> None:
    """ A function for writing feature processors to an HDF5 Group

    Parameters
    ----------
    group : h5py.Group
        The HDF5 Group to write the feature processor to
    processor : FeatureProcessor
        The Feature Processor to write to the HDF5 Group
    """

    def get_public_properties(obj: FeatureProcessor):
        for attr_name in dir(obj):
            attr_value = getattr(obj, attr_name)
            if isinstance(attr_value, property) or (not attr_name.startswith('_') and isinstance(getattr(type(obj), attr_name, None), property)):
                yield attr_name

    group.create_dataset("type", data=type(processor).__name__, dtype=h5py.string_dtype())
    for var in get_public_properties(processor):
        val = getattr(processor, var)
        group.create_dataset(var, data=val)


def read_feature_processor(group: h5py.Group) -> FeatureProcessor:
    """ A function for reading feature processors from an HDF5 Group

    Parameters
    ----------
    group : h5py.Group
        The HDF5 Group to read the feature processor from

    Returns
    -------
    FeatureProcessor
        The feature processor read from the HDF5 Group
    """

    processor_type = group["type"][()].decode('utf-8')
    if    processor_type == "MinMaxNormalize":  return MinMaxNormalize(group["min"][()], group["max"][()])
    elif  processor_type == "NoProcessing"   :  return NoProcessing()
    else:
        assert False, f"Unsupported processor type: {processor_type}"