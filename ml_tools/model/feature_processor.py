from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
from math import isclose
import numpy as np
import h5py

from ml_tools.model import register_feature_processor

class FeatureProcessor(ABC):
    """ An abstract class for pre-processing and post-processing input / output features
    """

    @abstractmethod
    def preprocess(self, orig_data: Sequence) -> np.ndarray:
        """ a method for pre-processing feature data

        Parameters
        ----------
        orig_data : Sequence
            The data in its original form (list, tuple, array, nested sequences, etc.)

        Returns
        -------
        np.ndarray
            The preprocessed form of the data
        """


    @abstractmethod
    def postprocess(self, processed_data: np.ndarray) -> Sequence:
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


    @abstractmethod
    def __eq__(self, other: FeatureProcessor) -> bool:
        """ Compare two FeatureProcessors for equality

        Parameters
        ----------
        other: FeatureProcessor
            The other FeatureProcessor to compare against

        Returns
        -------
        bool
            True if self and other are equal within the tolerance.  False otherwise

        Notes
        -----
        The relative tolerance is 1e-9 for float comparisons
        """

    def to_hdf5(self, group: h5py.Group) -> None:
        """ A method for writing a FeatureProcessor to an HDF5 Group

        Parameters
        ----------
        group : h5py.Group
            The HDF5 Group to write the FeatureProcessor to
        """
        group.create_dataset("type", data=self.__class__.__name__, dtype=h5py.string_dtype())

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> FeatureProcessor:  # pylint: disable=unused-argument
        """ A method for constructing a FeatureProcessor from an HDF5 Group

        Parameters
        ----------
        group : h5py.Group
            The HDF5 Group to read the FeatureProcessor from

        Returns
        -------
        FeatureProcessor
            The FeatureProcessor constructed from the HDF5 Group
        """
        return cls()

@register_feature_processor()
class MinMaxNormalize(FeatureProcessor):
    """ A feature processor that performs Min-Max normalization

    Parameters
    ----------
    min_value : float
        The minimum value of the value range
    max_value : float
        The maximum value of the value range

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
        assert min_value < max_value, f"min value = {min_value}, max value = {max_value}"
        self._min = min_value
        self._max = max_value

    def preprocess(self, orig_data: Sequence) -> np.ndarray:
        data = np.asarray(orig_data)
        return (data - self.min)/(self.max - self.min)

    def postprocess(self, processed_data: np.ndarray) -> np.ndarray:
        return processed_data * (self.max - self.min) + self.min

    def __eq__(self, other: FeatureProcessor) -> bool:
        return (isinstance(other, MinMaxNormalize) and
                isclose(self.min, other.min, rel_tol=1e-9) and
                isclose(self.max, other.max, rel_tol=1e-9))

    def to_hdf5(self, group: h5py.Group) -> None:
        super().to_hdf5(group)
        group.create_dataset("min", data=self.min)
        group.create_dataset("max", data=self.max)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> MinMaxNormalize:
        min_value = group["min"][()]
        max_value = group["max"][()]
        return cls(min_value, max_value)

@register_feature_processor()
class NoProcessing(FeatureProcessor):
    """ A feature processor that performs no processing operations
    """

    def __init__(self):
        pass

    def preprocess(self, orig_data: Sequence) -> np.ndarray:
        return np.array(orig_data, copy=True)

    def postprocess(self, processed_data: np.ndarray) -> np.ndarray:
        return np.array(processed_data, copy=True)

    def __eq__(self, other: FeatureProcessor) -> bool:
        return isinstance(other, NoProcessing)
