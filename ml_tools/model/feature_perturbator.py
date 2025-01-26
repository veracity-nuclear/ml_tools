from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

class FeaturePerturbator(ABC):
    """ An abstract class for perturbing input / output features in some prescribed manner
    """

    @abstractmethod
    def perturb(self, orig_data: np.ndarray) -> np.ndarray:
        """ a method for perturbing feature data

        Parameters
        ----------
        orig_data : np.ndarray
            The data in its original form

        Returns
        -------
        np.ndarray
            The resulting perturbed data
        """


class NonPerturbator(FeaturePerturbator):
    """ A feature perturbation that performs no perturbation operations
    """

    def __init__(self):
        pass

    def perturb(self, orig_data: np.ndarray) -> np.ndarray:
        return deepcopy(orig_data)


class NormalPerturbator(FeaturePerturbator):
    """ A feature perturbator that applies perturbations using random sampling from a normal distribution

    Parameters
    ----------
    std_dev: float
        The standard deviation of the random sampling normal distribution
    """

    @property
    def std_dev(self) -> float:
        return self._std_dev

    @std_dev.setter
    def std_dev(self, std_dev: float) -> None:
        assert std_dev > 0.
        self._std_dev = std_dev


    def __init__(self, std_dev: float):
        self.std_dev = std_dev

    def perturb(self, orig_data: np.ndarray) -> np.ndarray:
        return orig_data + np.random.normal(0.0, self.std_dev, orig_data.shape)


class RelativeNormalPerturbator(NormalPerturbator):
    """ A feature perturbator that applies perturbations using random sampling from a relative normal distribution
    """

    def perturb(self, orig_data: np.ndarray) -> np.ndarray:
        return orig_data * np.random.normal(1.0, self.std_dev, orig_data.shape)
