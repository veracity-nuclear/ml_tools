from abc import ABC, abstractmethod

from ml_tools.model.nn_strategy.layer import LayerType, Activation

class Layer(ABC):
    """ An abstract base class for defining search space neural network layer dimensions
    """