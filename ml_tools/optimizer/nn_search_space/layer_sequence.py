from typing import List

from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import ListDimension

class LayerSequence(Layer):
    """ A class representing a sequence of neural network layers in a hyperparameter search space

    Parameters
    ----------
    layers : List[Layer]
        A list of Layer instances representing a layer in the sequence

    Attributes
    ----------
    layers : List[Layer]
        A list of Layer instances representing a layer in the sequence
    """

    @property
    def layers(self) -> List[Layer]:
        return self.fields["layers"]

    @layers.setter
    def layers(self, value: List[Layer]) -> None:
        self.fields["layers"] = ListDimension(items=value, label="layer")

    def __init__(self, layers: List[Layer]) -> None:
        super().__init__()
        self.layers = layers
