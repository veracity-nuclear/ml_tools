from typing import List

from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import ListDimension


class LayerSequence(Layer):
    """Search-space dimension for a sequence of layers (domains, not values).

    Parameters
    ----------
    layers : List[Layer]
        List of layer dimension instances that form the sequence; wrapped
        as a ListDimension to represent a list of sub-dimensions.

    Attributes
    ----------
    layers : List[Layer]
        Domain describing the list of layers (via ListDimension).
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
