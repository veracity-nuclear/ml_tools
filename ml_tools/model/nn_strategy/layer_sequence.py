from __future__ import annotations
from typing import Any, List
import h5py

from tensorflow.keras import KerasTensor

from ml_tools.model.nn_strategy.layer import Layer

@Layer.register_subclass("LayerSequence")
class LayerSequence(Layer):
    """ A class for a sequence of layers

    A layer sequence does not require dropout rate or normalization,
    specifications as these will this will be dictated by the final
    layer's specifications. If any of these specifications are provided,
    they will be ignored in favor of the final layer specifications.

    Parameters
    ----------
    layers : List[Layer]
        The list of layers that comprise the sequence

    Attributes
    ----------
    layers : List[Layer]
        The list of layers that comprise the sequence
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @layers.setter
    def layers(self, value: List[Layer]) -> None:
        assert len(value) > 0, f"len(value) = {len(value)}"
        self._layers = value

    def __init__(self, layers: List[Layer]) -> None:
        super().__init__(0.0, False, False)
        self.layers = layers

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, LayerSequence) and
                  len(self.layers) == len(other.layers) and
                  all(s_layer == o_layer for s_layer, o_layer in zip(self.layers, other.layers)))
               )

    def __hash__(self) -> int:
        return hash(tuple(self.layers))

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = input_tensor
        for layer in self.layers:
            x = layer.build(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type', data='LayerSequence', dtype=h5py.string_dtype())
        for i, layer in enumerate(self.layers):
            layer_group = group.create_group('layer_' + str(i))
            layer.save(layer_group)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> LayerSequence:
        layers = Layer.layers_from_h5(group)
        return cls(layers=layers)
