from __future__ import annotations
from typing import Any
from math import isclose
from decimal import Decimal
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module
from tensorflow.keras import KerasTensor

from ml_tools.model.nn_strategy.layer import Layer

@Layer.register_subclass("PassThrough")
class PassThrough(Layer):
    """ A layer for passing through input features

    Parameters
    ----------
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout

    This layer type is useful when constructing composite layers that require passing some features
    straight through to the next layer while other features pass through an actual processing layer.
    """

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, PassThrough) and
                  isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9) and
                  self.batch_normalize == other.batch_normalize and
                  self.layer_normalize == other.layer_normalize)
               )

    def __hash__(self) -> int:
        return hash(Decimal(self.dropout_rate).quantize(Decimal('1e-9')),
                    self.batch_normalize,
                    self.layer_normalize)

    def _build(self, input_tensor: KerasTensor) -> KerasTensor:
        return input_tensor

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type'        ,     data='PassThrough', dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate',     data=self.dropout_rate)
        group.create_dataset('batch_normalize',  data=self.batch_normalize)
        group.create_dataset('layer_normalize',  data=self.layer_normalize)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> PassThrough:
        return cls(dropout_rate     = float(group["dropout_rate"    ][()]),
                   batch_normalize  =  bool(group["batch_normalize" ][()]),
                   layer_normalize  =  bool(group["layer_normalize" ][()]))
