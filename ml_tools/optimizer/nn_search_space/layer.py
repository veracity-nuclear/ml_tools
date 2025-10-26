from abc import ABC, abstractmethod

from ml_tools.optimizer.search_space import SearchSpace, StructDimension, FloatDimension, BoolDimension

class Layer(StructDimension):
    """ An abstract base class for defining search space neural network layer dimensions

    Parameters
    ----------
    dropout_rate : FloatDimension
        The dropout rate to use in the layer (defaults to 0.0, meaning no dropout)
    batch_normalize : BoolDimension
        Whether to apply batch normalization after the layer (defaults to False)
    layer_normalize : BoolDimension
        Whether to apply layer normalization after the layer (defaults to False)

    Attributes
    ----------
    layer_type : str
        The type of layer (e.g. "Dense", "Conv2D", etc.)
    dropout_rate : FloatDimension
        The dropout rate to use in the layer (defaults to 0.0, meaning no dropout)
    batch_normalize : BoolDimension
        Whether to apply batch normalization after the layer (defaults to False)
    layer_normalize : BoolDimension
        Whether to apply layer normalization after the layer (defaults to False)
    """

    @property
    def layer_type(self) -> str:
        return self.fields["type"]

    @property
    def dropout_rate(self) -> FloatDimension:
        return self.fields["dropout_rate"]

    @dropout_rate.setter
    def dropout_rate(self, value: FloatDimension) -> None:
        assert 0.0 <= value.low <= 1.0, f"dropout_rate = {value.low}"
        self.fields["dropout_rate"] = value

    @property
    def batch_normalize(self) -> BoolDimension:
        return self.fields["batch_normalize"]

    @batch_normalize.setter
    def batch_normalize(self, value: BoolDimension) -> None:
        self.fields["batch_normalize"] = value

    @property
    def layer_normalize(self) -> BoolDimension:
        return self.fields["layer_normalize"]

    @layer_normalize.setter
    def layer_normalize(self, value: BoolDimension) -> None:
        self.fields["layer_normalize"] = value

    def __init__(self,
                 dropout_rate:    FloatDimension = FloatDimension(0., 0.),
                 batch_normalize: BoolDimension  = BoolDimension([False]),
                 layer_normalize: BoolDimension  = BoolDimension([False])) -> None:

        self.fields = {}
        self.dropout_rate    = dropout_rate
        self.batch_normalize = batch_normalize
        self.layer_normalize = layer_normalize
        super().__init__(self.fields, self.__class__.__name__)