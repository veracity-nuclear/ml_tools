from ml_tools.optimizer.search_space import StructDimension, FloatDimension, BoolDimension

class Layer(StructDimension):
    """Base class for NN layer search-space dimensions (domains, not values).

    Parameters
    ----------
    dropout_rate : FloatDimension
        Domain for dropout fraction applied after the layer; inclusive bounds in [0, 1].
    batch_normalize : BoolDimension
        Domain for whether to apply batch normalization after the layer.
    layer_normalize : BoolDimension
        Domain for whether to apply layer normalization after the layer.

    Attributes
    ----------
    layer_type : str
        Identifier for the layer type (e.g., "Dense", "SpatialConv").
    dropout_rate : FloatDimension
        Domain for dropout fraction.
    batch_normalize : BoolDimension
        Domain for applying batch normalization.
    layer_normalize : BoolDimension
        Domain for applying layer normalization.
    """

    @property
    def layer_type(self) -> str:
        return self.fields["type"]

    @property
    def dropout_rate(self) -> FloatDimension:
        return self.fields["dropout_rate"]

    @dropout_rate.setter
    def dropout_rate(self, value: FloatDimension) -> None:
        assert 0.0 <= value.low <= 1.0, f"dropout_rate.low = {value.low}"
        assert 0.0 <= value.high <= 1.0, f"dropout_rate.high = {value.high}"
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
