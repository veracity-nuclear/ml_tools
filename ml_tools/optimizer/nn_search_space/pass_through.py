from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import FloatDimension, BoolDimension


class PassThrough(Layer):
    """Search-space dimension for a PassThrough layer (domains, not values).

    Parameters
    ----------
    dropout_rate : FloatDimension, optional
        Domain for dropout fraction after the passthrough. Defaults to 0.0.
    batch_normalize : BoolDimension, optional
        Whether to apply batch normalization. Defaults to False.
    layer_normalize : BoolDimension, optional
        Whether to apply layer normalization. Defaults to False.
    """

    def __init__(self,
                 dropout_rate:    FloatDimension = FloatDimension(0.0, 0.0),
                 batch_normalize: BoolDimension  = BoolDimension([False]),
                 layer_normalize: BoolDimension  = BoolDimension([False])) -> None:
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
