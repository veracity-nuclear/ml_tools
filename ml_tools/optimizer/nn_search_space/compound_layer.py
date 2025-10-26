from typing import List, Union

from ml_tools.optimizer.nn_search_space.layer import Layer
from ml_tools.optimizer.search_space import ListDimension, CategoricalDimension, FloatDimension, BoolDimension


class CompoundLayer(Layer):
    """Search-space dimension for a CompoundLayer (domains, not values).

    Parameters
    ----------
    layers : List[Layer]
        List of sub-layer dimensions to run in parallel.
    input_specifications : List[Union[slice, List[int]]]
        For each sub-layer, the input indices to gather from the incoming feature vector.
        Each element may be a list of non-negative indices, or a slice with an explicit
        non-negative stop index; slices are converted to lists eagerly.
    dropout_rate : FloatDimension, optional
        Dropout rate applied after merging outputs. Defaults to 0.0.
    batch_normalize : BoolDimension, optional
        Whether to apply batch normalization. Defaults to False.
    layer_normalize : BoolDimension, optional
        Whether to apply layer normalization. Defaults to False.

    Attributes
    ----------
    layers : ListDimension
        Dimension describing the parallel sub-layers.
    input_specifications : CategoricalDimension
        Single categorical choice containing the full list-of-lists of indices, enabling the
        sampler to pass the literal structure through unchanged (no resampling of structure).
    """

    @property
    def layers(self) -> ListDimension:
        return self.fields["layers"]

    @layers.setter
    def layers(self, value: List[Layer]) -> None:
        self.fields["layers"] = ListDimension(items=value, label="layer")

    @property
    def input_specifications(self):
        return self.fields["input_specifications"]

    @input_specifications.setter
    def input_specifications(self, value: List[Union[slice, List[int]]]) -> None:
        def to_list(spec: Union[slice, List[int]]) -> List[int]:
            if isinstance(spec, slice):
                assert spec.stop is not None and spec.stop >= 0, \
                    "Input specification slices must have explicit non-negative ending indices"
                start = 0 if spec.start is None else spec.start
                step = 1 if spec.step is None else spec.step
                return list(range(start, spec.stop, step))
            return list(spec)

        list_of_lists = [to_list(spec) for spec in value]
        self.fields["input_specifications"] = CategoricalDimension([list_of_lists])

    def __init__(self,
                 layers:               List[Layer],
                 input_specifications: List[Union[slice, List[int]]],
                 dropout_rate:         FloatDimension = FloatDimension(0.0, 0.0),
                 batch_normalize:      BoolDimension  = BoolDimension([False]),
                 layer_normalize:      BoolDimension  = BoolDimension([False])) -> None:
        super().__init__(dropout_rate, batch_normalize, layer_normalize)
        self.layers = layers
        self.input_specifications = input_specifications
