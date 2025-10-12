from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureProcessor

class SearchSpace(ABC):
    """ Abstract base class for defining hyperparameter search spaces.

    This class serves as a blueprint for creating specific search spaces
    that can be used in hyperparameter optimization strategies.

    Parameters
    ----------
    prediction_strategy_type : str
        The type of prediction strategy to be used.
    dimensions : Struct
        The root hyperparameter search space to explore
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding
        feature processing strategy
    predicted_feature : str
        The string specifying the feature to be predicted
    biasing_model : Optional[PredictionStrategy]
        A model that is used to provide an initial prediction of the predicted output,
        acting ultimately as an initial bias

    Attributes
    ----------
    prediction_strategy_type : str
        The type of prediction strategy to be used.
    dimensions : Struct
        The root hyperparameter search space to explore
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding
        feature processing strategy
    predicted_feature : str
        The string specifying the feature to be predicted
    biasing_model : Optional[PredictionStrategy]
        A model that is used to provide an initial prediction of the predicted output,
        acting ultimately as an initial bias
    """

    class Dimension(ABC):
        """ Abstract base class for defining a hyperparameter dimension in the search space.
        """
        pass

    @property
    def prediction_strategy_type(self) -> str:
        return self._prediction_strategy_type

    @prediction_strategy_type.setter
    def prediction_strategy_type(self, value: str) -> None:
        self._prediction_strategy_type = value

    @property
    def dimensions(self) -> Struct:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: Struct) -> None:
        self._dimensions = value

    @property
    def input_features(self) -> Dict[str, FeatureProcessor]:
        return self._input_features

    @input_features.setter
    def input_features(self, value: Dict[str, FeatureProcessor]) -> None:
        self._input_features = value

    @property
    def predicted_feature(self) -> str:
        return self._predicted_feature

    @predicted_feature.setter
    def predicted_feature(self, value: str) -> None:
        assert value not in self.input_features, f"predicted_feature '{value}' " + \
            f"cannot be one of the input_features {list(self.input_features.keys())}"
        self._predicted_feature = value

    @property
    def biasing_model(self) -> Optional[PredictionStrategy]:
        return self._biasing_model

    @biasing_model.setter
    def biasing_model(self, value: Optional[PredictionStrategy]) -> None:
        self._biasing_model = value



class Int(SearchSpace.Dimension):
    """ Integer hyperparameter dimension.

    Parameters
    ----------
    low : int
        Inclusive lower bound.
    high : int
        Inclusive upper bound.
        log : bool
        Whether to sample on a logarithmic scale (default: False).

    Attributes
    ----------
    low : int
        Inclusive lower bound.
    high : int
        Inclusive upper bound.
    log : bool
        Whether to sample on a logarithmic scale (default: False).
    """

    def __init__(self, low: int, high: int, log: bool = False) -> None:
        assert isinstance(low, int) and isinstance(high, int), "Int bounds must be integers"
        assert low <= high, f"low ({low}) must be <= high ({high})"

        self.low  = low
        self.high = high
        self.log  = log


class Float(SearchSpace.Dimension):
    """ Float hyperparameter dimension.

    Parameters
    ----------
    low : float
        Inclusive lower bound.
    high : float
        Inclusive upper bound.
    log : bool
        Whether to sample on a logarithmic scale (default: False).

    Attributes
    ----------
    low : float
        Inclusive lower bound.
    high : float
        Inclusive upper bound.
    log : bool
        Whether to sample on a logarithmic scale (default: False).
    """

    def __init__(self, low: float, high: float, log: bool = False) -> None:
        assert isinstance(low, (int, float)) and isinstance(high, (int, float)), "Float bounds must be numeric"
        assert float(low) < float(high), f"low ({low}) must be < high ({high})"
        self.low  = float(low)
        self.high = float(high)
        self.log  = bool(log)


class Categorical(SearchSpace.Dimension):
    """ Categorical hyperparameter dimension selecting one choice from a list.

    Parameters
    ----------
    choices : List[Any]
        List of possible choices.

    Attributes
    ----------
    choices : List[Any]
        List of possible choices.
    """

    def __init__(self, choices: List[Any]) -> None:
        assert isinstance(choices, list) and len(choices) > 0, "choices must be a non-empty list"
        self.choices = choices


class Bool(Categorical):
    """ Boolean hyperparameter dimension (categorical over [False, True])."""

    def __init__(self) -> None:
        super().__init__([False, True])


class Struct(SearchSpace.Dimension):
    """ Composite dimension consisting of named fields.

    Parameters
    ----------
    fields : Dict[str, Dimension]
        A dictionary mapping field names to their corresponding dimensions.

    Attributes
    ----------
    fields : Dict[str, Dimension]
        A dictionary mapping field names to their corresponding dimensions.
    """

    def __init__(self, fields: Dict[str, SearchSpace.Dimension]) -> None:
        assert isinstance(fields, dict) and len(fields) > 0, "fields must be a non-empty dict"
        for k, v in fields.items():
            assert isinstance(k, str) and isinstance(v, SearchSpace.Dimension), f"Invalid field {k}: {type(v)}"
        self.fields = fields


class Choice(SearchSpace.Dimension):
    """ Composite dimension that selects one option by key and samples its schema.

    Parameters
    ----------
    options : Dict[str, Dimension]
        A dictionary mapping option names to their corresponding dimensions.

    Attributes
    ----------
    options : Dict[str, Dimension]
        A dictionary mapping option names to their corresponding dimensions.
    """

    def __init__(self, options: Dict[str, SearchSpace.Dimension]) -> None:
        assert isinstance(options, dict) and len(options) > 0, "options must be a non-empty dict"
        for k, v in options.items():
            assert isinstance(k, str) and isinstance(v, SearchSpace.Dimension), f"Invalid option {k}: {type(v)}"
        self.options = options


class ListDim(SearchSpace.Dimension):
    """ A dimension which is a list of dimensions.

    Parameters
    ----------
    items : List[Dimension]
        A list of dimensions.

    Attributes
    ----------
    items : List[Dimension]
        A list of dimensions.
    """

    def __init__(self,
                 items: Optional[List[SearchSpace.Dimension]] = None) -> None:
        assert isinstance(items, list) and len(items) > 0, "items must be a non-empty list when provided"
        self.items = items if items is not None else []
