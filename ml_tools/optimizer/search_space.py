from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Optional
from math import isclose

from ml_tools.model import _PREDICTION_STRATEGY_REGISTRY

class SearchSpace(ABC):
    """ Abstract base class for defining hyperparameter search spaces.

    This class serves as a blueprint for creating specific search spaces
    that can be used in hyperparameter optimization strategies.

    Attributes
    ----------
    prediction_strategy_type : str
        The type of prediction strategy to be used.
    dimensions : Struct
        The root hyperparameter search space to explore
    """

    class Dimension(ABC):
        """ Abstract base class for defining a hyperparameter dimension in the search space.
        """

    @property
    def prediction_strategy_type(self) -> str:
        return self._prediction_strategy_type

    @property
    def dimensions(self) -> StructDimension:
        return self._dimensions

    def __init__(self,
                 prediction_strategy_type: str,
                 dimensions:               StructDimension) -> None:
        assert prediction_strategy_type in _PREDICTION_STRATEGY_REGISTRY, \
            f"Unknown prediction strategy: {prediction_strategy_type}"
        self._prediction_strategy_type = prediction_strategy_type
        self._dimensions               = dimensions

class IntDimension(SearchSpace.Dimension):
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


class FloatDimension(SearchSpace.Dimension):
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
        assert float(low) < float(high) or isclose(float(low), float(high)), f"low ({low}) must be <= high ({high})"
        self.low  = float(low)
        self.high = float(high)
        self.log  = bool(log)


class CategoricalDimension(SearchSpace.Dimension):
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


class BoolDimension(CategoricalDimension):
    """ Boolean hyperparameter dimension (categorical over [False, True])."""

    def __init__(self, choices: List[bool]) -> None:
        assert all(choice in [False, True] for choice in choices), \
            f"BoolDimension choices must be [False, True], got {choices}"
        super().__init__([False, True])


class StructDimension(SearchSpace.Dimension):
    """ Composite dimension consisting of named fields.

    Parameters
    ----------
    fields : Dict[str, SearchSpace.Dimension]
        A dictionary mapping field names to their corresponding dimensions.
    struct_type : Optional[str]
        An optional string indicating the type of structure (e.g., "Layer", "OptimizerConfig", etc.)

    Attributes
    ----------
    fields : Dict[str, SearchSpace.Dimension]
        A dictionary mapping field names to their corresponding dimensions.
    struct_type : Optional[str]
        An optional string indicating the type of structure (e.g., "Layer", "OptimizerConfig", etc.)
    """

    def __init__(self, fields: Dict[str, SearchSpace.Dimension], struct_type: Optional[str] = None) -> None:
        assert isinstance(fields, dict) and len(fields) > 0, "fields must be a non-empty dict"
        assert all(k not in ['type'] for k in fields.keys()), "'type' is a reserved field name"
        for k, v in fields.items():
            assert isinstance(k, str) and isinstance(v, SearchSpace.Dimension), f"Invalid field {k}: {type(v)}"
        self.fields      = fields
        self.struct_type = struct_type


class ChoiceDimension(SearchSpace.Dimension):
    """ Composite dimension that selects one option by key and samples its schema.

    Parameters
    ----------
    options : Dict[str, StructDimension]
        A dictionary mapping option names to their corresponding dimensions.

    Attributes
    ----------
    options : Dict[str, StructDimension]
        A dictionary mapping option names to their corresponding dimensions.
    """

    def __init__(self, options: Dict[str, StructDimension]) -> None:
        assert isinstance(options, dict) and len(options) > 0, "options must be a non-empty dict"
        for k, v in options.items():
            assert isinstance(k, str) and isinstance(v, StructDimension), f"Invalid option {k}: {type(v)}"
        self.options = options


class ListDimension(SearchSpace.Dimension):
    """ A dimension which is a list of dimensions.

    Parameters
    ----------
    items : List[SearchSpace.Dimension]
        A list of dimensions.
    label : Optional[str]
        An optional custom label for the list dimension.

    Attributes
    ----------
    items : List[SearchSpace.Dimension]
        A list of dimensions.
    label : Optional[str]
        An optional custom label for the list dimension.
    """

    def __init__(self,
                 items: Optional[List[SearchSpace.Dimension]] = None,
                 label: Optional[str] = None) -> None:
        assert isinstance(items, list) and len(items) > 0, "items must be a non-empty list when provided"
        self.items = items if items is not None else []
        self.label = label
