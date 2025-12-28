from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Optional
from math import isclose

from ml_tools.model import _PREDICTION_STRATEGY_REGISTRY
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureProcessor

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
    input_features : Dict[str, FeatureProcessor]
        Input feature and processor pairs, keyed by feature name.
    predicted_features : Dict[str, FeatureProcessor]
        Output feature and processor pairs, keyed by feature name.
    biasing_model : Optional[PredictionStrategy]
        Optional biasing/initial model for predictions.

    Parameters
    ----------
    prediction_strategy_type : str
        Registered PredictionStrategy type name (e.g., "NNStrategy").
    dimensions : StructDimension
        Root struct of the parameter domains (not sampled values).
    input_features : Dict[str, FeatureProcessor]
        Input feature and processor pairs, keyed by feature name.
    predicted_features : Dict[str, FeatureProcessor]
        Output feature and processor pairs, keyed by feature name.
    biasing_model : Optional[PredictionStrategy], optional
        Optional prior model to bias predictions, by default None.
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

    @property
    def input_features(self) -> Dict[str, FeatureProcessor]:
        return self._input_features

    @property
    def predicted_features(self) -> Dict[str, FeatureProcessor]:
        return self._predicted_features

    @property
    def biasing_model(self) -> Optional[PredictionStrategy]:
        return self._biasing_model

    def __init__(self,
                 prediction_strategy_type: str,
                 dimensions:               StructDimension,
                 input_features:           Dict[str, FeatureProcessor],
                 predicted_features:       Dict[str, FeatureProcessor],
                 biasing_model:            Optional[PredictionStrategy] = None) -> None:
        assert prediction_strategy_type in _PREDICTION_STRATEGY_REGISTRY, \
            f"Unknown prediction strategy: {prediction_strategy_type}"
        self._prediction_strategy_type = prediction_strategy_type
        self._dimensions               = dimensions
        assert isinstance(input_features, dict) and len(input_features) > 0, \
            "input_features must be a non-empty dict"
        assert isinstance(predicted_features, dict) and len(predicted_features) > 0, \
            "predicted_features must be a non-empty dict"
        self._input_features     = input_features
        self._predicted_features = predicted_features
        self._biasing_model     = biasing_model

class IntDimension(SearchSpace.Dimension):
    """ Integer hyperparameter dimension.

    Parameters
    ----------
    low : int
        Inclusive lower bound.
    high : int
        Inclusive upper bound.
    log : Optional[int]
        Optional logarithmic sampling base (>1) for exponent sampling. When None, sample linearly.

    Attributes
    ----------
    low : int
        Inclusive lower bound.
    high : int
        Inclusive upper bound.
    log : Optional[int]
        Optional logarithmic sampling base (>1) for exponent sampling. When None, sample linearly.
    """

    def __init__(self, low: int, high: int, log: Optional[int] = None) -> None:
        assert isinstance(low, int) and isinstance(high, int), "Int bounds must be integers"
        assert low <= high, f"low ({low}) must be <= high ({high})"
        if log is not None:
            assert isinstance(log, int) and log > 1, f"IntDimension.log base must be int > 1, got {log}"
            assert low > 0 and high > 0, "IntDimension.log sampling requires low, high > 0"

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
    log : Optional[int]
        Optional logarithmic sampling base (>1) for exponent sampling. When None, sample linearly.

    Attributes
    ----------
    low : float
        Inclusive lower bound.
    high : float
        Inclusive upper bound.
    log : Optional[int]
        Optional logarithmic sampling base (>1) for exponent sampling. When None, sample linearly.
    """

    def __init__(self, low: float, high: float, log: Optional[int] = None) -> None:
        assert isinstance(low, (int, float)) and isinstance(high, (int, float)), "Float bounds must be numeric"
        assert float(low) < float(high) or isclose(float(low), float(high)), f"low ({low}) must be <= high ({high})"
        if log is not None:
            assert isinstance(log, int) and log > 1, f"FloatDimension.log base must be int > 1, got {log}"
            assert float(low) > 0.0 and float(high) > 0.0, "FloatDimension.log sampling requires low, high > 0"
        self.low  = float(low)
        self.high = float(high)
        self.log  = log


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
        self.items = items if items is not None else []
        self.label = label
