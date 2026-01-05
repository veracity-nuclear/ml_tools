from __future__ import annotations
from typing import Callable, Dict, Optional
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureProcessor, FeatureSpec

# Registry for PredictionStrategy builders
_PREDICTION_STRATEGY_REGISTRY: Dict[str, type] = {}


def register_prediction_strategy(name: Optional[str] = None):
    """Decorator to register a PredictionStrategy subclass by name.

    If name is not provided, the class name is used.
    """
    def decorator(cls):
        key = name or cls.__name__
        _PREDICTION_STRATEGY_REGISTRY[key] = cls
        return cls
    return decorator


def build_prediction_strategy(strategy_type:     str,
                              params:            dict,
                              input_features:    FeatureSpec,
                              predicted_features: FeatureSpec,
                              biasing_model:     Optional[PredictionStrategy] = None):
    """Factory to build a PredictionStrategy from a registered type and params dict."""
    if strategy_type not in _PREDICTION_STRATEGY_REGISTRY:
        raise KeyError(f"Unknown PredictionStrategy type: {strategy_type}")
    cls = _PREDICTION_STRATEGY_REGISTRY[strategy_type]
    if hasattr(cls, 'from_dict') and callable(getattr(cls, 'from_dict')):
        return cls.from_dict(params             = params,
                             input_features     = input_features,
                             predicted_features = predicted_features,
                             biasing_model      = biasing_model)
    raise NotImplementedError(f"Class {cls.__name__} does not implement from_dict method")
