from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING
import h5py

if TYPE_CHECKING:
    from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
    from ml_tools.model.feature_processor import FeatureProcessor

# Registry for PredictionStrategy builders
_PREDICTION_STRATEGY_REGISTRY: Dict[str, type] = {}

# Registry for FeatureProcessor builders
_FEATURE_PROCESSOR_REGISTRY: Dict[str, type] = {}

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

def register_feature_processor(name: Optional[str] = None):
    """Decorator to register a FeatureProcessor subclass by name.

    If name is not provided, the class name is used.
    """
    def decorator(cls):
        key = name or cls.__name__
        _FEATURE_PROCESSOR_REGISTRY[key] = cls
        return cls
    return decorator

def build_feature_processor(processor_type:     str,
                            group:           h5py.Group) -> FeatureProcessor:
    """Factory to build a FeatureProcessor from a registered type and params dict."""
    if processor_type not in _FEATURE_PROCESSOR_REGISTRY:
        raise KeyError(f"Unknown FeatureProcessor type: {processor_type}")
    cls = _FEATURE_PROCESSOR_REGISTRY[processor_type]
    if hasattr(cls, 'from_hdf5') and callable(getattr(cls, 'from_hdf5')):
        return cls.from_hdf5(group)
    raise NotImplementedError(f"Class {cls.__name__} does not implement from_hdf5 method")
