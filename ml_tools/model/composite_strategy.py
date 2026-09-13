from __future__ import annotations
from typing import Dict, Optional, Sequence

import h5py
import numpy as np

from ml_tools.model import register_prediction_strategy, _PREDICTION_STRATEGY_REGISTRY
from ml_tools.model.feature_processor import FeatureProcessor
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
from ml_tools.model.state import State, StateSeries, SeriesCollection


@register_prediction_strategy()
class CompositeStrategy(PredictionStrategy):
    """ A concrete class for assembling selected predicted features from a collection of models.

    This prediction strategy constructs a composite model by selecting specific predicted features
    from a collection of supporting models.  Prediction passes the original SeriesCollection
    to each supporting model and selects the specified predicted features for the composite output.

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        Named supporting models. Each model handles its own preprocessing,
        training, and postprocessing.
    predicted_features : Dict[str, str]
        Mapping from output feature name to supplying model name. Insertion
        order defines the composite output order. Unlike this constructor
        argument, the predicted_features property maps features to processors.
    frozen_models : Optional[Sequence[str]]
        Names of models to leave unchanged during training. Frozen models must
        already be trained before training or predicting with the composite.

    Attributes
    ----------
    models : Dict[str, PredictionStrategy]
        Named supporting models.
    frozen_models : frozenset[str]
        Names of models to leave unchanged during training.
    """

    @property
    def models(self) -> Dict[str, PredictionStrategy]:
        return dict(self._models)

    @property
    def frozen_models(self) -> frozenset[str]:
        return self._frozen_models

    @frozen_models.setter
    def frozen_models(self, names: Sequence[str]) -> None:
        names = frozenset(names)
        assert names.issubset(self._models), "frozen_models contains unknown model names"
        self._frozen_models = names

    @property
    def input_features(self) -> Dict[str, FeatureProcessor]:
        combined = {}
        for model_name, model in self._models.items():
            for name, processor in model.input_features.items():
                if name in combined and combined[name] != processor:
                    raise AssertionError(
                        f"Input processor mismatch for '{name}' in model '{model_name}'; "
                        "access input_features on the individual models."
                    )
                combined.setdefault(name, processor)
        return combined

    @input_features.setter
    def input_features(self, features: FeatureSpec) -> None:
        raise AttributeError("input_features is a read-only property for CompositeStrategy")

    @property
    def predicted_features(self) -> Dict[str, FeatureProcessor]:
        return {name: self._models[source].predicted_features[name]
                for name, source in self._feature_sources.items()}

    @predicted_features.setter
    def predicted_features(self, features: FeatureSpec) -> None:
        raise AttributeError("predicted_features is a read-only property for CompositeStrategy")

    @property
    def isTrained(self) -> bool:
        return all(model.isTrained for model in self._models.values())

    def __init__(self,
                 models: Dict[str, PredictionStrategy],
                 predicted_features: Dict[str, str],
                 frozen_models: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        assert isinstance(models, dict) and models, "models must be a non-empty dict"
        for name, model in models.items():
            assert isinstance(name, str) and name, "model names must be non-empty strings"
            assert isinstance(model, PredictionStrategy), f"Invalid prediction strategy for '{name}'"
        assert isinstance(predicted_features, dict) and predicted_features, \
            "predicted_features must be a non-empty feature-to-model mapping"
        for name, source in predicted_features.items():
            assert isinstance(name, str) and name, "predicted feature names must be non-empty strings"
            assert isinstance(source, str) and source in models, f"Unknown model '{source}' for feature '{name}'"
            assert name in models[source].predicted_features, f"Model '{source}' does not predict feature '{name}'"

        self._models = dict(models)
        self._feature_sources = dict(predicted_features)
        self._predicted_feature_order = list(predicted_features)
        self.frozen_models = frozen_models if frozen_models is not None else ()

    def train(self, train_data: SeriesCollection, test_data: Optional[SeriesCollection] = None,
              num_procs: int = 1) -> None:
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"
        for name in self.frozen_models:
            assert self._models[name].isTrained, f"Frozen model '{name}' must already be trained"
        for name, model in self._models.items():
            if name not in self.frozen_models:
                model.train(train_data, test_data, num_procs)

    def predict(self, series_collection: SeriesCollection, num_procs: int = 1) -> SeriesCollection:
        assert self.isTrained, "All supporting models must be trained before prediction"
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"

        predictions = {}
        for name, model in self._models.items():
            prediction        = model.predict(series_collection, num_procs=num_procs)
            predictions[name] = prediction

        return SeriesCollection([
            StateSeries([
                State({feature: predictions[source][i][j][feature]
                       for feature, source in self._feature_sources.items()})
                for j in range(len(series))
            ])
            for i, series in enumerate(series_collection)
        ])

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CompositeStrategy requires predict() with a SeriesCollection")

    def _predict_all(self, series_collection: np.ndarray, num_procs: int = 1) -> np.ndarray:
        raise NotImplementedError("CompositeStrategy requires predict() with a SeriesCollection")

    def __eq__(self, other: object) -> bool:
        # Comparing children avoids requesting a common input processor map.
        return (isinstance(other, CompositeStrategy) and
                list(self._feature_sources.items()) == list(other._feature_sources.items()) and
                list(self._models) == list(other._models) and
                self._models == other._models and
                self.frozen_models == other.frozen_models)

    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        string_type = h5py.string_dtype()
        h5_group.create_dataset('predicted_feature_order', data=self.predicted_feature_names, dtype=string_type)
        h5_group.create_dataset('feature_sources', data=list(self._feature_sources.values()), dtype=string_type)
        h5_group.create_dataset('frozen_models', data=sorted(self.frozen_models), dtype=string_type)
        models_group = h5_group.create_group('models', track_order=True)
        for i, (name, model) in enumerate(self._models.items()):
            # Indexed groups also allow model names containing HDF5 path separators.
            group = models_group.create_group(str(i))
            group.attrs['model_name'] = name
            group.attrs['strategy_type'] = type(model).__name__
            model.write_model_to_hdf5(group)

    def load_model(self, h5_group: h5py.Group) -> None:
        models = {}
        for group in h5_group['models'].values():
            strategy_type = group.attrs['strategy_type']
            if isinstance(strategy_type, bytes):
                strategy_type = strategy_type.decode('utf-8')
            strategy_cls = _PREDICTION_STRATEGY_REGISTRY.get(strategy_type)
            if strategy_cls is None:
                raise KeyError(f"Unknown PredictionStrategy type in HDF5: {strategy_type}")
            model = strategy_cls.__new__(strategy_cls)
            PredictionStrategy.__init__(model)
            model.load_model(group)
            name = group.attrs['model_name']
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            models[name] = model

        order = h5_group['predicted_feature_order'].asstr()[()].tolist()
        sources = h5_group['feature_sources'].asstr()[()].tolist()
        assert len(order) == len(sources) and len(order) == len(set(order)), "Invalid predicted feature routing"
        CompositeStrategy.__init__(self, models=models,
                                   predicted_features=dict(zip(order, sources)),
                                   frozen_models=h5_group['frozen_models'].asstr()[()].tolist())

    @classmethod
    def read_from_file(cls, file_name: str) -> CompositeStrategy:
        file_name = file_name if file_name.endswith('.h5') else file_name + '.h5'
        instance = cls.__new__(cls)
        with h5py.File(file_name, 'r') as h5_file:
            instance.load_model(h5_file)
        return instance

    def to_dict(self) -> dict:
        """Serialize child configurations without requesting common input processors."""
        return {
            "strategy_type": type(self).__name__,
            "predicted_features": self.features_to_dict(self.predicted_features),
            "params": self._params_to_dict(),
        }

    def _params_to_dict(self) -> dict:
        return {
            "models": {name: model.to_dict() for name, model in self._models.items()},
            "predicted_features": dict(self._feature_sources),
            "frozen_models": sorted(self.frozen_models),
        }

    @classmethod
    def _from_params_dict(cls, params: Dict, input_features: Optional[FeatureSpec],
                          predicted_features: Optional[FeatureSpec]) -> CompositeStrategy:
        instance = cls(models={name: PredictionStrategy.from_dict(payload)
                               for name, payload in params['models'].items()},
                       predicted_features=params['predicted_features'],
                       frozen_models=params.get('frozen_models'))
        if input_features is not None:
            assert input_features == instance.input_features, "input_features must match the supporting models"
        if predicted_features is not None:
            assert predicted_features == instance.predicted_features, "predicted_features must match the supplying models"
        return instance
