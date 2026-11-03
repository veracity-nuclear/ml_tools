from __future__ import annotations
from typing import Any, Dict, Optional, Type
import os

import numpy as np
import h5py


from ml_tools.model.state import State, StateSeries, SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureSpec
from ml_tools.model.feature_processor import FeatureProcessor
from ml_tools.model import register_prediction_strategy, _PREDICTION_STRATEGY_REGISTRY


def is_stepwise(model: PredictionStrategy) -> bool:
    """ Whether a strategy predicts each state independently of the other states in its series

    Such strategies can be evaluated one state at a time, e.g. as the residual model of an
    incremental :class:`ResidualCorrectionStrategy.Trajectory`.  NN strategies qualify only if
    every layer operates per time step; recurrent and attention layers mix time steps.

    Parameters
    ----------
    model : PredictionStrategy
        The strategy to check

    Returns
    -------
    bool
        True if the strategy's prediction for each state depends only on that state
    """
    # Imported here to avoid import cycles between strategy modules
    # pylint: disable=import-outside-toplevel
    from ml_tools.model.gbm_strategy import GBMStrategy
    from ml_tools.model.sklearn_strategy import SklearnStrategy
    from ml_tools.model.pod_strategy import PODStrategy
    from ml_tools.model.enhancedPOD_strategy import EnhancedPODStrategy
    from ml_tools.model.nn_strategy import NNStrategy, Dense, PassThrough, SpatialConv, SpatialMaxPool, \
                                           GraphConv, LayerSequence, CompoundLayer

    def stepwise_layer(layer) -> bool:
        if isinstance(layer, (LayerSequence, CompoundLayer)):
            return all(stepwise_layer(sublayer) for sublayer in layer.layers)
        return isinstance(layer, (Dense, PassThrough, SpatialConv, SpatialMaxPool, GraphConv))

    if isinstance(model, (GBMStrategy, SklearnStrategy, PODStrategy, EnhancedPODStrategy)):
        return True
    if isinstance(model, NNStrategy):
        return all(stepwise_layer(layer) for layer in model.layers)
    if isinstance(model, ResidualCorrectionStrategy):
        return is_stepwise(model.reference_model) and is_stepwise(model.residual_model)
    return False


@register_prediction_strategy()
class ResidualCorrectionStrategy(PredictionStrategy):
    """ A concrete class for a residual correction prediction strategy

    This prediction strategy uses a reference prediction strategy to make an initial prediction,
    then trains a secondary model to predict the residual error of the reference model's predictions.
    The final prediction is the sum of the reference model's prediction and the residual correction.
    ResidualCorrectionStrategy can be daisy-chained to create a chain of residual correction models.

    Parameters
    ----------
    residual_model : Optional[PredictionStrategy]
        The prediction strategy used to predict residual corrections.
    reference_model : Optional[PredictionStrategy]
        Optional reference prediction strategy to use for initial predictions.
        This can be set after initialization via the reference_model setter.
    reference_model_frozen : Optional[bool]
        If True, the reference model is not trained when training this ResidualCorrectionStrategy.
        Default is False.

    Attributes
    ----------
    residual_model : Optional[PredictionStrategy]
        The prediction strategy used to predict the residuals of the reference model.
    reference_model : Optional[PredictionStrategy]
        The reference prediction strategy used for the initial predictions of this residual correction model.
    reference_model_frozen : bool
        If True, the reference model is not trained when training this ResidualCorrectionStrategy.
    """

    class Trajectory:
        """ The evolving state of a single incremental residual-corrected prediction

        Created by :meth:`ResidualCorrectionStrategy.create_trajectory`.  Each step advances the
        reference model's trajectory and adds the residual model's prediction for that step's
        inputs, matching :meth:`ResidualCorrectionStrategy.predict`.

        Parameters
        ----------
        strategy : ResidualCorrectionStrategy
            The trained strategy that created this trajectory
        reference_trajectory : Any
            The trajectory of the reference model
        residual : State
            The residual model's prediction for the current step (post-processed)

        Attributes
        ----------
        strategy : ResidualCorrectionStrategy
            The strategy that created this trajectory
        reference_trajectory : Any
            The trajectory of the reference model
        step : int
            The current time step index
        state : State
            The current residual-corrected predicted features (post-processed)
        """

        @property
        def strategy(self) -> ResidualCorrectionStrategy:
            return self._strategy

        @property
        def reference_trajectory(self) -> Any:
            return self._reference_trajectory

        @property
        def step(self) -> int:
            return self._reference_trajectory.step

        @property
        def state(self) -> State:
            return self._strategy._add_residual(self._reference_trajectory.state, self._residual)  # pylint: disable=protected-access

        def __init__(self, strategy: ResidualCorrectionStrategy, reference_trajectory: Any, residual: State) -> None:
            self._strategy             = strategy
            self._reference_trajectory = reference_trajectory
            self._residual             = residual

        def predict_next(self, inputs: State) -> State:
            """ Advance one step and predict the next state

            Parameters
            ----------
            inputs : State
                A state containing the input features of both the reference and residual models
                for the next step

            Returns
            -------
            State
                The residual-corrected predicted features of the next step (post-processed)
            """
            self._reference_trajectory.predict_next(inputs)
            self._residual = self._strategy._predict_residual(inputs)  # pylint: disable=protected-access
            return self.state

        def copy(self) -> ResidualCorrectionStrategy.Trajectory:
            """ An independent copy of this trajectory, e.g. for branching what-if predictions """
            return type(self)(self._strategy, self._reference_trajectory.copy(), self._residual)

    @property
    def residual_model(self) -> Optional[PredictionStrategy]:
        return self._residual_model if hasattr(self, "_residual_model") else None

    @residual_model.setter
    def residual_model(self, model: Optional[PredictionStrategy]) -> None:
        self._residual_model = model
        self._check_predicted_features_are_consistent()
        if model is not None:
            PredictionStrategy.predicted_features.fset(self, model.predicted_features)

    @property
    def reference_model(self) -> Optional[PredictionStrategy]:
        return self._reference_model if hasattr(self, "_reference_model") else None

    @reference_model.setter
    def reference_model(self, model: Optional[PredictionStrategy]) -> None:
        self._reference_model = model
        self._check_predicted_features_are_consistent()


    def _check_predicted_features_are_consistent(self) -> None:
        if self.residual_model is None:
            return
        if self.reference_model is None:
            return
        res_outputs = self.residual_model.predicted_features
        ref_outputs = self.reference_model.predicted_features
        for name, processor in res_outputs.items():
            assert name in ref_outputs and ref_outputs[name] == processor, \
                f"Predicted feature '{name}' processor mismatch between residual and reference models."
        return


    @property
    def reference_model_frozen(self) -> bool:
        return self._reference_model_frozen

    @reference_model_frozen.setter
    def reference_model_frozen(self, frozen: bool) -> None:
        self._reference_model_frozen = frozen

    @property
    def input_features(self) -> Dict[str, FeatureProcessor]:
        res_inputs = self.residual_model.input_features if self.residual_model else {}
        ref_inputs = self.reference_model.input_features if self.reference_model else {}
        combined = dict(res_inputs)
        for name, processor in ref_inputs.items():
            if name in combined and combined[name] != processor:
                raise AssertionError(
                    f"Input processor mismatch for '{name}' between residual and reference models; "
                    "access via residual_model.input_features or reference_model.input_features."
                )
            combined.setdefault(name, processor)
        return combined

    @input_features.setter
    def input_features(self, features: FeatureSpec) -> None:
        raise AttributeError("input_features is a read-only property for ResidualCorrectionStrategy")

    @property
    def predicted_features(self) -> Dict[str, FeatureProcessor]:
        return self._predicted_features

    @predicted_features.setter
    def predicted_features(self, features: FeatureSpec) -> None:
        raise AttributeError("predicted_features is a read-only property for ResidualCorrectionStrategy")

    @property
    def isTrained(self) -> bool:
        if self.reference_model is not None and self.residual_model is not None:
            if self.reference_model.isTrained and self.residual_model.isTrained:
                return True
        return False


    def __init__(self,
                 residual_model:         Optional[PredictionStrategy] = None,
                 reference_model:        Optional[PredictionStrategy] = None,
                 reference_model_frozen: Optional[bool] = False):

        super().__init__()

        self.residual_model         = residual_model
        self.reference_model        = reference_model
        self.reference_model_frozen = reference_model_frozen


    def train(self,
              train_data: SeriesCollection,
              validation_data: Optional[SeriesCollection] = None,
              num_procs: int = 1,
              *,
              validation_split: float = 0.2,
              validation_seed: Optional[int] = 42) -> None:
        """Train the component models using the supplied validation policy."""

        assert self.residual_model is not None, "residual_model must be set before training"
        assert self.reference_model is not None, "reference_model must be set before training"

        if not self.reference_model_frozen:
            self.reference_model.train(train_data,
                                       validation_data,
                                       num_procs=num_procs,
                                       validation_split=validation_split,
                                       validation_seed=validation_seed)

        assert self.reference_model.isTrained, "Reference model must be trained successfully before training residual model."

        predicted_features = self.residual_model.predicted_feature_names
        train_reference    = self.reference_model.predict(train_data, num_procs)
        train_residuals    = train_data.featurewise(op        = np.subtract,
                                                    other     = train_reference,
                                                    features  = predicted_features,
                                                    num_procs = num_procs)

        validation_residuals = None
        if validation_data is not None:
            validation_reference = self.reference_model.predict(validation_data, num_procs)
            validation_residuals = validation_data.featurewise(op        = np.subtract,
                                                               other     = validation_reference,
                                                               features  = predicted_features,
                                                               num_procs = num_procs)

        self.residual_model.train(train_residuals,
                                  validation_residuals,
                                  num_procs=num_procs,
                                  validation_split=validation_split,
                                  validation_seed=validation_seed)


    def predict(self, series_collection: SeriesCollection, num_procs: int = 1) -> SeriesCollection:

        assert self.isTrained, "Model must be trained before prediction"
        assert num_procs > 0, f"num_procs must be > 0, got {num_procs}"

        reference_prediction = self.reference_model.predict(series_collection, num_procs=num_procs)
        residual_prediction  = self.residual_model.predict(series_collection, num_procs=num_procs)
        return reference_prediction.featurewise(op        = np.add,
                                                other     = residual_prediction,
                                                features  = self.residual_model.predicted_feature_names,
                                                num_procs = num_procs)

    def create_trajectory(self, initial_state: State, initialization: str = 'project') -> ResidualCorrectionStrategy.Trajectory:
        """ Create an incremental trajectory starting at an initial state

        Requires a reference model that supports incremental prediction (e.g. DMDPODStrategy, or a
        ResidualCorrectionStrategy built on one) and a residual model that predicts each state
        independently (see :func:`is_stepwise`).

        Note that the residual model is trained on the residuals of :meth:`predict`, whose reference
        predictions use ``initialization='infer'``.  With ``initialization='project'`` the reference
        starts from the known initial state, so its errors (and therefore the ideal residuals) can
        differ from those seen in training.

        Parameters
        ----------
        initial_state : State
            The state at step 0, containing the input features of both models, and the predicted
            features when ``initialization='project'``
        initialization : str
            Passed to the reference model's ``create_trajectory``

        Returns
        -------
        ResidualCorrectionStrategy.Trajectory
            A trajectory at step 0
        """
        if not hasattr(self.reference_model, 'create_trajectory'):
            raise TypeError(f"Reference model {type(self.reference_model).__name__} does not support incremental "
                            "prediction (create_trajectory)")
        if not is_stepwise(self.residual_model):
            raise TypeError(f"Residual model {type(self.residual_model).__name__} does not predict each state "
                            "independently (e.g. it has recurrent or attention layers), so it cannot be stepped "
                            "incrementally")
        assert self.isTrained, "Both reference_model and residual_model must be trained before prediction."
        reference_trajectory = self.reference_model.create_trajectory(initial_state, initialization)
        return ResidualCorrectionStrategy.Trajectory(self, reference_trajectory, self._predict_residual(initial_state))

    def _predict_residual(self, state: State) -> State:
        return self.residual_model.predict(SeriesCollection([StateSeries([state])]))[0][0]

    def _add_residual(self, reference: State, residual: State) -> State:
        features = dict(reference.features)
        for name in self.residual_model.predicted_feature_names:
            features[name] = np.add(reference[name], residual[name])
        return State(features)

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        """Note: This method expects already-preprocessed inputs and therefore
        requires identical input feature definitions for reference and residual
        models. If input features differ, preprocess and predict with each model
        separately before combining outputs."""
        assert self.isTrained, "Both reference_model and residual_model must be trained before prediction."

        assert self.residual_model.input_features == self.reference_model.input_features, \
            "Input features of residual_model and reference_model must match for this method."

        reference_preds = self.reference_model._predict_one(state_series) # pylint: disable=protected-access
        residual_preds  = self.residual_model._predict_one(state_series) # pylint: disable=protected-access
        final_preds     = reference_preds + residual_preds
        return final_preds

    def _predict_all(self, series_collection: np.ndarray, num_procs: int = 1) -> np.ndarray:
        """Note: This method expects already-preprocessed inputs and therefore
        requires identical input feature definitions for reference and residual
        models. If input features differ, preprocess and predict with each model
        separately before combining outputs."""
        assert self.isTrained, "Both reference_model and residual_model must be trained before prediction."

        assert self.residual_model.input_features == self.reference_model.input_features, \
            "Input features of residual_model and reference_model must match for this method."

        reference_preds = self.reference_model._predict_all(series_collection, num_procs) # pylint: disable=protected-access
        residual_preds  = self.residual_model._predict_all(series_collection, num_procs) # pylint: disable=protected-access
        final_preds     = reference_preds + residual_preds
        return final_preds


    def __eq__(self, other: object) -> bool:
        return (super().__eq__(other) and
                isinstance(other, ResidualCorrectionStrategy)        and
                self.reference_model        == other.reference_model and
                self.residual_model         == other.residual_model  and
                self.reference_model_frozen == other.reference_model_frozen)


    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        h5_group.create_dataset('reference_model_frozen', data=int(self.reference_model_frozen))

        assert self.residual_model is not None, "residual_model must be set before saving"
        residual_group = h5_group.create_group('residual_model')
        residual_group.attrs['strategy_type'] = type(self.residual_model).__name__
        self.residual_model.write_model_to_hdf5(residual_group)

        if self.reference_model is not None:
            reference_group = h5_group.create_group('reference_model')
            reference_group.attrs['strategy_type'] = type(self.reference_model).__name__
            self.reference_model.write_model_to_hdf5(reference_group)

    def load_model(self, h5_group: h5py.Group) -> None:
        self.reference_model_frozen = bool(h5_group['reference_model_frozen'][()])

        self.residual_model = self._load_strategy_from_group(h5_group['residual_model'])

        if 'reference_model' in h5_group:
            self.reference_model = self._load_strategy_from_group(h5_group['reference_model'])
        else:
            self.reference_model = None

    @staticmethod
    def _load_strategy_from_group(group: h5py.Group) -> PredictionStrategy:
        strategy_type = group.attrs.get('strategy_type')
        if isinstance(strategy_type, bytes):
            strategy_type = strategy_type.decode('utf-8')
        if isinstance(strategy_type, np.ndarray):
            strategy_type = strategy_type[()].decode('utf-8')
        if strategy_type is None and 'strategy_type' in group:
            strategy_type = group['strategy_type'][()].decode('utf-8')

        cls = _PREDICTION_STRATEGY_REGISTRY.get(strategy_type, None)
        if cls is None:
            raise KeyError(f"Unknown PredictionStrategy type in HDF5: {strategy_type}")

        instance = cls.__new__(cls)
        PredictionStrategy.__init__(instance)

        instance.load_model(group)
        return instance

    @classmethod
    def read_from_file(cls: ResidualCorrectionStrategy, file_name: str) -> Type[ResidualCorrectionStrategy]:
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        assert os.path.exists(file_name), f"file name = {file_name}"

        instance = cls.__new__(cls)
        PredictionStrategy.__init__(instance)
        instance.load_model(h5py.File(file_name, "r"))
        return instance

    @classmethod
    def _from_params_dict(cls,
                          params:             Dict,
                          input_features:     Optional[FeatureSpec],
                          predicted_features: Optional[FeatureSpec]) -> ResidualCorrectionStrategy:
        reference_model_frozen = params.get("reference_model_frozen", False)
        residual_model = cls._build_nested_model_from_params(params=params,
                                                             model_key="residual_model",
                                                             default_input_features=input_features,
                                                             default_predicted_features=predicted_features,
                                                             required=True)
        reference_model = cls._build_nested_model_from_params(params=params,
                                                              model_key="reference_model",
                                                              default_input_features=input_features,
                                                              default_predicted_features=predicted_features,
                                                              required=False)

        return cls(residual_model, reference_model, reference_model_frozen)



    @classmethod
    def _build_nested_model_from_params(cls,
                                        params: Dict,
                                        model_key: str,
                                        default_input_features: Optional[FeatureSpec],
                                        default_predicted_features: Optional[FeatureSpec],
                                        required: bool = False) -> Optional[PredictionStrategy]:
        strategy_dict = params.get(model_key)
        if strategy_dict is None:
            if required:
                raise AssertionError(f"'{model_key}' is required in params")
            return None
        assert isinstance(strategy_dict, dict), f"'{model_key}' must be a dict"

        if "input_features" in strategy_dict:
            input_features_arg = None
        else:
            assert default_input_features is not None, \
                f"'{model_key}.input_features' is missing and no default input_features were provided"
            input_features_arg = cls.create_feature_processor_map(default_input_features)

        if "predicted_features" in strategy_dict:
            predicted_features_arg = None
        else:
            assert default_predicted_features is not None, \
                f"'{model_key}.predicted_features' is missing and no default predicted_features were provided"
            predicted_features_arg = cls.create_feature_processor_map(default_predicted_features)

        from_dict_kwargs = {"strategy_dict": strategy_dict}
        if input_features_arg is not None:
            from_dict_kwargs["input_features"] = input_features_arg
        if predicted_features_arg is not None:
            from_dict_kwargs["predicted_features"] = predicted_features_arg

        return PredictionStrategy.from_dict(**from_dict_kwargs)


    def _params_to_dict(self) -> dict:
        assert self.residual_model is not None, "residual_model must be set before saving to dict"

        params = {"residual_model": self.residual_model.to_dict(),
                  "reference_model_frozen": self.reference_model_frozen}

        if self.reference_model is not None:
            params["reference_model"] = self.reference_model.to_dict()

        return params
