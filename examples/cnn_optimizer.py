from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.model_selection import KFold
import optuna
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from ml_tools import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy, FeatureProcessor
from ml_tools.model.nn_strategy import Activation, NNStrategy, Dense, SpatialConv, PassThrough, LayerSequence, CompoundLayer
from optimizer import Optimizer


class CNNOptimizer(Optimizer):
    """ An abstract class for CNN model optimizers
    """

    @dataclass
    class Dimensions(Optimizer.Dimensions):
        """ A data class for the dimensions and bounds of the model hyperparameter search space
        """

        assembly_CNN_stencil_size:      Tuple[int, int]      # Min and Max assembly CNN stencil sizes
        assembly_CNN_filters_log2:      Tuple[int, int]      # Min and Max assembly CNN filters expressed as an exponential of 2
        assembly_CNN_activation:        List[Activation]     # List of activiation functions to consider for the assembly CNN Layers
        detector_CNN_stencil_size:      Tuple[int, int]      # Min and Max detector CNN stencil sizes
        detector_CNN_filters_log2:      Tuple[int, int]      # Min and Max detector CNN filters expressed as an exponential of 2
        detector_CNN_activation:        List[Activation]     # List of activiation functions to consider for the detector CNN Layers
        CNN_dropout:           Tuple[float, float]  # Min and Max CNN layer dropout rates

        def __post_init__(self):
            super().__post_init__()

            assert all(value >= 1 for value in self.assembly_CNN_stencil_size), \
                f"assembly_CNN_stencil_size = {self.assembly_CNN_stencil_size}"
            assert all(value >= 0 for value in self.assembly_CNN_filters_log2), \
                f"assembly_CNN_filters_log2 = {self.assembly_CNN_filters_log2}"
            assert self.assembly_CNN_stencil_size[0] <= self.assembly_CNN_stencil_size[1], \
                f"assembly_CNN_stencil_size = {self.assembly_CNN_stencil_size}"
            assert self.assembly_CNN_filters_log2[0] <= self.assembly_CNN_filters_log2[1], \
               f"assembly_CNN_filters_log2 = {self.assembly_CNN_filters_log2}"


            assert all(value >= 1 for value in self.detector_CNN_stencil_size), \
                f"detector_CNN_stencil_size = {self.detector_CNN_stencil_size}"
            assert all(value >= 0 for value in self.detector_CNN_filters_log2), \
                f"detector_CNN_filters_log2 = {self.detector_CNN_filters_log2}"
            assert self.detector_CNN_stencil_size[0] <= self.detector_CNN_stencil_size[1], \
                f"detector_CNN_stencil_size = {self.detector_CNN_stencil_size}"
            assert self.detector_CNN_filters_log2[0] <= self.detector_CNN_filters_log2[1], \
               f"detector_CNN_filters_log2 = {self.detector_CNN_filters_log2}"

            assert all(0.0 <= value <= 1.0 for value in self.CNN_dropout), \
                f"CNN_dropout = {self.CNN_dropout}"
            assert self.CNN_dropout[0] <= self.CNN_dropout[1], \
               f"CNN_dropout = {self.CNN_dropout}"


    @property
    def dimensions(self) -> Dimensions:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: Dimensions) -> None:
        self._dimensions = dimensions


    def _build_model(self,
                     initial_learning_rate:     float,
                     learning_decay_rate:       float,
                     batch_size_log2:           int,
                     assembly_CNN_stencil_size: int,
                     assembly_CNN_filters_log2: int,
                     assembly_CNN_activation:   Activation,
                     detector_CNN_stencil_size: int,
                     detector_CNN_filters_log2: int,
                     detector_CNN_activation:   Activation,
                     CNN_dropout:               float,
                     num_dens_layers:           int,
                     dens_layers:               List[Tuple[int, str, float]]) -> PredictionStrategy:
        """ Helper method for creating models from a given set of hyperparameter settings
        """
        assert num_dens_layers <= len(dens_layers), \
            f"num_dens_layers = {num_dens_layers}, len(dens_layers) = {len(dens_layers)}"

        assembly_inputs     = list(set(["average_exposure", "assembly_enrichment"]) &
                                   set(self.input_features.keys()))

        detector_inputs     = list(set(["measured_fixed_detector"]) &
                                   set(self.input_features.keys()))

        # This section assumes all 3x3 assembly inputs are listed first as the input_features of the models
        len_all_inputs      = sum([len(self.series_collection[0][0][feature]) for feature in self.input_features])
        len_assembly_inputs = sum([len(self.series_collection[0][0][feature]) for feature in assembly_inputs if feature in self.input_features])
        len_detector_inputs = sum([len(self.series_collection[0][0][feature]) for feature in detector_inputs if feature in self.input_features])
        assembly_inputs     = slice(0, len_assembly_inputs)                                       # Slice of 3x3 assembly inputs for CNN
        detector_inputs     = slice(len_assembly_inputs, len_assembly_inputs+len_detector_inputs) # Slice of 7x1 detector inputs for CNN
        other_inputs        = slice(len_assembly_inputs+len_detector_inputs, len_all_inputs)      # Slice of all other inputs

        assembly_layers = LayerSequence(layers = [SpatialConv(input_shape     = (3,3),
                                                              padding         = True,
                                                              batch_normalize = True,
                                                              kernel_size     = (assembly_CNN_stencil_size, assembly_CNN_stencil_size),
                                                              filters         = 8 * (2**assembly_CNN_filters_log2),
                                                              activation      = assembly_CNN_activation)])

        detector_layers = LayerSequence(layers = [SpatialConv(input_shape     = (7,),
                                                              padding         = True,
                                                              batch_normalize = True,
                                                              kernel_size     = (detector_CNN_stencil_size, detector_CNN_stencil_size),
                                                              filters         = 8 * (2**detector_CNN_filters_log2),
                                                              activation      = detector_CNN_activation)])

        cnn_layers = CompoundLayer(input_specifications = [assembly_inputs, detector_inputs, other_inputs],
                                   layers               = [assembly_layers, detector_layers, PassThrough()],
                                   dropout_rate         = CNN_dropout,
                                   batch_normalize      = True)
        dense_layers = LayerSequence(layers=[Dense(units=neurons, activation=activation, dropout_rate=dropout)
                                             for (neurons, activation, dropout) in dens_layers[:num_dens_layers]])

        model = NNStrategy(self.input_features, self.predicted_feature, [cnn_layers, dense_layers],
                           initial_learning_rate, learning_decay_rate, self.epoch_limit, self.convergence_criteria,
                           self.convergence_patience, 2 ** batch_size_log2)

        if self.biasing_model:
            model.biasing_model = self.biasing_model

        return model



class CNNSkoptOptimizer(CNNOptimizer):
    """ A CNN model hyper parameter optimizer using skopt

    Parameters
    ----------
    dimensions : Dimensions
        The degrees of freedom and bounds for the optimization search
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding feature processing strategy
    predicted_feature : str
        The string specifying the feature to be predicted
    series_collection : SeriesCollection
        The input state series collection which to predict outputs for
    num_procs : int
        The number of parallel processors to use when reading data from the HDF5
    test_fraction : float
        The fraction of training data to withold for testing
    number_of_folds : int
        The number of folds to use for cross-fold validation
    epoch_limit : int
        The limit on the number of training epochs conducted during training
    convergence_criteria : float
        The convergence criteria for training
    convergence_patience : int
        Number of epochs with no improvement (i.e. error improves by greater than the convergence_criteria)
        after which training will be stopped
    biasing_model : Optional[PredictionStrategy]
        A model that is used to provide an initial prediction of the predicted output, acting ultimately as an initial bias
    """

    def __init__(self,
                 dimensions:           Dimensions,
                 input_features:       Dict[str, FeatureProcessor],
                 predicted_feature:    str,
                 series_collection:    SeriesCollection,
                 num_procs:            int = 1,
                 test_fraction:        int = 0.2,
                 number_of_folds:      int = 5,
                 epoch_limit:          int = 1000,
                 convergence_criteria: float = 1E-14,
                 convergence_patience: int = 100,
                 biasing_model:        Optional[PredictionStrategy] = None):
        self.dimensions           = dimensions
        self.input_features       = input_features
        self.predicted_feature    = predicted_feature
        self.series_collection    = series_collection
        self.num_procs            = num_procs
        self.test_fraction        = test_fraction
        self.number_of_folds      = number_of_folds
        self.epoch_limit          = epoch_limit
        self.convergence_criteria = convergence_criteria
        self.convergence_patience = convergence_patience
        self.biasing_model        = biasing_model

        self._search_space = None
        self._set_objective_function()


    def optimize(self, num_trials: int, output_file: str) -> PredictionStrategy:

        self._build_search_space()

        def log_callback(res):
            with open(output_file, "a") as output:
                # Calculate the iteration based on function values
                current_iteration = len(res.func_vals)

                 # Write iteration number and the current parameters
                output.write(f"Iteration {current_iteration}: {res.x}, Value: {res.func_vals[-1]}\n")

        print("Starting NN Optimization")
        with open(output_file, 'w') as output:
            output.write("RESULTS\n---------\n")

        results = gp_minimize(func       = objective,
                              dimensions = self._search_space,
                              n_calls    = num_trials,
                              acq_func   = "EI",
                              callback   = [log_callback])

        with open(output_file, 'a') as output:
            output.write("Best parameters: {}\n".format(results.x))

        best_model = self._build_model(initial_learning_rate     = results.x[0],
                                       learning_decay_rate       = results.x[1],
                                       batch_size_log2           = results.x[2],
                                       assembly_CNN_stencil_size = results.x[3],
                                       assembly_CNN_filters_log2 = results.x[4],
                                       assembly_CNN_activation   = results.x[5],
                                       detector_CNN_stencil_size = results.x[6],
                                       detector_CNN_filters_log2 = results.x[7],
                                       detector_CNN_activation   = results.x[8],
                                       CNN_dropout               = results.x[9],
                                       num_dens_layers           = results.x[10],
                                       dens_layers               = [(results.x[11 + i*3],
                                                                     results.x[12 + i*3],
                                                                     results.x[13 + i*3])
                                                                    for i in range(results.x[10])
                ])

        return best_model


    def _set_objective_function(self) -> None:
        """ Helper method for setting the objective function
        """

        @use_named_args(dimensions=self._search_space)
        def objective(**params):

            rms = []
            kf = KFold(n_splits=self.number_of_folds, shuffle=True)
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.series_collection)):
                print(f"Starting fold {fold + 1}/{self.number_of_folds}...")

                training_set   = [self.series_collection[i] for i in train_idx]
                validation_set = [self.series_collection[i] for i in val_idx]

                model = self._build_model(initial_learning_rate     = params["initial_learning_rate"],
                                          learning_decay_rate       = params["learning_decay_rate"],
                                          batch_size_log2           = params["batch_size_log2"],
                                          assembly_CNN_stencil_size = params["assembly_CNN_stencil_size"],
                                          assembly_CNN_filters_log2 = params["assembly_CNN_filters_log2"],
                                          assembly_CNN_activation   = params["assembly_CNN_activation"],
                                          detector_CNN_stencil_size = params["detector_CNN_stencil_size"],
                                          detector_CNN_filters_log2 = params["detector_CNN_filters_log2"],
                                          detector_CNN_activation   = params["detector_CNN_activation"],
                                          CNN_dropout               = params["CNN_dropout"],
                                          num_dens_layers           = params["num_dens_layers"],
                                          dens_layers               = [(params[f"dense_layer_{i}_neurons"],
                                                                        params[f"dense_layer_{i}_activation"],
                                                                        params[f"dense_layer_{i}_dropout"])
                                                                       for i in range(len(self.dimensions.dens_layers))
                ])

                print("Training started...")
                model.train(training_set, num_procs=self.num_procs)
                print("Training completed.")

                print("Validating model...")
                measured  = np.asarray([[series[0][self.predicted_feature]] for series in validation_set])
                predicted = model.predict(validation_set)
                diff      = measured - predicted
                fold_rms  = np.sqrt(np.dot(diff.flatten(), diff.flatten())) / float(len(diff))
                print(f"Fold {fold + 1} RMS: {fold_rms}")

                rms.append(fold_rms)

            average_rms = sum(rms) / len(rms)
            print(f"Optimization step completed. Average RMS across folds: {average_rms}")
            return average_rms

        self._objective = objective


    def _build_search_space(self) -> None:
        """ Helper method for constructing the optimization search space
        """

        dims  = self.dimensions
        space = [Real(       dims.initial_learning_rate[0],     dims.initial_learning_rate[1],     name="initial_learning_rate"    ),
                 Real(       dims.learning_decay_rate[0],       dims.learning_decay_rate[1],       name="learning_decay_rate"      ),
                 Integer(    dims.batch_size_log2[0],           dims.batch_size_log2[1],           name="batch_size_log2"          ),
                 Integer(    dims.assembly_CNN_stencil_size[0], dims.assembly_CNN_stencil_size[1], name="assembly_CNN_stencil_size"),
                 Integer(    dims.assembly_CNN_filters_log2[0], dims.assembly_CNN_filters_log2[1], name="assembly_CNN_filters_log2"),
                 Categorical(dims.assembly_CNN_activation,                                         name="assembly_CNN_activation"  ),
                 Integer(    dims.detector_CNN_stencil_size[0], dims.detector_CNN_stencil_size[1], name="detector_CNN_stencil_size"),
                 Integer(    dims.detector_CNN_filters_log2[0], dims.detector_CNN_filters_log2[1], name="detector_CNN_filters_log2"),
                 Categorical(dims.detector_CNN_activation,                                         name="detector_CNN_activation"  ),
                 Real(       dims.CNN_dropout[0],               dims.CNN_dropout[1],               name="CNN_dropout"              ),
                 Integer(    1,                                 len(dim.dens_layers),              name="num_dens_layers"          )]

        for i, layer in enumerate(dim.dens_layers):
            space.append(Integer(    layer.neurons[0], layer.neurons[1], name=f"dense_layer_{i}_neurons"   ))
            space.append(Categorical(layer.activation,                   name=f"dense_layer_{i}_activation"))
            space.append(Real(       layer.dropout[0], layer.dropout[1], name=f"dense_layer_{i}_dropout"   ))

        self._search_space = space


class CNNOptunaOptimizer(CNNOptimizer):
    """ A CNN model hyper parameter optimizer using optuna

    Parameters
    ----------
    dimensions : Dimensions
        The degrees of freedom and bounds for the optimization search
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding feature processing strategy
    predicted_feature : str
        The string specifying the feature to be predicted
    series_collection : SeriesCollection
        The input state series collection which to predict outputs for
    num_procs : int
        The number of parallel processors to use when reading data from the HDF5
    test_fraction : float
        The fraction of training data to withold for testing
    number_of_folds : int
        The number of folds to use for cross-fold validation
    epoch_limit : int
        The limit on the number of training epochs conducted during training
    convergence_criteria : float
        The convergence criteria for training
    convergence_patience : int
        Number of epochs with no improvement (i.e. error improves by greater than the convergence_criteria)
        after which training will be stopped
    biasing_model : Optional[PredictionStrategy]
        A model that is used to provide an initial prediction of the predicted output, acting ultimately as an initial bias
    """

    def __init__(self,
                 dimensions:           Dimensions,
                 input_features:       Dict[str, FeatureProcessor],
                 predicted_feature:    str,
                 series_collection:    SeriesCollection,
                 num_procs:            int = 1,
                 test_fraction:        int = 0.2,
                 number_of_folds:      int = 5,
                 epoch_limit:          int = 1000,
                 convergence_criteria: float = 1E-14,
                 convergence_patience: int = 100,
                 biasing_model:        Optional[PredictionStrategy] = None):
        self.dimensions           = dimensions
        self.input_features       = input_features
        self.predicted_feature    = predicted_feature
        self.series_collection    = series_collection
        self.num_procs            = num_procs
        self.test_fraction        = test_fraction
        self.number_of_folds      = number_of_folds
        self.epoch_limit          = epoch_limit
        self.convergence_criteria = convergence_criteria
        self.convergence_patience = convergence_patience
        self.biasing_model        = biasing_model

        self._set_objective_function()


    def optimize(self, num_trials: int, output_file: str) -> PredictionStrategy:

        def log_progress(study, trial):
            with open(output_file, "a") as output:
                output.write(f"Trial {trial.number}: Params: {trial.params}, Value: {trial.value}\n")
            print(f"Trial {trial.number} completed: Value: {trial.value}")

        print("Starting NN Optimization")
        with open(output_file, 'w') as output:
            output.write("RESULTS\n---------\n")

        study = optuna.create_study(direction='minimize')

        study.optimize(self._objective, n_trials=num_trials, callbacks=[log_progress])

        with open(output_file, 'a') as output:
            output.write(f"Best parameters: {study.best_params}\n")

        print(f"Best parameters: {study.best_params}")

        best_model = self._build_model(initial_learning_rate     = study.best_params["initial_learning_rate"],
                                       learning_decay_rate       = study.best_params["learning_decay_rate"],
                                       batch_size_log2           = study.best_params["batch_size_log2"],
                                       assembly_CNN_stencil_size = study.best_params["assembly_CNN_stencil_size"],
                                       assembly_CNN_filters_log2 = study.best_params["assembly_CNN_filters_log2"],
                                       assembly_CNN_activation   = study.best_params["assembly_CNN_activation"],
                                       detector_CNN_stencil_size = study.best_params["detector_CNN_stencil_size"],
                                       detector_CNN_filters_log2 = study.best_params["detector_CNN_filters_log2"],
                                       detector_CNN_activation   = study.best_params["detector_CNN_activation"],
                                       CNN_dropout               = study.best_params["CNN_dropout"],
                                       num_dens_layers           = study.best_params["num_dens_layers"],
                                       dens_layers               = [(study.best_params[f"dense_layer_{i}_neurons"],
                                                                     study.best_params[f"dense_layer_{i}_activation"],
                                                                     study.best_params[f"dense_layer_{i}_dropout"])
                                                                     for i in range(study.best_params["num_dens_layers"])])
        return best_model


    def _set_objective_function(self) -> None:
        """ Helper method for setting the objective function
        """

        dims = self.dimensions
        def objective(trial):
            initial_learning_rate     = trial.suggest_float(      "initial_learning_rate",         dims.initial_learning_rate[0],     dims.initial_learning_rate[1], log=True)
            learning_decay_rate       = trial.suggest_float(      "learning_decay_rate",           dims.learning_decay_rate[0],       dims.learning_decay_rate[1]            )
            batch_size_log2           = trial.suggest_int(        "batch_size_log2",               dims.batch_size_log2[0],           dims.batch_size_log2[1]                )
            assembly_CNN_stencil_size = trial.suggest_int(        "assembly_CNN_stencil_size",     dims.assembly_CNN_stencil_size[0], dims.assembly_CNN_stencil_size[1]      )
            assembly_CNN_filters_log2 = trial.suggest_int(        "assembly_CNN_filters_log2",     dims.assembly_CNN_filters_log2[0], dims.assembly_CNN_filters_log2[1]      )
            assembly_CNN_activation   = trial.suggest_categorical("assembly_CNN_activation",       dims.assembly_CNN_activation                                              )
            detector_CNN_stencil_size = trial.suggest_int(        "detector_CNN_stencil_size",     dims.detector_CNN_stencil_size[0], dims.detector_CNN_stencil_size[1]      )
            detector_CNN_filters_log2 = trial.suggest_int(        "detector_CNN_filters_log2",     dims.detector_CNN_filters_log2[0], dims.detector_CNN_filters_log2[1]      )
            detector_CNN_activation   = trial.suggest_categorical("detector_CNN_activation",       dims.detector_CNN_activation                                              )
            CNN_dropout               = trial.suggest_float(      "CNN_dropout",                   dims.CNN_dropout[0],           dims.CNN_dropout[1]                        )
            num_dens_layers           = trial.suggest_int(        "num_dens_layers",               dims.num_dens_layers[0],       dims.num_dens_layers[1]                    )
            dens_layers               = [(trial.suggest_int(        f"dense_layer_{i}_neurons",    layer.neurons[0], layer.neurons[1]),
                                          trial.suggest_categorical(f"dense_layer_{i}_activation", layer.activation                  ),
                                          trial.suggest_float(      f"dense_layer_{i}_dropout",    layer.dropout[0], layer.dropout[1]))
                                          for i, layer in enumerate(self.dimensions.dens_layers[:num_dens_layers])]

            rms = []
            kf = KFold(n_splits=self.number_of_folds, shuffle=True)
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.series_collection)):
                print(f"Starting fold {fold + 1}/{self.number_of_folds}...")

                training_set   = [self.series_collection[i] for i in train_idx]
                validation_set = [self.series_collection[i] for i in val_idx]

                model = self._build_model(initial_learning_rate     = initial_learning_rate,
                                          learning_decay_rate       = learning_decay_rate,
                                          batch_size_log2           = batch_size_log2,
                                          assembly_CNN_stencil_size = assembly_CNN_stencil_size,
                                          assembly_CNN_filters_log2 = assembly_CNN_filters_log2,
                                          assembly_CNN_activation   = assembly_CNN_activation,
                                          detector_CNN_stencil_size = detector_CNN_stencil_size,
                                          detector_CNN_filters_log2 = detector_CNN_filters_log2,
                                          detector_CNN_activation   = detector_CNN_activation,
                                          CNN_dropout               = CNN_dropout,
                                          num_dens_layers           = num_dens_layers,
                                          dens_layers               = dens_layers)

                print("Training started...")
                model.train(training_set, num_procs=self.num_procs)
                print("Training completed.")

                print("Validating model...")
                measured  = np.asarray([[series[0][self.predicted_feature]] for series in validation_set])
                predicted = model.predict(validation_set)
                diff      = measured - predicted
                fold_rms  = np.sqrt(np.dot(diff.flatten(), diff.flatten())) / float(len(diff))
                print(f"Fold {fold + 1} RMS: {fold_rms}")
                rms.append(fold_rms)

            average_rms = sum(rms) / len(rms)
            print(f"Optimization step completed. Average RMS across folds: {average_rms}")
            return average_rms

        self._objective = objective
