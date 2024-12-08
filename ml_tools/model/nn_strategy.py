from __future__ import annotations
from typing import List, Dict
from dataclasses import dataclass
import os
from math import isclose
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

from ml_tools.model.state import State, StateSeries
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import FeatureProcessor


valid_activation_functions = ['elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'mish',
                              'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh']


class NNStrategy(PredictionStrategy):
    """ A concrete class for a Neural-Network-based prediction strategy

    This prediction strategy is only intended for use with static State-Points, meaning
    non-temporal series, or said another way, State Series with series lengths of one.

    Attributes
    ----------
    hidden_layers : List[Layer]
        The hidden layers of the neural network
    initial_learning_rate : float
        The initial learning rate of the training
    learning_decay_rate : float
        The decay rate of the learning using Exponential Decay
    epoch_limit : int
        The limit on the number of training epochs conducted during training
    convergence_criteria : float
        The convergence criteria for training
    batch_size : int
        The training batch sizes
    """

    @dataclass
    class Layer:
        """ Basic data class for dense layers
        """
        number_of_nodes: int
        dropout_rate: float = 0.
        activation_function: str = 'tanh' # This is good for regression

        def __post_init__(self):
            assert isinstance(self.number_of_nodes, int), "number of nodes must be an integer"
            assert isinstance(self.dropout_rate, float), "dropout rate must be a float"
            assert isinstance(self.activation_function, str), "activation function must be a string"
            assert self.number_of_nodes > 0, "number of nodes must be greater than 0"
            assert True if self.dropout_rate is None else self.dropout_rate >= 0. and self.dropout_rate < 1., \
                 "dropout rate must >= 0. and < 1."
            assert self.activation_function in valid_activation_functions, \
                 "activation function must be a valid activation function string"

        def __eq__(self, other: NNStrategy.Layer) -> bool:
            return (isclose(self.dropout_rate, other.dropout_rate) and
                    self.number_of_nodes     == other.number_of_nodes and
                    self.activation_function == other.activation_function)

    @property
    def hidden_layers(self) -> List[Layer]:
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, hidden_layers: List[Layer]):
        assert len(hidden_layers) > 0
        self._hidden_layers = hidden_layers

    @property
    def initial_learning_rate(self) -> float:
        return self._initial_learning_rate

    @initial_learning_rate.setter
    def initial_learning_rate(self, initial_learning_rate: float):
        assert initial_learning_rate >= 0.
        self._initial_learning_rate = initial_learning_rate

    @property
    def learning_decay_rate(self) -> float:
        return self._learning_decay_rate

    @learning_decay_rate.setter
    def learning_decay_rate(self, learning_decay_rate: float):
        assert learning_decay_rate >= 0. and learning_decay_rate <= 1.
        self._learning_decay_rate = learning_decay_rate

    @property
    def epoch_limit(self) -> int:
        return self._epoch_limit

    @epoch_limit.setter
    def epoch_limit(self, epoch_limit: int):
        assert epoch_limit > 0
        self._epoch_limit = epoch_limit

    @property
    def convergence_criteria(self) -> float:
        return self._convergence_criteria

    @convergence_criteria.setter
    def convergence_criteria(self, convergence_criteria: float):
        assert convergence_criteria > 0.
        self._convergence_criteria = convergence_criteria

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        assert batch_size > 0
        self._batch_size = batch_size

    @property
    def isTrained(self) -> bool:
        return self._nn is not None


    def __init__(self,
                 input_features        : Dict[str, FeatureProcessor],
                 predicted_feature     : str,
                 hidden_layers         : List[Layer]=None,
                 initial_learning_rate : float=0.01,
                 learning_decay_rate   : float=1.,
                 epoch_limit           : int=20,
                 convergence_criteria  : float=1E-4,
                 batch_size            : int=32) -> None:

        super().__init__()
        self.input_features         = input_features
        self.predicted_feature      = predicted_feature
        self.hidden_layers          = [NNStrategy.Layer(5)] if hidden_layers is None else hidden_layers
        self.initial_learning_rate  = initial_learning_rate
        self.learning_decay_rate    = learning_decay_rate
        self.epoch_limit            = epoch_limit
        self.convergence_criteria   = convergence_criteria
        self.batch_size             = batch_size

        self._nn = None


    def train(self, train_data: List[StateSeries], test_data: List[StateSeries] = None, num_procs: int = 1) -> None:

        assert all(len(series) == 1 for series in train_data) # All State Series must be static statepoints (i.e. len(series) == 1)
        assert test_data is None  # This model does not use Test Data as part of training

        X = self.preprocess_inputs(train_data, num_procs)[:,0,:]
        y = self._get_targets(train_data)[:,0]

        number_of_input_features = len(X[0])

        layers = [Dense(units       = layer.number_of_nodes,
                        activation  = layer.activation_function,
                        input_shape = (number_of_input_features,) if i == 0 else ())
                  for i, layer in enumerate(self.hidden_layers)]
        layers += [Dropout(layer.dropout_rate) for layer in self.hidden_layers if layer.dropout_rate]
        layers.append(Dense(1))

        learning_rate_schedule = ExponentialDecay(initial_learning_rate = self.initial_learning_rate,
                                                  decay_steps           = self.epoch_limit,
                                                  decay_rate            = self.learning_decay_rate, staircase=True)

        self._nn = Sequential(layers)

        self._nn.compile(optimizer=Adam(learning_rate = learning_rate_schedule),
                                        loss          = MeanSquaredError(),
                                        metrics       = [MeanAbsoluteError()])

        early_stop = EarlyStopping(monitor              = 'val_loss',
                                   min_delta            = self.convergence_criteria,
                                   patience             = 5,
                                   verbose              = 1,
                                   mode                 = 'auto',
                                   restore_best_weights = True)

        self._nn.fit(X, y, epochs=self.epoch_limit, batch_size=self.batch_size, callbacks=[early_stop])


    def _predict_one(self, state_series: StateSeries) -> np.ndarray:
        return self._predict_all([state_series])[0]


    def _predict_all(self, state_series: List[StateSeries]) -> np.ndarray:
        assert self.isTrained
        assert all(len(series) == 1 for series in state_series) # All State Series must be static statepoints (i.e. len(series) == 1)

        X = self.preprocess_inputs(state_series)[:,0,:]
        tf.convert_to_tensor(X, dtype=tf.float32)
        y = self._nn.predict(X).flatten()
        return y


    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        This method handles everything via the HDF5 file format, so any file name specified
        that doesn't end with '.h5' will be assumed to end with '.h5'

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """

        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"

        self._nn.save(file_name)

        with h5py.File(file_name, 'a') as h5_file:
            self.base_save_model(h5_file)
            h5_file.create_dataset('initial_learning_rate', data=self.initial_learning_rate)
            h5_file.create_dataset('learning_decay_rate',   data=self.learning_decay_rate)
            h5_file.create_dataset('epoch_limit',           data=self.epoch_limit)
            h5_file.create_dataset('convergence_criteria',  data=self.convergence_criteria)
            h5_file.create_dataset('batch_size',            data=self.batch_size)

            hidden_layers_group = h5_file.create_group('hidden_layers')
            for i, layer in enumerate(self.hidden_layers):
                layer_group = hidden_layers_group.create_group('layer_' + str(i))
                layer_group.create_dataset('number_of_nodes'     , data=layer.number_of_nodes)
                layer_group.create_dataset('dropout_rate'        , data=layer.dropout_rate   )
                layer_group.create_dataset('activation_function' , data=layer.activation_function, dtype=h5py.string_dtype())


    def load_model(self, file_name: str) -> None:
        """ A method for loading a trained model

        This method handles everything via the HDF5 file format, so any file name specified
        that doesn't end with '.h5' will be assumed to end with '.h5'

        Parameters
        ----------
        file_name : str
            The name of the file to load the model from
        """

        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"

        assert os.path.exists(file_name)

        with h5py.File(file_name, 'r') as h5_file:
            self.base_load_model(h5_file)
            self.initial_learning_rate = float( h5_file['initial_learning_rate'][()] )
            self.learning_decay_rate   = float( h5_file['learning_decay_rate'][()]   )
            self.epoch_limit           = int(   h5_file['epoch_limit'][()]           )
            self.convergence_criteria  = float( h5_file['convergence_criteria'][()]  )
            self.batch_size            = int(   h5_file['batch_size'][()]            )

            layers = []
            for i in range(len(h5_file['hidden_layers'])):
                layer_group = h5_file['hidden_layers']['layer_'+str(i)]

                # Pylint mistakenly interpretting layer_group["activation_function"][()] as an HDF5 Group
                # and subsequently complaining that it has no "decode" member
                # pylint: disable=no-member
                layers.append(NNStrategy.Layer(number_of_nodes     = int(layer_group["number_of_nodes"    ][()]),
                                               dropout_rate        = float(layer_group["dropout_rate"       ][()]),
                                               activation_function = layer_group["activation_function"][()].decode('utf-8')))
            self.hidden_layers = layers

        self._nn = load_model(file_name)


    @classmethod
    def read_from_hdf5(cls: NNStrategy, file_name: str) -> NNStrategy:
        """ A basic factory method for building NN Strategy from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the file from which to read and build the model

        Returns
        -------
        NNStrategy:
            The model from the hdf5 file
        """
        assert os.path.exists(file_name)

        new_nn = cls({}, None)
        new_nn.load_model(file_name)

        return new_nn
