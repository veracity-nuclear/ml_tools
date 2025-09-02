from __future__ import annotations
from typing import List, Dict, Type, Optional
import os
import h5py
import numpy as np

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

from ml_tools.model.state import SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import FeatureProcessor
from ml_tools.model.nn_strategy.layer import Layer, gather_indices
from ml_tools.model.nn_strategy.graph_conv import GraphSAGEConv
from ml_tools.model.nn_strategy.layer_sequence import LayerSequence
from ml_tools.model.nn_strategy.dense import Dense


class NNStrategy(PredictionStrategy):
    """ A concrete class for a Neural-Network-based prediction strategy

    This prediction strategy is only intended for use with static State-Points, meaning
    non-temporal series, or said another way, State Series with series lengths of one.

    Parameters
    ----------
    input_features : Dict[str, FeatureProcessor]
        A dictionary specifying the input features of this model and their corresponding feature processing strategy
    predicted_feature : str
        The string specifying the feature to be predicted
    layers : List[Layer]
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

    Attributes
    ----------
    layers : LayerSequence
        The hidden layers of the neural network
    initial_learning_rate : float
        The initial learning rate of the training
    learning_decay_rate : float
        The decay rate of the learning using Exponential Decay
    epoch_limit : int
        The limit on the number of training epochs conducted during training
    convergence_criteria : float
        The convergence precision criteria for training
    convergence_patience : int
        Number of epochs with no improvement (i.e. error improves by greater than the convergence_criteria)
        after which training will be stopped
    batch_size : int
        The training batch sizes
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layer_sequence.layers

    @layers.setter
    def layers(self, layers: List[Layer]):
        assert len(layers) > 0, f"len(layers) = {len(layers)}"
        self._layer_sequence = LayerSequence(layers)

    @property
    def initial_learning_rate(self) -> float:
        return self._initial_learning_rate

    @initial_learning_rate.setter
    def initial_learning_rate(self, initial_learning_rate: float):
        assert initial_learning_rate >= 0., f"initial_learning_rate = {initial_learning_rate}"
        self._initial_learning_rate = initial_learning_rate

    @property
    def learning_decay_rate(self) -> float:
        return self._learning_decay_rate

    @learning_decay_rate.setter
    def learning_decay_rate(self, learning_decay_rate: float):
        assert 0. <= learning_decay_rate <= 1., f"learning_decay_rate = {learning_decay_rate}"
        self._learning_decay_rate = learning_decay_rate

    @property
    def epoch_limit(self) -> int:
        return self._epoch_limit

    @epoch_limit.setter
    def epoch_limit(self, epoch_limit: int):
        assert epoch_limit > 0, f"epoch_limit = {epoch_limit}"
        self._epoch_limit = epoch_limit

    @property
    def convergence_criteria(self) -> float:
        return self._convergence_criteria

    @convergence_criteria.setter
    def convergence_criteria(self, convergence_criteria: float):
        assert convergence_criteria > 0., f"convergence_criteria = {convergence_criteria}"
        self._convergence_criteria = convergence_criteria

    @property
    def convergence_patience(self) -> int:
        return self._convergence_patience

    @convergence_patience.setter
    def convergence_patience(self, convergence_patience: int):
        assert convergence_patience > 0, f"convergence_patience = {convergence_patience}"
        self._convergence_patience = convergence_patience

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        assert batch_size > 0, f"batch_size = {batch_size}"
        self._batch_size = batch_size

    @property
    def isTrained(self) -> bool:
        return self._model is not None


    def __init__(self,
                 input_features        : Dict[str, FeatureProcessor],
                 predicted_feature     : str,
                 layers                : List[Layer]=None,
                 initial_learning_rate : float=0.01,
                 learning_decay_rate   : float=1.,
                 epoch_limit           : int=1000,
                 convergence_criteria  : float=1E-14,
                 convergence_patience  : int=100,
                 batch_size            : int=32) -> None:

        super().__init__()
        self.input_features         = input_features
        self.predicted_feature      = predicted_feature
        self.layers                 = [Dense(units=5, activation='relu')] if layers is None else layers
        self.initial_learning_rate  = initial_learning_rate
        self.learning_decay_rate    = learning_decay_rate
        self.epoch_limit            = epoch_limit
        self.convergence_criteria   = convergence_criteria
        self.convergence_patience   = convergence_patience
        self.batch_size             = batch_size

        self._model = None


    def train(self, train_data: SeriesCollection, test_data: Optional[SeriesCollection] = None, num_procs: int = 1) -> None:
        assert test_data is None, "The Neural Network Prediction Strategy does not use test data"
        assert all(len(series) == len(train_data[0]) for series in train_data)

        X = self.preprocess_inputs(train_data, num_procs)
        y = self._get_targets(train_data)

        input_tensor = tf.keras.layers.Input(shape=(None, len(X[0][0])))

        x = self._layer_sequence.build(input_tensor)
        output = tf.keras.layers.Dense(y.shape[2])(x)

        self._model = tf.keras.Model(inputs=input_tensor, outputs=output)

        learning_rate_schedule = ExponentialDecay(initial_learning_rate = self.initial_learning_rate,
                                                  decay_steps           = self.epoch_limit,
                                                  decay_rate            = self.learning_decay_rate, staircase=True)

        self._model.compile(optimizer=Adam(learning_rate = learning_rate_schedule),
                                           loss          = MeanSquaredError(),
                                           metrics       = [MeanAbsoluteError()])

        early_stop = EarlyStopping(monitor              =  'mean_absolute_error',
                                   min_delta            = self.convergence_criteria,
                                   patience             = self.convergence_patience,
                                   verbose              = 1,
                                   mode                 = 'auto',
                                   restore_best_weights = True)
        mask       = np.any(X != 0.0, axis=-1).astype(np.float32)
        dataset    = tf.data.Dataset.from_tensor_slices((X, y, mask))
        dataset    = dataset.batch(self.batch_size, drop_remainder=True)
        self._model.fit(dataset, epochs=self.epoch_limit, batch_size=self.batch_size, callbacks=[early_stop])

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        return self._predict_all([state_series])[0]

    def _predict_all(self, state_series: np.ndarray) -> np.ndarray:
        """ Doing predictions as a padded `_predict_all` allows for much more optimal parallelization
            as opposed to doing a for-loop of `_predict_one` calls over all series individually.
        """
        assert self.isTrained

        X = tf.convert_to_tensor(state_series, dtype=tf.float32)

        return self._model.predict(X)


    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """

        self._model.save(file_name + ".keras")

        with h5py.File(file_name + ".h5", 'w') as h5_file:
            self.base_save_model(h5_file)
            h5_file.create_dataset('initial_learning_rate', data=self.initial_learning_rate)
            h5_file.create_dataset('learning_decay_rate',   data=self.learning_decay_rate)
            h5_file.create_dataset('epoch_limit',           data=self.epoch_limit)
            h5_file.create_dataset('convergence_criteria',  data=self.convergence_criteria)
            h5_file.create_dataset('batch_size',            data=self.batch_size)
            self._layer_sequence.save(h5_file.create_group('neural_network'))


    @classmethod
    def read_from_file(cls: NNStrategy, file_name: str) -> Type[NNStrategy]:
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

        new_model = cls({}, None)

        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        keras_name = file_name[:-3] + ".keras"
        assert os.path.exists(keras_name), f"file name = {keras_name}"
        assert os.path.exists(file_name), f"file name = {file_name}"

        with h5py.File(file_name, 'r') as h5_file:
            new_model.base_load_model(h5_file)
            new_model.initial_learning_rate = float( h5_file['initial_learning_rate'][()] )
            new_model.learning_decay_rate   = float( h5_file['learning_decay_rate'][()]   )
            new_model.epoch_limit           = int(   h5_file['epoch_limit'][()]           )
            new_model.convergence_criteria  = float( h5_file['convergence_criteria'][()]  )
            new_model.batch_size            = int(   h5_file['batch_size'][()]            )
            new_model._layer_sequence       = LayerSequence.from_h5(h5_file['neural_network'])

        new_model._model = load_model(keras_name, custom_objects={
            "gather_indices": gather_indices,
            "GraphSAGEConv": GraphSAGEConv,
        })

        return new_model
