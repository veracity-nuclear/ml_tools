from __future__ import annotations
from typing import List, Dict, Type, Optional
import os
from math import isclose
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
from ml_tools.model import register_prediction_strategy
from ml_tools.model.feature_processor import FeatureProcessor
from ml_tools.model.nn_strategy.layer import Layer, gather_indices
from ml_tools.model.nn_strategy.layer_sequence import LayerSequence
from ml_tools.model.nn_strategy.dense import Dense
from ml_tools.model.nn_strategy.graph.sage import GraphSAGEConv
from ml_tools.model.nn_strategy.graph.gat import GraphAttentionConv


@register_prediction_strategy()  # registers under 'NNStrategy'
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
    convergence_patience : int
        Number of epochs with no improvement (i.e. error improves by greater than the convergence_criteria)
        after which training will be stopped
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
        assert 0. < learning_decay_rate <= 1., f"learning_decay_rate = {learning_decay_rate}"
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

        X = self.preprocess_inputs(train_data, num_procs)
        y = self._get_targets(train_data, num_procs=num_procs)

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

        # Mask the padded timesteps in the loss calculation to prevent training on them
        lengths = np.asarray([len(series) for series in train_data], dtype=np.int32)
        mask = np.zeros((len(train_data), y.shape[1]), dtype=np.float32)
        for i, L in enumerate(lengths):
            mask[i, :L] = 1.0

        dataset    = tf.data.Dataset.from_tensor_slices((X, y, mask))
        dataset    = dataset.batch(self.batch_size, drop_remainder=True)
        self._model.fit(dataset, epochs=self.epoch_limit, batch_size=self.batch_size, callbacks=[early_stop])

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        return self._predict_all([state_series])[0]

    def _predict_all(self, series_collection: np.ndarray) -> np.ndarray:
        assert self.isTrained

        X = tf.convert_to_tensor(series_collection, dtype=tf.float32)

        return self._model.predict(X)

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        assert isinstance(other, NNStrategy)
        return (len(self.layers) == len(other.layers)                   and
                all(a == b for a, b in zip(self.layers, other.layers))  and
                self.epoch_limit          == other.epoch_limit          and
                self.convergence_patience == other.convergence_patience and
                self.batch_size           == other.batch_size           and
                isclose(self.initial_learning_rate, other.initial_learning_rate, rel_tol=1e-9) and
                isclose(self.learning_decay_rate,   other.learning_decay_rate,   rel_tol=1e-9) and
                isclose(self.convergence_criteria,  other.convergence_criteria,  rel_tol=1e-9))


    def write_model_to_hdf5(self, h5_group: h5py.Group) -> None:
        """ A method for writing the model to an already opened HDF5 group or file

        Parameters
        ----------
        h5_group : h5py.Group
            The opened HDF5 group or file to which the model should be written
        """
        file_name = h5_group.file.filename
        keras_name = file_name.removesuffix(".h5") + ".keras" if file_name.endswith(".h5") else file_name + ".keras"

        self._model.save(keras_name)
        with open(keras_name, 'rb') as file:
            file_data = file.read()
            h5_group.create_dataset('serialized_keras_file', data=np.void(file_data))

        self.base_save_model(h5_group)
        h5_group.create_dataset('initial_learning_rate', data=self.initial_learning_rate)
        h5_group.create_dataset('learning_decay_rate',   data=self.learning_decay_rate)
        h5_group.create_dataset('epoch_limit',           data=self.epoch_limit)
        h5_group.create_dataset('convergence_criteria',  data=self.convergence_criteria)
        h5_group.create_dataset('batch_size',            data=self.batch_size)
        self._layer_sequence.save(h5_group.create_group('neural_network'))

    def load_model(self, h5_group: h5py.Group) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        h5_group : h5py.Group
            The opened HDF5 group or file from which the model should be loaded
        """
        file_name = h5_group.file.filename
        keras_name = file_name[:-3] + ".keras"

        self.base_load_model(h5_group)
        self.initial_learning_rate = float( h5_group['initial_learning_rate'][()] )
        self.learning_decay_rate   = float( h5_group['learning_decay_rate'][()]   )
        self.epoch_limit           = int(   h5_group['epoch_limit'][()]           )
        self.convergence_criteria  = float( h5_group['convergence_criteria'][()]  )
        self.batch_size            = int(   h5_group['batch_size'][()]            )
        self._layer_sequence       = LayerSequence.from_h5(h5_group['neural_network'])

        read_keras_h5 = not os.path.exists(keras_name)
        self.base_load_model(h5_group)
        if read_keras_h5:
            file_data = bytes(h5_group['serialized_keras_file'][()])
            with open(keras_name, 'wb') as file:
                file.write(file_data)

        self._model = load_model(keras_name, custom_objects={
            "gather_indices": gather_indices,
            "GraphSAGEConv": GraphSAGEConv,
            "GraphAttentionConv": GraphAttentionConv,
        })        

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
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"
        assert os.path.exists(file_name), f"file name = {file_name}"

        new_model = cls({}, None)
        new_model.load_model(h5py.File(file_name, "r"))
        return new_model

    @classmethod
    def from_dict(cls,
                  params:            Dict,
                  input_features:    Dict[str, FeatureProcessor],
                  predicted_feature: str,
                  biasing_model:     Optional[PredictionStrategy] = None) -> NNStrategy:

        nn_cfg = params.get('neural_network')
        if nn_cfg is None:
            if 'layers' in params:
                nn_cfg = { 'layers': params['layers'] }
            else:
                raise KeyError("NNStrategy.from_dict requires 'neural_network' or 'layers'")

        layers                = LayerSequence.from_dict(nn_cfg).layers
        initial_learning_rate = params.get("initial_learning_rate", 0.01)
        learning_decay_rate   = params.get("learning_decay_rate", 1.0)
        epoch_limit           = params.get("epoch_limit", 1000)
        convergence_criteria  = params.get("convergence_criteria", 1e-14)
        convergence_patience  = params.get("convergence_patience", 100)
        if "batch_size" in params:
            batch_size = params["batch_size"]
        elif "batch_size_log2" in params:
            batch_size = 2 ** int(params["batch_size_log2"])
        else:
            batch_size = 32

        instance = cls(input_features        = input_features,
                       predicted_feature     = predicted_feature,
                       layers                = layers,
                       initial_learning_rate = initial_learning_rate,
                       learning_decay_rate   = learning_decay_rate,
                       epoch_limit           = epoch_limit,
                       convergence_criteria  = convergence_criteria,
                       convergence_patience  = convergence_patience,
                       batch_size            = batch_size)
        if biasing_model is not None:
            instance.biasing_model = biasing_model
        return instance

    def to_dict(self) -> dict:
        return {'initial_learning_rate': self.initial_learning_rate,
                'learning_decay_rate':   self.learning_decay_rate,
                'epoch_limit':           self.epoch_limit,
                'convergence_criteria':  self.convergence_criteria,
                'convergence_patience':  self.convergence_patience,
                'batch_size':            self.batch_size,
                'neural_network':        self._layer_sequence.to_dict()}
