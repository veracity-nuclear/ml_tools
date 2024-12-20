from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import os
from math import isclose
from decimal import Decimal
import h5py
import numpy as np

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module
import tensorflow as tf
from tensorflow.keras import KerasTensor
import tensorflow.keras.layers
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

from ml_tools.model.state import StateSeries
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import FeatureProcessor


valid_layer_types = ['Dense']

valid_activation_functions = ['elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'mish',
                              'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh']

# Pylint mistakenly interpretting layer_group["activation_function"][()] as an HDF5 Group
# and subsequently complaining that it has no "decode" member
# pylint: disable=no-member

class Layer(ABC):
    """ Abstract class for neural network layers. Not meant to be instantiated directly.

    Parameters
    ----------
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).

    Attributes
    ----------
    dropout_rate : float
        Dropout rate for the layer.
    """

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate: float) -> None:
        assert 0.0 <= dropout_rate <= 1.0
        self._dropout_rate = dropout_rate


    def __init__(self, dropout_rate: float = 0.0) -> None:
        self.dropout_rate = dropout_rate

    @abstractmethod
    def __eq__(self, other: Layer) -> bool:
        """ Compare two layers for equality

        Parameters
        ----------
        other: Layer
            The other Layer to compare against

        Returns
        -------
        bool
            True if self and other are equal within the tolerance.  False otherwise

        Notes
        -----
        The relative tolerance is 1e-9 for float comparisons
        """

    @abstractmethod
    def __hash__(self) -> int:
        """ Generate a hash key for the layer

        Returns
        -------
        int
            The hash key corresponding to this layer

        Notes
        -----
        Hash generation is consistent with the 1e-9 float comparison equality relative tolerance
        """

    @abstractmethod
    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        """ Method for constructing the layer

        Parameters
        ----------
        input_tensor : KerasTensor
            The input tensor for the layer
        """

    @abstractmethod
    def save(self, group: h5py.Group) -> None:
        """ Method for saving the layer to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            The h5py group to save the layer to
        """

    @classmethod
    @abstractmethod
    def from_h5(cls, group: h5py.Group) -> Layer:
        """ Method for creating a new layer from an HDF5 Group

        Parameters
        ----------
        group : h5py.Group
            The h5py group to build the layer from

        Returns
        -------
        Layer
            The layer constructed from the HDF5 group
        """



class LayerSequence(Layer):
    """ A class for a sequence of layers

    A layer sequence does not require a dropout rate specification because
    this will be dictated by the final layer's dropout rate specification.
    If a dropout rate is provided, it will be ignored in favor of the final
    layer's dropout rate

    Parameters
    ----------
    layers : List[Layer]
        The list of layers that comprise the sequence

    Attributes
    ----------
    layers : List[Layer]
        The list of layers that comprise the sequence
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @layers.setter
    def layers(self, value: List[Layer]) -> None:
        assert len(value) > 0
        self._layers = value

    def __init__(self, layers: List[Layer]) -> None:
        super().__init__(0.0)
        self.layers = layers

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, LayerSequence) and
                  len(self.layers) == len(other.layers) and
                  all(s_layer == o_layer for s_layer, o_layer in zip(self.layers, other.layers))
                 )
               )

    def __hash__(self) -> int:
        return hash(tuple(self.layers))

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = input_tensor
        for layer in self.layers:
            x = layer.build(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type', data='LayerSequence', dtype=h5py.string_dtype())
        for i, layer in enumerate(self.layers):
            layer_group = group.create_group('layer_' + str(i))
            layer.save(layer_group)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> LayerSequence:
        layers = []
        layer_names = [key for key in group.keys() if key.startswith("layer_")]
        layer_names = sorted(layer_names, key=lambda x: int(x.split('_')[1]))
        for layer_name in layer_names:
            layer_group = group[layer_name]
            layer_type = layer_group['type'][()].decode('utf-8')
            if layer_type == 'Dense':
                layers.append(Dense.from_h5(layer_group))
            elif layer_type == 'LayerSequence':
                layers.append(LayerSequence.from_h5(layer_group))
            elif layer_type == 'CompoundLayer':
                layers.append(CompoundLayer.from_h5(layer_group))
        return cls(layers=layers)


class CompoundLayer(Layer):
    """ A class for compound / composite layers consisting layers that are executed in parallel

    This class effectively splits the input into the layer across multiple layers which will
    each execute in parallel and then merge their output at the end.

    A compound layer does require a dropout rate on account of the merged outputs.  If any
    of the composite layers are provided a drop out rate, said rate will be ignored in favor
    of the compound layer's dropout rate.

    Also, input features need not be "exclusive" to a given layer, but rather may be used by multiple
    constituent layers.

    Parameters
    ----------
    layers : List[Layer]
        The list of constituent layers that will be executed in parallel
    input_specifications : List[List[int]]
        The list of input indices each layer should use to pull from the incoming input
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).

    Attributes
    ----------
    layers : List[Layer]
        The list of constituent layers that will be executed in parallel
    input_specifications : List[List[int]]
        The list of input indices each layer should use to pull from the incoming input
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @property
    def input_specifications(self) -> List[List[int]]:
        return self._input_specifications

    def __init__(self,
                 layers:               List[Layer],
                 input_specifications: List[Union[slice, List[int]]],
                 dropout_rate:         float = 0.0) -> None:

        super().__init__(dropout_rate)

        assert len(layers) > 0
        assert len(layers) == len(input_specifications)

         # Input layer length is not known until at build
        assert all(not(spec.stop is None) for spec in input_specifications if isinstance(spec, slice))

        self._layers = layers

        self._input_specifications = []
        for specification in input_specifications:
            if isinstance(specification, slice):
                specification = list(range(specification.start if specification.start is not None else 0,
                                           specification.stop,
                                           specification.step  if specification.step is not None else 1))
                self._input_specifications.append(specification)
            else:
                self._input_specifications.append(specification)

    def __eq__(self, other: Any) -> bool:
        return (self is other or
                 (isinstance(other, CompoundLayer) and
                  isclose(self.dropout_rate, other.dropout_rate) and
                  len(self.layers) == len(other.layers) and
                  all(s_layer == o_layer for s_layer, o_layer in zip(self.layers, other.layers))
                 )
               )

    def __hash__(self) -> int:
        return hash(tuple(self.layers),
                     tuple(tuple(specification) for specification in self.input_specifications),
                     Decimal(self.dropout_rate).quantize(Decimal('1e-9')))

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        @register_keras_serializable()
        def gather_indices(x, indices):
            return tf.gather(x, indices, axis=-1)

        assert all(index < input_tensor.shape[1] for spec in self.input_specifications for index in spec)
        split_inputs = [tensorflow.keras.layers.Lambda(gather_indices, arguments={'indices': indices})(input_tensor)
                        for indices in self.input_specifications]

        outputs = [layer.build(split) for layer, split in zip(self._layers, split_inputs)]

        x = tensorflow.keras.layers.Concatenate(axis=-1)(outputs)

        if self._dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(self._dropout_rate)(x)

        return x


    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type', data='CompoundLayer', dtype=h5py.string_dtype())
        specs_array = np.array([np.array(inner, dtype=np.int32) for inner in self.input_specifications], dtype=object)
        group.create_dataset('input_specifications', data=specs_array, dtype=h5py.vlen_dtype(np.int32))
        for i, layer in enumerate(self.layers):
            layer_group = group.create_group('layer_' + str(i))
            layer.save(layer_group)
        group.attrs['dropout_rate'] = self._dropout_rate


    @classmethod
    def from_h5(cls, group: h5py.Group) -> CompoundLayer:
        input_specifications = [list(item) for item in group['input_specifications'][:]]
        layers = []
        layer_names = [key for key in group.keys() if key.startswith("layer_")]
        layer_names = sorted(layer_names, key=lambda x: int(x.split('_')[1]))
        for layer_name in layer_names:
            layer_group = group[layer_name]
            layer_type = layer_group['type'][()].decode('utf-8')
            if layer_type == 'Dense':
                layers.append(Dense.from_h5(layer_group))
            elif layer_type == 'LayerSequence':
                layers.append(LayerSequence.from_h5(layer_group))
            elif layer_type == 'CompoundLayer':
                layers.append(CompoundLayer.from_h5(layer_group))
        dropout_rate = group.attrs.get('dropout_rate', 0.0)
        return cls(layers=layers, input_specifications=input_specifications, dropout_rate=dropout_rate)


class Dense(Layer):
    """ A Dense Neural Network Layer

    Parameters
    ----------
    units : init
        Number of nodes (i.e. neurons) to use in the dense layer
    activation : str
        Activation function to use
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).

    Attributes
    ----------
    units : init
        Number of nodes (i.e. neurons) to use in the dense layer
    activation : str
        Activation function to use
    """

    @property
    def units(self) -> int:
        return self._units

    @units.setter
    def units(self, units: int) -> None:
        assert units > 0, f"units = {units}"
        self._units = units

    @property
    def activation(self) -> str:
        return self._activation

    @activation.setter
    def activation(self, activation: str) -> None:
        assert activation in valid_activation_functions, f"activation = {activation}"
        self._activation = activation


    def __init__(self, units: int, activation: str, dropout_rate: float = 0.):
        super().__init__(dropout_rate)
        self.units      = units
        self.activation = activation

    def __eq__(self, other: Layer) -> bool:
        return (self is other or
                (isinstance(other, Dense) and
                 self.units == other.units and
                 self.activation == other.activation and
                 isclose(self.dropout_rate, other.dropout_rate, rel_tol=1e-9))
               )

    def __hash__(self) -> int:
        return hash(tuple(self.units,
                          self.activation,
                          Decimal(self.dropout_rate).quantize(Decimal('1e-9'))
                         ))

    def build(self, input_tensor: KerasTensor) -> KerasTensor:
        x = tf.keras.layers.Dense(units=self._units, activation=self._activation)(input_tensor)
        if self._dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(self._dropout_rate)(x)
        return x

    def save(self, group: h5py.Group) -> None:
        group.create_dataset('type'                , data='Dense', dtype=h5py.string_dtype())
        group.create_dataset('dropout_rate'        , data=self.dropout_rate)
        group.create_dataset('number_of_units'     , data=self.units)
        group.create_dataset('activation_function' , data=self.activation, dtype=h5py.string_dtype())

    @classmethod
    def from_h5(cls, group: h5py.Group) -> Dense:
        return Dense(units        =   int(group["number_of_units"    ][()]),
                     activation   =       group["activation_function"][()].decode('utf-8'),
                     dropout_rate = float(group["dropout_rate"       ][()]))


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
        The convergence criteria for training
    batch_size : int
        The training batch sizes
    """

    @property
    def layers(self) -> List[Layer]:
        return self._layer_sequence.layers

    @layers.setter
    def layers(self, layers: List[Layer]):
        assert len(layers) > 0
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
                 epoch_limit           : int=400,
                 convergence_criteria  : float=1E-14,
                 batch_size            : int=32) -> None:

        super().__init__()
        self.input_features         = input_features
        self.predicted_feature      = predicted_feature
        self.layers                 = [Dense(units=5, activation='relu')] if layers is None else layers
        self.initial_learning_rate  = initial_learning_rate
        self.learning_decay_rate    = learning_decay_rate
        self.epoch_limit            = epoch_limit
        self.convergence_criteria   = convergence_criteria
        self.batch_size             = batch_size

        self._model = None


    def train(self, train_data: List[StateSeries], test_data: Optional[List[StateSeries]] = None, num_procs: int = 1) -> None:

        assert all(len(series) == 1 for series in train_data), \
            "All State Series must be static statepoints (i.e. len(series) == 1)"
        assert test_data is None, "The Neural Network Prediction Strategy does not use test data"

        X = self.preprocess_inputs(train_data, num_procs)[:,0,:]
        y = self._get_targets(train_data)[:,0]

        input_tensor = tf.keras.Input(shape=(len(X[0]),))

        x = self._layer_sequence.build(input_tensor)
        output = tf.keras.layers.Dense(1)(x)

        self._model = tf.keras.Model(inputs=input_tensor, outputs=output)

        learning_rate_schedule = ExponentialDecay(initial_learning_rate = self.initial_learning_rate,
                                                  decay_steps           = self.epoch_limit,
                                                  decay_rate            = self.learning_decay_rate, staircase=True)

        self._model.compile(optimizer=Adam(learning_rate = learning_rate_schedule),
                                           loss          = MeanSquaredError(),
                                           metrics       = [MeanAbsoluteError()])

        early_stop = EarlyStopping(monitor              = 'mean_absolute_error',
                                   min_delta            = self.convergence_criteria,
                                   patience             = 5,
                                   verbose              = 1,
                                   mode                 = 'auto',
                                   restore_best_weights = True)

        self._model.fit(X, y, epochs=self.epoch_limit, batch_size=self.batch_size, callbacks=[early_stop])

    def _predict_one(self, state_series: StateSeries) -> np.ndarray:
        return self._predict_all([state_series])[0]


    def _predict_all(self, state_series: List[StateSeries]) -> np.ndarray:
        assert self.isTrained
        assert all(len(series) == 1 for series in state_series), \
            "All State Series must be static statepoints (i.e. len(series) == 1)"

        X = self.preprocess_inputs(state_series)[:,0,:]
        tf.convert_to_tensor(X, dtype=tf.float32)
        y = self._model.predict(X).flatten()
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

        self._model.save(file_name)

        with h5py.File(file_name, 'a') as h5_file:
            self.base_save_model(h5_file)
            h5_file.create_dataset('initial_learning_rate', data=self.initial_learning_rate)
            h5_file.create_dataset('learning_decay_rate',   data=self.learning_decay_rate)
            h5_file.create_dataset('epoch_limit',           data=self.epoch_limit)
            h5_file.create_dataset('convergence_criteria',  data=self.convergence_criteria)
            h5_file.create_dataset('batch_size',            data=self.batch_size)
            self._layer_sequence.save(h5_file.create_group('neural_network'))



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

        assert os.path.exists(file_name), f"file name = {file_name}"

        with h5py.File(file_name, 'r') as h5_file:
            self.base_load_model(h5_file)
            self.initial_learning_rate = float( h5_file['initial_learning_rate'][()] )
            self.learning_decay_rate   = float( h5_file['learning_decay_rate'][()]   )
            self.epoch_limit           = int(   h5_file['epoch_limit'][()]           )
            self.convergence_criteria  = float( h5_file['convergence_criteria'][()]  )
            self.batch_size            = int(   h5_file['batch_size'][()]            )
            self._layer_sequence       = LayerSequence.from_h5(h5_file['neural_network'])

        self._model = load_model(file_name)


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
        assert os.path.exists(file_name), f"file name = {file_name}"

        new_model = cls({}, None)
        new_model.load_model(file_name)

        return new_model
