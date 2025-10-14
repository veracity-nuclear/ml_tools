from __future__ import annotations
from typing import Literal, Dict, Type, List
from abc import ABC, abstractmethod
import h5py

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable



LayerType  = Literal['Dense', 'PassThrough', 'LSTM', 'LayerSequence', 'CompoundLayer',
                     'SpatialConv', 'SpatialMaxPool', 'Transformer', 'GraphConv']
Activation = Literal['elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'mish',
                     'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh']


@register_keras_serializable()
def gather_indices(x, indices):
    """
    Gathers specified indices along the last axis of the input tensor.

    This must be defined as a top-level (free) function and
    decorated with `@register_keras_serializable()` to ensure it can be
    serialized and deserialized correctly by TensorFlow, especially when used
    in distributed or parallel execution contexts.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    indices : List[int]
        The list of indices to gather from the last dimension of the tensor.

    Returns
    -------
    tf.Tensor
        A tensor with values gathered from the specified indices.
    """
    return tf.gather(x, indices, axis=-1)



# Pylint mistakenly interpretting layer_group["activation_function"][()] as an HDF5 Group
# and subsequently complaining that it has no "decode" member
# pylint: disable=no-member
class Layer(ABC):
    """ Abstract class for neural network layers. Not meant to be instantiated directly.

    Parameters
    ----------
    dropout_rate : float, optional
        Dropout rate for the layer. Default is 0.0 (no dropout).
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout

    Attributes
    ----------
    dropout_rate : float
        Dropout rate for the layer.
    batch_normalize : bool
        Whether or not batch normalization will be performed on the layer output prior to dropout
    layer_normalize : bool
        Whether or not layer normalization will be performed on the layer output prior to dropout
    """

    _registry: Dict[str, Type[Layer]] = {} # Registry for child classes

    @classmethod
    def register_subclass(cls, layer_type: str) -> None:
        """ Method for registering a subclass for a specific layer type

        Parameters
        ----------
        layer_type : str
            The string corresponding to the layer type to be registered
        """
        def decorator(subclass: Type[Layer]):
            cls._registry[layer_type] = subclass
            return subclass
        return decorator

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate: float) -> None:
        assert 0.0 <= dropout_rate <= 1.0, f"dropout rate = {dropout_rate}"
        self._dropout_rate = dropout_rate

    @property
    def batch_normalize(self) -> bool:
        return self._batch_normalize

    @batch_normalize.setter
    def batch_normalize(self, batch_normalize: bool) -> None:
        self._batch_normalize = batch_normalize

    @property
    def layer_normalize(self) -> bool:
        return self._layer_normalize

    @layer_normalize.setter
    def layer_normalize(self, layer_normalize: bool) -> None:
        self._layer_normalize = layer_normalize

    def __init__(self,
                 dropout_rate:     float = 0.0,
                 batch_normalize:  bool = False,
                 layer_normalize:  bool = False) -> None:
        self.dropout_rate     = dropout_rate
        self.batch_normalize  = batch_normalize
        self.layer_normalize  = layer_normalize

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
    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """ Method for constructing the layer without any dropout or normalization

        Parameters
        ----------
        input_tensor : tf.Tensor
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

    @staticmethod
    def layers_from_h5(group: h5py.Group) -> List[Layer]:
        """ Create a list of Layers from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group containing layer data.

        Returns
        -------
        List[Layer]
            A list of Layer instances created from the HDF5 group.
        """
        layers = []
        layer_names = [key for key in group.keys() if key.startswith("layer_")]
        layer_names = sorted(layer_names, key=lambda x: int(x.split('_')[1]))

        for layer_name in layer_names:
            layer_group = group[layer_name]
            layer_type = layer_group['type'][()].decode('utf-8')
            if layer_type not in Layer._registry:
                raise ValueError(f"Unknown layer type: {layer_type}")
            layers.append(Layer._registry[layer_type].from_h5(layer_group))

        return layers

    @classmethod
    def from_dict(cls, data: Dict) -> "Layer":
        """ Default factory from a parameter dict.

        Subclasses may override for custom parsing; by default this
        assumes the dict keys match the constructor signature.
        """
        return cls(**data)

    def to_dict(self) -> Dict:
        """Serialize this layer into a dict.

        Subclasses should override to include their parameters. This default
        returns only the type name.
        """
        return {"type": self.__class__.__name__}


    @staticmethod
    def layers_from_dict(data: Dict) -> List[Layer]:
        """ Create a list of Layers from a dict.

        Parameters
        ----------
        data : Dict
            A dict containing layer specifications.

        Returns
        -------
        List[Layer]
            A list of Layer instances created from the parameter dicts.
        """

        layers = []
        layer_names = [key for key in data.keys() if key.startswith("layer_")]
        layer_names = sorted(layer_names, key=lambda x: int(x.split('_')[1]))

        for layer_name in layer_names:
            layer_type = data[layer_name]['type']
            layer_dict = {k: v for k, v in data[layer_name].items() if not(k == 'type')}
            if layer_type not in Layer._registry:
                raise ValueError(f"Unknown layer type: {layer_type}")
            layers.append(Layer._registry[layer_type].from_dict(layer_dict))

        return layers

