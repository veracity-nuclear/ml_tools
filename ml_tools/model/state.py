from __future__ import annotations
from typing import List, Dict, Union, Optional, Tuple
import os
import random
import re
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
import pandas as pd
from ml_tools.utils.status_bar import StatusBar
from ml_tools.model.feature_perturbator import FeaturePerturbator


class State:
    """A class for storing and accessing generic state data

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        The features which describe the state
    """

    @property
    def features(self) -> Dict[str, np.ndarray]:
        return self._features

    def __init__(self, features: Dict[str, np.ndarray]):
        self._features = features

    def __getitem__(self, feature_name: str) -> np.ndarray:
        """Method for retrieving the feature data from a state

        Parameters
        ----------
        name : str
            The name of the feature to be retrieved

        Returns
        -------
        np.ndarray
            The feature data that was retrieved
        """

        assert (
            feature_name in self.features
        ), f"'{feature_name}' not found in state features. Available features: {list(self.features.keys())}"
        return self._features[feature_name]

    def __setitem__(self, feature_name: str, data_array: Union[np.ndarray, List[float]]) -> None:
        """Method for setting the feature data of a state

        Parameters
        ----------
        name : str
            The name of the feature to be set
        data_array : Union[np.ndarray, List[float]]
            The array to assign to the state
        """

        assert (
            feature_name in self.features
        ), f"'{feature_name}' not found in state features. Available features: {list(self.features.keys())}"
        data_array = data_array if isinstance(data_array, np.ndarray) else np.asarray(data_array)
        self._features[feature_name] = data_array

    def __repr__(self) -> str:
        """Method for printing the state

        Returns
        -------
        str
            The string representation of the state
        """

        s = "State:\n"
        for feature_name, values in self.features.items():
            s += f"  {feature_name}: {values}\n"
        return s

    def to_dataframe(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert the State into a Pandas DataFrame.

        Parameters
        ----------
        features : Optional[List[str]]
            List of features to extract to the dataframe, default is all
            features of the state

        Returns
        -------
        pd.DataFrame
            A DataFrame where each feature is a column, and each row
            corresponds to an element in the feature arrays.
        """
        features = self.features if features is None else {k: self.features[k] for k in features}

        flat_data = {}
        for feature_name, values in features.items():
            values = np.asarray(values)

            # Handle scalar values
            if values.ndim == 0:
                flat_data[feature_name] = values.item()
            elif len(values) == 1:
                flat_data[feature_name] = values[0]
            else:
                for i, v in enumerate(values.flat):
                    flat_data[f"{feature_name}_{i}"] = v

        return pd.DataFrame([flat_data])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> State:
        """Convert a Pandas DataFrame into a State object.
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be converted, where each feature is a column,
            and each row corresponds to an element in the feature arrays.
        Returns
        -------
        State
            The State object constructed from the dataframe.
        """

        assert not df.empty, "DataFrame is empty"

        features = {}
        for col in df.columns:
            vector_feature = re.match(r"^(.*)_(\d+)$", col)
            if vector_feature:
                base_feature, index = vector_feature.groups()
                if base_feature not in features:
                    features[base_feature] = []
                features[base_feature].append(df[col].values)
            else:
                features[col] = [df[col].values[0]]

        for key, feature in features.items():
            features[key] = np.array(feature).flatten()

        return cls(features)

    @staticmethod
    def from_hdf5(file_name: str, state: str, features: List[str]) -> State:
        """A factory method for extracting state feature data from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file from which to read and build the state from
        state : str
            The group in the HDF5 file which holds the feature data of the state
        features : List[str]
            The list of features expected to be read in for each state

        Returns
        -------
        State
            The state read from the data in the HDF5 file
        """

        assert os.path.exists(file_name), f"File does not exist: {file_name}"
        assert len(features) > 0, f"'len(features) = {len(features)}'"

        with h5py.File(file_name, "r") as h5_file:
            # Allow nested HDF5 paths for the state group (e.g., SERIES/STATE/ASSEM)
            try:
                state_group = h5_file[state]
            except KeyError as exc:
                raise AssertionError(f"'{state}' not found in {file_name}") from exc

            state_data = {}
            for feature in features:
                # Support nested feature paths relative to the state group (e.g., 'outputs/cips_index')
                try:
                    data = state_group[feature][()]
                except KeyError as exc:
                    raise AssertionError(
                        f"'{feature}' not found under '{state}' in {file_name}"
                    ) from exc
                feature = os.path.basename(feature)
                state_data[feature] = data
                if np.isscalar(state_data[feature]):
                    state_data[feature] = np.array([state_data[feature]])
                else:
                    state_data[feature] = state_data[feature].flatten()

        return State(state_data)

    @staticmethod
    def read_states_from_hdf5(
        file_name: str,
        features: List[str],
        states: Union[List[str], str] = None,
        silent: bool = False,
        num_procs: int = 1,
        random_sample_size: int = None,
    ) -> List[State]:
        """A factory method for building a collection of States from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file from which to read and build the states from
        features : List[str]
            The list of features expected to be read in for each state
        states : Union[List[str], str]
            The ordered list of groups in the HDF5 file which hold the feature data for each state.
            If no states are provided, it will be assumed that all first-level groups are the states
            and will read them all.
        silent : bool
            A flag indicating whether or not to display the progress bar to the screen
        num_procs : int
            The number of parallel processors to use when reading data from the HDF5
        random_sample_size : int
            Number of random state samples to draw from the list of specified states.
            If this argument is not provided, all states of the list will be considered.

        Returns
        -------
        List[State]
            A list of states read from the data in the HDF5 file
        """

        assert os.path.exists(file_name), f"File does not exist: {file_name}"
        assert len(features) > 0, f"'len(features) = {len(features)}'"
        assert num_procs > 0, f"'num_procs = {num_procs}'"

        if states is None:
            with h5py.File(file_name, "r") as h5_file:
                states = list(h5_file.keys())
        elif isinstance(states, str):
            states = [states]

        if random_sample_size:
            assert random_sample_size > 0, f"'random_sample_size = {random_sample_size}'"
            assert random_sample_size < len(
                states
            ), f"'random_sample_size = {random_sample_size}, len(states) = {len(states)}'"
            states = random.sample(states, random_sample_size)

        if not silent:
            print(f"Reading state data from: '{file_name}'")
            statusbar = StatusBar(len(states))

        state_data = []

        if num_procs == 1:
            for i, state in enumerate(states):
                state_data.append(State.from_hdf5(file_name, state, features))
                if not silent:
                    statusbar.update(i)

        else:

            def chunkify(states: List[str], chunk_size: int):
                for i in range(0, len(states), chunk_size):
                    yield states[i : i + chunk_size]

            chunk_size = max(1, len(states) // num_procs)
            chunks = list(chunkify(states, chunk_size))

            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                jobs = {
                    executor.submit(State.read_states_from_hdf5, file_name, features, chunk, silent=True): chunk
                    for chunk in chunks
                }

                completed = 0
                for job in as_completed(jobs):
                    result = job.result()
                    state_data.extend(result)
                    if not silent:
                        for _ in result:
                            statusbar.update(completed)
                            completed += 1

        if not silent:
            statusbar.finalize()

        return state_data

    @staticmethod
    def perturb_state(perturbators: Dict[str, FeaturePerturbator], state: State) -> State:
        """A method for perturbing the features of a given state

        Parameters
        ----------
        perturbators : Dict[str, FeaturePerturbator]
            The collection of perturbators to be applied with keys corresponding to the
            feature to be perturbed
        state : State
            The state to be perturbed

        Returns
        -------
        State
            A new state which is a perturbation of the original state
        """

        assert len(perturbators) > 0, f"'len(perturbators) = {len(perturbators)}'"
        assert all(feature in state.features for feature in perturbators.keys())

        perturbed_state = deepcopy(state)
        for feature, perturbator in perturbators.items():
            perturbed_state[feature] = perturbator.perturb(perturbed_state[feature])

        return perturbed_state

    @staticmethod
    def perturb_states(
        perturbators: Dict[str, FeaturePerturbator],
        states: Union[StateSeries, State],
        silent: bool = False,
        num_procs: int = 1,
    ) -> StateSeries:
        """A method for perturbing the features of a given collection of states

        Parameters
        ----------
        perturbators : Dict[str, FeaturePerturbator]
            The collection of perturbators to be applied with keys corresponding to the
            feature to be perturbed
        state : Union[StateSeries, State]
            The states to be perturbed
        silent : bool
            A flag indicating whether or not to display the progress bar to the screen
        num_procs : int
            The number of parallel processors to use when perturbing states

        Returns
        -------
        StateSeries
            A new states which are perturbations of the original states
        """

        assert len(states) > 0, f"'len(states) = {len(states)}'"
        assert len(perturbators) > 0, f"'len(perturbators) = {len(perturbators)}'"
        assert num_procs > 0, f"'num_procs = {num_procs}'"

        states = [states] if isinstance(states, State) else states

        if not silent:
            print("Perturbing state data")
            statusbar = StatusBar(len(states))
        state_data = []
        i = 0

        if num_procs == 1:
            for state in states:
                state_data.append(State.perturb_state(perturbators, state))
                if not silent:
                    statusbar.update(i)
                    i += 1

        else:

            def chunkify(states: StateSeries, chunk_size: int):
                for i in range(0, len(states), chunk_size):
                    yield states[i : i + chunk_size]

            chunk_size = max(1, len(states) // num_procs)
            chunks = list(chunkify(states, chunk_size))

            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                jobs = {executor.submit(State.perturb_state(perturbators, state)): chunk for chunk in chunks}

                completed = 0
                for job in as_completed(jobs):
                    result = job.result()
                    state_data.extend(result)
                    if not silent:
                        for _ in result:
                            statusbar.update(completed)
                            completed += 1

        if not silent:
            statusbar.finalize()

        return state_data


class StateSeries:
    """A class for storing and accessing a series of states

    Parameters
    ----------
    states : List[State]
        The list of states which describe the series

    Attributes
    ----------
    features : List[str]
        The list of features of the state series
    shape : Tuple[int, int]
        The shape of the state series, where the first element is the number of states
        and the second element is the number of features in each state
    """

    def __init__(self, states: List[State]):
        self.states = states

    def __getitem__(self, index: Union[int, slice]) -> Union[State, StateSeries]:
        if isinstance(index, slice):
            return StateSeries(self.states[index])
        return self.states[index]

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self):
        return iter(self.states)

    def __repr__(self) -> str:
        keys = list(self.states[0].features.keys())
        col_width = max((max(len(key) for key in keys) + 2, 11))

        # prints the column headers
        s = "StateSeries:\n"
        s += " State "
        for key in keys:
            s += f"{key:^{col_width}s}"

        # prints the features of the states in columns
        for i, state in enumerate(self.states):
            s += f"\n{i + 1:^7d}"
            for key in keys:
                value = state[key]
                if isinstance(value, np.ndarray):
                    value = value.squeeze()
                s += f"{value:^{col_width}.3e}"

            # if there are more than 10 states, print the last 3 states
            if i + 1 >= 5 and len(self.states) >= 10:
                s += "\n  ...  "
                for j in range(len(self.states) - 3, len(self.states)):
                    s += f"\n{j + 1:^7d}"
                    for key in keys:
                        value = self.states[j][key]
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        s += f"{value:^{col_width}.3e}"
                break

        s += "\n"
        return s

    def __add__(self, other: StateSeries) -> StateSeries:
        assert (
            self.features == other.features
        ), f"Features of the two StateSeries do not match: {self.features} != {other.features}"
        assert isinstance(other, StateSeries), f"'{other}' is not a StateSeries object"
        return StateSeries(self.states + other.states)

    @property
    def features(self) -> List[str]:
        if not self.states:
            return {}

        return list(self.states[0].features.keys())

    @property
    def shape(self) -> Tuple[int, int]:
        if not self.states:
            return (0, 0)

        num_states = len(self.states)
        num_features = len(self.states[0].features)
        return (num_states, num_features)

    def append(self, state: State) -> None:
        """Append a state to the series

        Parameters
        ----------
        state : State
            The state to be appended to the series
        """

        if self.features:
            assert (
                list(state.features.keys()) == self.features
            ), f"State features do not match: {list(state.features.keys())} != {self.features}"
        assert isinstance(state, State), f"'{state}' is not a State object"
        self.states.append(state)

    def extend(self, other: StateSeries) -> None:
        """Extend the current series with another StateSeries.

        Parameters
        ----------
        other : StateSeries
            The StateSeries to extend the current series with
        """

        assert (
            self.features == other.features
        ), f"Features of the two StateSeries do not match: {self.features} != {other.features}"
        assert isinstance(other, StateSeries), f"'{other}' is not a StateSeries object"
        self.states.extend(other.states)

    def pop(self) -> State:
        """Pop a state from the series

        Returns
        -------
        State
            The state that was popped from the series
        """

        assert len(self.states) > 0, "StateSeries is empty"
        return self.states.pop()

    def to_numpy(self, features: List[str] = None) -> np.ndarray:
        """Convert to a 2D NumPy array: rows are states, columns are features

        Parameters
        ----------
        features : List[str]
            The list of features to extract to the numpy array, default is all
            features of the state

        Returns
        -------
        np.ndarray
            A 2D NumPy array where each row corresponds to a state and each
            column corresponds to a feature.
        """
        if not self.states:
            return np.array([])

        if features is None:
            features = self.features

        data = [[state[k] for k in features] for state in self.states]
        return np.array(data)

    def mean(self, keys: List[str] = None) -> np.ndarray:
        """Calculate the mean of the features in the series

        Parameters
        ----------
        keys : List[str]
            The list of features to calculate the mean for, default is all
            features of the state

        Returns
        -------
        np.ndarray
            The mean of the features in the series
        """
        return self.to_numpy(keys).mean(axis=0)

    def std(self, keys: List[str] = None) -> np.ndarray:
        """Calculate the standard deviation of the features in the series

        Parameters
        ----------
        keys : List[str]
            The list of features to calculate the standard deviation for,
            default is all features of the state

        Returns
        -------
        np.ndarray
            The standard deviation of the features in the series
        """
        return self.to_numpy(keys).std(axis=0)

    def max(self, keys: List[str] = None) -> np.ndarray:
        """Calculate the maximum of the features in the series

        Parameters
        ----------
        keys : List[str]
            The list of features to calculate the maximum for, default is all
            features of the state

        Returns
        -------
        np.ndarray
            The maximum of the features in the series
        """
        return self.to_numpy(keys).max(axis=0)

    def min(self, keys: List[str] = None) -> np.ndarray:
        """Calculate the minimum of the features in the series

        Parameters
        ----------
        keys : List[str]
            The list of features to calculate the minimum for, default is all
            features of the state

        Returns
        -------
        np.ndarray
            The minimum of the features in the series
        """
        return self.to_numpy(keys).min(axis=0)

    @classmethod
    def from_hdf5(
        cls,
        file_name: str,
        features: List[str],
        state_series: List[str],
        silent: bool = False,
        num_procs: int = 1,
    ) -> StateSeries:
        """
        Build a StateSeries from an HDF5 file using a list of state groups.

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file from which to read and build the state series from
        features : List[str]
            The list of features expected to be read in for each state
        state_series : List[str]
            A list of HDF5 group paths, where each group path represents a single state.
            These states will be combined to form the StateSeries.
        silent : bool
            Whether to suppress progress output
        num_procs : int
            Number of parallel processors to use

        Returns
        -------
        StateSeries
            The StateSeries read from the HDF5 file
        """
        states = State.read_states_from_hdf5(
            file_name=file_name,
            features=features,
            states=state_series,
            silent=silent,
            num_procs=num_procs,
        )
        return cls(states)


    def to_hdf5(self, file_name: str, group_name: Optional[str] = "/") -> None:
        """Write the state series to an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file to write to
        group_name : str
            The group name in the HDF5 file to write the state series to
        """
        with h5py.File(file_name, "w") as h5_file:
            group = h5_file if group_name == "/" else h5_file.create_group(group_name)
            for i, state in enumerate(self.states):
                state_group = group.create_group(f"state_{i:06d}")
                for feature, data in state.features.items():
                    state_group.create_dataset(feature, data=data)

    @classmethod
    def from_csv(cls, file_name: str, features: List[str]) -> StateSeries:
        """A factory method for extracting state feature data from a CSV file

        Parameters
        ----------
        file_name : str
            The name of the CSV file from which to read and build the state from
        features : List[str]
            The list of features to be read in for each state

        Returns
        -------
        StateSeries
            A list of states read from the data in the CSV file
        """
        assert os.path.exists(file_name), f"File does not exist: {file_name}"
        assert len(features) > 0, f"'len(features) = {len(features)}'"

        df = pd.read_csv(file_name)
        return StateSeries.from_dataframe(df, features)

    def to_csv(self, file_name: str, features: Optional[List[str]] = None) -> None:
        """Write the state series to a CSV file

        Parameters
        ----------
        file_name : str
            The name of the CSV file to write to
        features : Optional[List[str]]
            The dictionary of features to be written to the CSV file
        """
        if features is None:
            features = self.features

        df = self.to_dataframe(features)
        df.to_csv(file_name, index=False)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, features: List[str] = None) -> StateSeries:
        """Convert a Pandas DataFrame into a State object.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be converted, where each feature is a column,
            and each row corresponds to an element in the feature arrays.
        features : List[str]
            The list of features to be read in for each state

        Returns
        -------
        State
            The State object constructed from the dataframe.
        """
        assert not df.empty, "DataFrame is empty"

        state_series = cls([])
        for _, group in df.groupby(level=0):
            filtered_group = group[features] if features else group
            state_series.append(State.from_dataframe(filtered_group))

        return state_series

    def to_dataframe(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert the State into a Pandas DataFrame.

        Parameters
        ----------
        features : Optional[List[str]]
            List of features to extract to the dataframe, default is all features of the state

        Returns
        -------
        pd.DataFrame
            A DataFrame where each feature is a column, and each row corresponds to an element in the feature arrays.
        """
        if features is None:
            features = self.features

        series_np = self.to_numpy(features)
        return pd.DataFrame(series_np, columns=features, index=None)


class SeriesCollection:
    """A class for storing and accessing a list of StateSeries

    Parameters
    ----------
    state_series_list : List[StateSeries]
        The list of StateSeries which describe the series

    Attributes
    ----------
    features : List[str]
        The list of features of the state series list, which is the same for all StateSeries
    """

    def __init__(self, state_series_list: List[StateSeries]):
        self.state_series_list = state_series_list

    def __getitem__(self, index: Union[int, slice]) -> Union[StateSeries, SeriesCollection]:
        if isinstance(index, slice):
            return SeriesCollection(self.state_series_list[index])
        return self.state_series_list[index]

    def __len__(self) -> int:
        return len(self.state_series_list)

    def __iter__(self):
        return iter(self.state_series_list)

    def __repr__(self) -> str:
        """Method for printing the state series list

        Returns
        -------
        str
            The string representation of the state series list
        """

        s = f"SeriesCollection of length {len(self)} with features: {self.features}\n"
        return s

    def __add__(self, other: SeriesCollection) -> SeriesCollection:
        assert (
            self.features == other.features
        ), f"Features of the two SeriesCollections do not match: {self.features} != {other.features}"
        assert isinstance(other, SeriesCollection), f"'{other}' is not a SeriesCollection object"
        if other is None:
            return self
        return SeriesCollection(self.state_series_list + other.state_series_list)

    @property
    def features(self) -> List[str]:
        if not self.state_series_list:
            return []

        return list(self.state_series_list[0][0].features.keys())

    def append(self, state_series: StateSeries) -> None:
        """Append a state series to the list

        Parameters
        ----------
        state_series : StateSeries
            The state series to be appended to the list
        """

        assert (
            state_series.features == self.features
        ), f"StateSeries features do not match: {state_series.features} != {self.features}"
        assert isinstance(state_series, StateSeries), f"'{state_series}' is not a StateSeries object"
        self.state_series_list.append(state_series)

    def extend(self, other: SeriesCollection) -> None:
        """Extend the current list with another SeriesCollection.

        Parameters
        ----------
        other : SeriesCollection
            The SeriesCollection to extend the current list with
        """

        assert (
            self.features == other.features
        ), f"Features of the two SeriesCollections do not match: {self.features} != {other.features}"
        assert isinstance(other, SeriesCollection), f"'{other}' is not a SeriesCollection object"
        self.state_series_list.extend(other.state_series_list)

    def random_sample(self, num_samples: int, seed: Optional[int] = None) -> SeriesCollection:
        """Return a random subset of this SeriesCollection.

        Parameters
        ----------
        num_samples : int
            Number of series to draw. Must be <= len(self).
        seed : Optional[int]
            Optional random seed for reproducibility.

        Returns
        -------
        SeriesCollection
            New SeriesCollection containing the sampled series.
        """
        assert num_samples <= len(self), \
            f"Cannot sample {num_samples} elements from SeriesCollection of length {len(self)}"

        rng = random.Random(seed) if seed is not None else random
        return SeriesCollection(rng.sample(self.state_series_list, num_samples))

    def train_test_split(self,
                         test_size: Union[int, float] = 0.2,
                         shuffle: bool = True,
                         seed: Optional[int] = None) -> Tuple[SeriesCollection, SeriesCollection]:
        """Split the collection into train/test SeriesCollections.

        Parameters
        ----------
        test_size : Union[int, float], optional
            If float, represents the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
        shuffle : bool, optional
            Whether to shuffle before splitting. Default True.
        seed : Optional[int], optional
            Random seed used when shuffling.

        Returns
        -------
        Tuple[SeriesCollection, SeriesCollection]
            (train_collection, test_collection) pair.
        """
        total = len(self)
        assert total >= 2, "Need at least two series to perform train/test split."

        if isinstance(test_size, float):
            assert 0.0 < test_size < 1.0, f"test_size fraction must be between 0 and 1, got {test_size}"
            test_count = max(1, int(round(total * test_size)))
        else:
            test_count = int(test_size)

        assert 0 < test_count < total, f"test_size must yield between 1 and {total - 1} samples."

        indices = list(range(total))
        if shuffle:
            rng = random.Random(seed) if seed is not None else random
            rng.shuffle(indices)

        test_indices = set(indices[:test_count])
        train = [self.state_series_list[i] for i in indices if i not in test_indices]
        test = [self.state_series_list[i] for i in indices[:test_count]]
        return SeriesCollection(train), SeriesCollection(test)

    @classmethod
    def from_hdf5(
        cls,
        file_name: str,
        features: Optional[List[str]] = None,
        series_collection: Optional[List[List[str]]] = None,
        silent: bool = False,
        num_procs: int = 1,
    ) -> SeriesCollection:
        """
        Build a SeriesCollection from an HDF5 file using a list of lists of state names.

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file from which to read and build the state series from
        features : Optional[List[str]]
            The list of features expected to be read in for each state. If None, all features
            in the HDF5 file will be read.
        series_collection : Optional[List[List[str]]]
            A list of lists of HDF5 group paths, where each group path represents a single state.
            The nested list structure defines the organization: each inner list becomes a StateSeries,
            and the collection of StateSeries forms the SeriesCollection. If None, assumes the file
            was written with to_hdf5() and reads the series_XXX/state_XXXXXX structure.
        silent : bool
            Whether to suppress progress output
        num_procs : int
            Number of parallel processors to use

        Returns
        -------
        SeriesCollection
            The SeriesCollection read from the HDF5 file
        """
        assert os.path.exists(file_name), f"File does not exist: {file_name}"

        # If series_collection is not provided, infer structure from file
        if series_collection is None or features is None:
            with h5py.File(file_name, "r") as h5_file:
                if series_collection is None:
                    series_collection = []
                    # Look for series_XXX groups
                    series_keys = sorted([k for k in h5_file.keys() if k.startswith("series_")])

                    for series_key in series_keys:
                        series_group = h5_file[series_key]
                        # Look for state_XXXXXX groups within each series
                        state_keys = sorted([k for k in series_group.keys() if k.startswith("state_")])
                        # Build full paths: series_XXX/state_XXXXXX
                        state_paths = [f"{series_key}/{state_key}" for state_key in state_keys]
                        series_collection.append(state_paths)

                # If features are not provided, infer them from the first state
                if features is None and series_collection[0]:
                    first_state_path = series_collection[0][0]
                    features = list(h5_file[first_state_path].keys())

        # Flatten all state names and keep track of their series indices
        flattened_states = []
        series_lengths = []
        for state_series in series_collection:
            flattened_states.extend(state_series)
            series_lengths.append(len(state_series))

        # Read all states in a single parallelized call
        all_states = State.read_states_from_hdf5(
            file_name=file_name,
            features=features,
            states=flattened_states,
            silent=silent,
            num_procs=num_procs,
        )

        # Partition the states back into StateSeries based on original structure
        state_series_list = []
        start_idx = 0
        for length in series_lengths:
            end_idx = start_idx + length
            series_states = all_states[start_idx:end_idx]
            state_series_list.append(StateSeries(series_states))
            start_idx = end_idx

        return cls(state_series_list)

    def to_hdf5(self, file_name: str) -> None:
        """Write the state series list to an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file to write to
        """
        with h5py.File(file_name, "w") as h5_file:
            for series_idx, series in enumerate(self.state_series_list):
                series_group_name = f"series_{series_idx:03d}"
                series_group = h5_file.create_group(series_group_name)

                # Use StateSeries.to_hdf5 logic for each series
                for state_idx, state in enumerate(series):
                    state_group = series_group.create_group(f"state_{state_idx:06d}")
                    for feature, data in state.features.items():
                        state_group.create_dataset(feature, data=data)

    @classmethod
    def from_csv(cls, file_name: str, features: List[str] = None) -> SeriesCollection:
        """A factory method for building a collection of StateSeries from a CSV file

        Parameters
        ----------
        file_name : str
            The name of the CSV file from which to read and build the state series from
        features : List[str]
            List of features to extract to the dataframe, default is all features of the state

        Returns
        -------
        SeriesCollection
            A list of state series read from the data in the CSV file
        """
        assert os.path.exists(file_name), f"File does not exist: {file_name}"

        df = pd.read_csv(file_name)
        if 'series_index' in df.columns and 'state_index' in df.columns:
            df = df.set_index(['series_index', 'state_index'])

        return cls.from_dataframe(df, features=features)

    def to_csv(self, file_name: str, features: List[str] = None) -> None:
        """Write the state series list to a CSV file

        Parameters
        ----------
        file_name : str
            The name of the CSV file to write to
        features : List[str]
            List of features to extract to the dataframe, default is all features of the state
        """
        if features is None:
            features = self.features

        df = self.to_dataframe(features)
        df.to_csv(file_name)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, features: List[str] = None) -> SeriesCollection:
        """Convert a Pandas DataFrame into a List of StateSeries

        "DataFrame index must be a MultiIndex with 'series_index' and 'state_index'"

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be converted.
        features : List[str]
            The list of features to be read in for each state

        Returns
        -------
        SeriesCollection
            The list of state series
        """
        assert not df.empty, "DataFrame is empty"
        assert isinstance(
            df.index, pd.MultiIndex
        ), "DataFrame index must be a MultiIndex with 'series_index' and 'state_index'"

        state_series_dict = {}

        for (series_idx, state_idx), state_df in df.groupby(level=["series_index", "state_index"]):
            state = State.from_dataframe(state_df.reset_index(drop=True))

            # If features are specified, filter the resulting state to only include those features
            if features:
                filtered_features = {k: state.features[k] for k in features if k in state.features}
                state = State(filtered_features)

            if series_idx not in state_series_dict:
                state_series_dict[series_idx] = []
            state_series_dict[series_idx].append(state)

        max_series_idx = max(state_series_dict.keys())
        return cls([state_series_dict.get(i) for i in range(max_series_idx + 1)])

    def to_dataframe(
        self,
        features: List[str] = None,
    ) -> pd.DataFrame:
        """Convert a List of StateSeries to a Pandas Dataframe

        Parameters
        ----------
        features : List[str]
            List of features to extract to the dataframe, default is all features of the state

        Returns
        -------
        pd.DataFrame
            The created Pandas Dataframe
        """
        if features is None:
            features = self.features
        series_dfs = []

        for series_idx, series in enumerate(self.state_series_list):
            if not series:
                continue

            state_dfs = [state.to_dataframe(features) for state in series]
            assert all(isinstance(df, pd.DataFrame) for df in state_dfs), "One or more states returned an invalid DataFrame"

            df = pd.concat(state_dfs, ignore_index=True)
            df.index = pd.MultiIndex.from_product([[series_idx], range(len(df))], names=["series_index", "state_index"])
            series_dfs.append(df)

        return pd.concat(series_dfs)
