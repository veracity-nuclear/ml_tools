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

    def to_dataframe(self, features: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """Convert the State into a Pandas DataFrame.

        Parameters
        ----------
        features : Optional[Dict[str, np.ndarray]]
            Dictionary of features to extract to the dataframe, default is all
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

            if len(values) == 1:
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
            assert state in h5_file.keys(), f"'{state}' not found in {file_name}"
            assert all(feature in h5_file[state].keys() for feature in features)

            state_data = {}
            for feature in features:
                data = h5_file[state][feature][()]
                feature = os.path.basename(feature)
                state_data[feature] = data
                if np.isscalar(state_data[feature]):
                    state_data[feature] = np.array([state_data[feature]])
                else:
                    state_data[feature] = state_data[feature].flatten()

        return State(state_data)

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
        states: Union[List[State], State],
        silent: bool = False,
        num_procs: int = 1,
    ) -> List[State]:
        """A method for perturbing the features of a given collection of states

        Parameters
        ----------
        perturbators : Dict[str, FeaturePerturbator]
            The collection of perturbators to be applied with keys corresponding to the
            feature to be perturbed
        state : Union[List[State], State]
            The states to be perturbed
        silent : bool
            A flag indicating whether or not to display the progress bar to the screen
        num_procs : int
            The number of parallel processors to use when perturbing states

        Returns
        -------
        List[State]
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

            def chunkify(states: List[State], chunk_size: int):
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
                    value = value.tolist()
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
        assert isinstance(other, StateSeries), f"'{other}' is not a StateSeries object"
        return StateSeries(self.states + other.states)

    @property
    def features(self) -> List[str]:
        """Get the features of the state series

        Returns
        -------
        List[str]
            The features of the state series
        """

        if not self.states:
            return {}

        return list(self.states[0].features.keys())

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the state series

        Returns
        -------
        Tuple[int, int]
            The shape of the state series
        """

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

        assert isinstance(state, State), f"'{state}' is not a State object"
        self.states.append(state)

    def pop(self) -> State:
        """Pop a state from the series

        Returns
        -------
        State
            The state that was popped from the series
        """

        assert len(self.states) > 0, "StateSeries is empty"
        return self.states.pop()

    def to_numpy(self, keys: List[str] = None) -> np.ndarray:
        """Convert to a 2D NumPy array: rows are states, columns are features

        Parameters
        ----------
        keys : List[str]
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

        if keys is None:
            keys = list(self.states[0].features.keys())

        data = [[state[k] for k in keys] for state in self.states]
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
        states: Union[List[str], str] = None,
        silent: bool = False,
        num_procs: int = 1,
        random_sample_size: int = None,
    ) -> StateSeries:
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
        StateSeries
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

        state_objs = []

        if num_procs == 1:
            for i, state in enumerate(states):
                state_objs.append(State.from_hdf5(file_name, state, features))
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
                    executor.submit(StateSeries.from_hdf5, file_name, features, chunk, silent=True): chunk
                    for chunk in chunks
                }

                completed = 0
                for job in as_completed(jobs):
                    result = job.result()
                    state_objs.extend(result)
                    if not silent:
                        for _ in result:
                            statusbar.update(completed)
                            completed += 1

        if not silent:
            statusbar.finalize()

        return cls(state_objs)

    def to_hdf5(self, file_name: str, group_name: str) -> None:
        """Write the state series to an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file to write to
        group_name : str
            The group name in the HDF5 file to write the state series to
        """
        with h5py.File(file_name, "w") as h5_file:
            group = h5_file.create_group(group_name)
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

    def to_csv(self, file_name: str, features: List[str]) -> None:
        """Write the state series to a CSV file

        Parameters
        ----------
        file_name : str
            The name of the CSV file to write to
        features : Dict[str, FeaturePerturbator]
            The dictionary of features to be written to the CSV file
        """

        df = self.to_dataframe(features)
        df.to_csv(file_name, index=False)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, features: List[str] = None) -> State:
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

        state_series = []
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
        return pd.DataFrame(series_np, columns=features.keys(), index=None)


class StateSeriesList:
    """A class for storing and accessing a list of StateSeries

    Parameters
    ----------
    state_series_list : List[StateSeries]
        The list of StateSeries which describe the series
    """

    def __init__(self, state_series_list: List[StateSeries]):
        self.state_series_list = state_series_list

    def __getitem__(self, index: Union[int, slice]) -> Union[StateSeries, StateSeriesList]:
        if isinstance(index, slice):
            return StateSeriesList(self.state_series_list[index])
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

        s = f"StateSeriesList of length {len(self)} with features: {self.features}\n"
        return s

    def __add__(self, other: StateSeriesList) -> StateSeriesList:
        assert isinstance(other, StateSeriesList), f"'{other}' is not a StateSeriesList object"
        return StateSeriesList(self.state_series_list + other.state_series_list)

    @property
    def features(self) -> List[str]:
        """Get the features of the state series list

        Returns
        -------
        List[str]
            The features of the state series list
        """

        if not self.state_series_list:
            return {}

        return self.state_series_list[0].features

    def append(self, state_series: StateSeries) -> None:
        """Append a state series to the list

        Parameters
        ----------
        state_series : StateSeries
            The state series to be appended to the list
        """

        assert isinstance(state_series, StateSeries), f"'{state_series}' is not a StateSeries object"
        self.state_series_list.append(state_series)

    @classmethod
    def from_hdf5(cls, file_name: str) -> StateSeriesList:
        """A factory method for building a collection of StateSeries from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file from which to read and build the state series from

        Returns
        -------
        StateSeriesList
            A list of state series read from the data in the HDF5 file
        """
        raise NotImplementedError("StateSeriesList.from_hdf5 is not implemented yet")

    def to_hdf5(self, file_name: str) -> None:
        """Write the state series list to an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file to write to
        """
        raise NotImplementedError("StateSeriesList.to_hdf5 is not implemented yet")

    @classmethod
    def from_csv(cls, file_name: str, features: List[str] = None) -> StateSeriesList:
        """A factory method for building a collection of StateSeries from a CSV file

        Parameters
        ----------
        file_name : str
            The name of the CSV file from which to read and build the state series from
        features : List[str]
            List of features to extract to the dataframe, default is all features of the state

        Returns
        -------
        StateSeriesList
            A list of state series read from the data in the CSV file
        """
        raise NotImplementedError("StateSeriesList.from_csv is not implemented yet")

    def to_csv(self, file_name: str, features: List[str] = None) -> None:
        """Write the state series list to a CSV file

        Parameters
        ----------
        file_name : str
            The name of the CSV file to write to
        features : List[str]
            List of features to extract to the dataframe, default is all features of the state
        """
        raise NotImplementedError("StateSeriesList.to_csv is not implemented yet")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> StateSeriesList:
        """Convert a Pandas DataFrame into a List of StateSeries

        "DataFrame index must be a MultiIndex with 'series_index' and 'state_index'"

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be converted.

        Returns
        -------
        StateSeriesList
            The list of state series
        """
        assert not df.empty, "DataFrame is empty"
        assert isinstance(
            df.index, pd.MultiIndex
        ), "DataFrame index must be a MultiIndex with 'series_index' and 'state_index'"

        state_series_dict = {}

        for (series_idx, state_idx), state_df in df.groupby(level=["series_index", "state_index"]):
            state = State.from_dataframe(state_df.reset_index(drop=True))

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
