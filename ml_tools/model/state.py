from __future__ import annotations
from typing import List, Dict, Union, Optional
import os
import random
import re
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
import pandas as pd
from ml_tools.utils.status_bar import StatusBar
from ml_tools.model.feature_perturbator import FeaturePerturbator

class State():
    """ A class for storing and accessing generic state data

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        The features which describe the state

    Attributes
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
        """ Method for retrieving the feature data from a state

        Parameters
        ----------
        name : str
            The name of the feature to be retrieved

        Returns
        -------
        np.ndarray
            The feature data that was retrieved
        """

        assert feature_name in self.features, \
            f"'{feature_name}' not found in state features. Available features: {list(self.features.keys())}"
        return self._features[feature_name]

    def __setitem__(self, feature_name: str, data_array: Union[np.ndarray, List[float]]) -> None:
        """ Method for setting the feature data of a state

        Parameters
        ----------
        name : str
            The name of the feature to be set
        data_array : Union[np.ndarray, List[float]]
            The array to assign to the state
        """

        assert feature_name in self.features, \
            f"'{feature_name}' not found in state features. Available features: {list(self.features.keys())}"
        data_array = data_array if isinstance(data_array, np.ndarray) else np.asarray(data_array)
        self._features[feature_name] = data_array


    def to_dataframe(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """ Convert the State into a Pandas DataFrame.

        Parameters
        ----------
        features : Optional[List[str]]
            List of features to extract to the dataframe, default is all features of the state

        Returns
        -------
        pd.DataFrame
            A DataFrame where each feature is a column, and each row corresponds to an element in the feature arrays.
        """

        features = self.features if features is None else {k: self.features[k] for k in features}

        flat_data = {}
        for feature_name, values in features.items():
            values = np.asarray(values)

            if len(values) == 1:
                flat_data[feature_name] = values
            else:
                for i, v in enumerate(values.flat):
                    flat_data[f"{feature_name}_{i}"] = v

        return pd.DataFrame([flat_data])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> State:
        """ Convert a Pandas DataFrame into a State object.
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
                features[col] = df[col].values[0]

        for key, feature in features.items():
            if isinstance(feature, list):
                features[key] = np.array(feature).flatten()

        return cls(features)


    @staticmethod
    def read_state_from_hdf5(file_name: str, state: str, features: List[str]) -> State:
        """ A factory method for extracting state feature data from an HDF5 file
            file_name : str
                The name of the HDF5 file from which to read and build the state from
            state : str
                The group in the HDF5 file which holds the feature data of the state
            features : List[str]
                The list of features expected to be read in for each state
        """

        assert os.path.exists(file_name), f"File does not exist: {file_name}"
        assert len(features) > 0, f"'len(features) = {len(features)}'"

        with h5py.File(file_name, 'r') as h5_file:
            assert state in h5_file.keys(), f"'{state}' not found in {file_name}"
            assert all(feature in h5_file[state].keys() for feature in features)

            state_data = {}
            for feature in features:
                data                = h5_file[state][feature][()]
                feature             = os.path.basename(feature)
                state_data[feature] = data
                if np.isscalar(state_data[feature]):
                    state_data[feature] = np.array([state_data[feature]])
                else:
                    state_data[feature] = state_data[feature].flatten()


        return State(state_data)


    @staticmethod
    def read_states_from_hdf5(file_name:          str,
                              features:           List[str],
                              states:             Union[List[str], str] = None,
                              silent:             bool = False,
                              num_procs:          int = 1,
                              random_sample_size: int = None
    ) -> List[State]:
        """ A factory method for building a collection of States from an HDF5 file

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
        assert len(states) > 0, f"'len(states) = {len(states)}'"
        assert len(features) > 0, f"'len(features) = {len(features)}'"
        assert num_procs > 0, f"'num_procs = {num_procs}'"

        if states is None:
            with h5py.File(file_name, 'r') as h5_file:
                states = h5_file.keys()
        elif isinstance(states, str):
            states = [states]

        if random_sample_size:
            assert random_sample_size > 0, f"'random_sample_size = {random_sample_size}'"
            assert random_sample_size < len(states), \
                f"'random_sample_size = {random_sample_size}, len(states) = {len(states)}'"
            states = random.sample(states, random_sample_size)

        if not silent:
            statusbar = StatusBar(len(states))
        state_data = []
        i = 0

        if num_procs == 1:
            for state in states:
                state_data.append(State.read_state_from_hdf5(file_name, state, features))
                if not silent:
                    statusbar.update(i)
                    i+=1

        else:
            def chunkify(states: List[str], chunk_size: int):
                for i in range(0, len(states), chunk_size):
                    yield states[i:i + chunk_size]

            chunk_size = max(1, len(states) // num_procs)
            chunks     = list(chunkify(states, chunk_size))

            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                jobs = {executor.submit(State.read_states_from_hdf5, file_name, features, chunk, silent=True): \
                    chunk for chunk in chunks}

                for i, job in enumerate(jobs):
                    for state in job.result():
                        state_data.append(state)
                        if not silent:
                            statusbar.update(i)
                            i+=1

        if not silent:
            statusbar.finalize()

        return state_data


    @staticmethod
    def perturb_state(perturbators: Dict[str, FeaturePerturbator],
                      state:        State) -> State:
        """ A method for perturbing the features of a given state

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
    def perturb_states(perturbators: Dict[str, FeaturePerturbator],
                       states:             Union[List[State], State],
                       silent:             bool = False,
                       num_procs:          int = 1) -> List[State]:
        """ A method for perturbing the features of a given collection of states

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
            statusbar = StatusBar(len(states))
        state_data = []
        i = 0

        if num_procs == 1:
            for state in states:
                state_data.append(State.perturb_state(perturbators, state))
                if not silent:
                    statusbar.update(i)
                    i+=1

        else:
            def chunkify(states: List[State], chunk_size: int):
                for i in range(0, len(states), chunk_size):
                    yield states[i:i + chunk_size]

            chunk_size = max(1, len(states) // num_procs)
            chunks     = list(chunkify(states, chunk_size))

            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                jobs = {executor.submit(State.perturb_state(perturbators, state)): chunk for chunk in chunks}

                for i, job in enumerate(jobs):
                    for state in job.result():
                        state_data.append(state)
                        if not silent:
                            statusbar.update(i)
                            i+=1

        if not silent:
            statusbar.finalize()

        return state_data

# Defining a series of States as an order list
StateSeries = List[State]


def series_to_pandas(state_series: List[StateSeries], features: List[str] = None,) -> pd.DataFrame:
    """ Convert a List of StateSeries to a Pandas Dataframe

    Parameters
    ----------
    state_series : List[StateSeries]
        The list of state series to be converted
    features : List[str]
        List of features to extract to the dataframe, default is all features of the state

    Returns
    -------
    pd.DataFrame
        The created Pandas Dataframe
    """

    series_dfs = []

    for series_idx, series in enumerate(state_series):
        if not series:
            continue

        state_dfs = [state.to_dataframe(features) for state in series]
        assert all(isinstance(df, pd.DataFrame) for df in state_dfs), \
            "One or more states returned an invalid DataFrame"

        df       = pd.concat(state_dfs, ignore_index=True)
        df.index = pd.MultiIndex.from_product([[series_idx], range(len(df))], names=["series_index", "state_index"])
        series_dfs.append(df)

    return pd.concat(series_dfs)


def pandas_to_series(df: pd.DataFrame) -> List[StateSeries]:
    """ Convert a Pandas DataFrame into a List of StateSeries

    "DataFrame index must be a MultiIndex with 'series_index' and 'state_index'"

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be converted.

    Returns
    -------
    List[StateSeries]
        The list of state series
    """
    assert not df.empty, "DataFrame is empty"
    assert isinstance(df.index, pd.MultiIndex), \
        "DataFrame index must be a MultiIndex with 'series_index' and 'state_index'"

    state_series_dict = {}

    for (series_idx, state_idx), state_df in df.groupby(level=["series_index", "state_index"]):
        state = State.from_dataframe(state_df.reset_index(drop=True))

        if series_idx not in state_series_dict:
            state_series_dict[series_idx] = []
        state_series_dict[series_idx].append(state)

    max_series_idx = max(state_series_dict.keys())
    return [state_series_dict.get(i) for i in range(max_series_idx + 1)]
