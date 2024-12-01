from __future__ import annotations
from typing import List, Dict, Union
import os
import h5py
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from ml_tools.utils.status_bar import StatusBar

class State(object):
    """ A class for storing and accessing generic state data

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


    def feature(self, name: "str") -> np.ndarray:
        """ Method for retrieving the feature data from a state

        This method might not be entirely necessary, but I think it's a cleaner
        way of accessing the underlying data than using the "features" property directly.

        Parameters
        ----------
        name :str
            The name of the feature to be retrieved

        Returns
        -------
        np.ndarray
            The feature data that was retrieved
        """

        assert name in self.features, f"'{name}' not found in state features. Available features: {list(self.features.keys())}"
        return self._features[name]


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
                states = h5_groups
        elif isinstance(states, str):
            states = [states]

        if random_sample_size:
            assert random_sample_size > 0, f"'random_sample_size = {random_sample_size}'"
            assert random_sample_size < len(states), f"'random_sample_size = {random_sample_size}, len(states) = {len(states)}'"
            states = random.sample(states, random_sample_size)

        if not silent: statusbar = StatusBar(len(states))
        state_data = []
        i = 0

        if num_procs == 1:
            for state in states:
                state_data.append(State.read_state_from_hdf5(file_name, state, features))
                if not silent: statusbar.update(i); i+=1

        else:
            def chunkify(states: List[str], chunk_size: int):
                for i in range(0, len(states), chunk_size):
                    yield states[i:i + chunk_size]

            chunk_size = max(1, len(states) // num_procs)
            chunks     = list(chunkify(states, chunk_size))

            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                jobs = {executor.submit(State.read_states_from_hdf5, file_name, features, chunk, silent=True): chunk for chunk in chunks}

                for job in as_completed(jobs):
                    for state in job.result():
                        state_data.append(state)
                        if not silent: statusbar.update(i); i+=1

        if not silent: statusbar.finalize()

        return state_data

# Defining a series of States as an order list
StateSeries = List[State]
