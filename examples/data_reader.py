from typing import List

import numpy as np

from ml_tools import State, StateSeries
from ml_tools.utils.h5_utils import get_groups_with_prefix

class DataReader():
    """ Class for reading data from HDF5 files
    """

    def read_data(file_name:          str,
                  num_procs:          int = 1,
                  random_sample_size: int = None) -> List[StateSeries]:
        """ Method for reading all relevent data from the HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the HDF5 file from which to read and build the states from
        num_procs : int
            The number of parallel processors to use when reading data from the HDF5
        random_sample_size : int
            Number of random state samples to draw from the list of specified states.
            If this argument is not provided, all states of the list will be considered.

        Returns
        -------
        List[StateSeries]
            The relevent data for all states.  This is returned as a list of state series
            because that is what the prediction strategies take.  These are effectively a
            list of series with length 1 (i.e., list of single state points)
        """

        features_to_read = [    "2d_assembly_exposure", "average_enrichment",
                                 "boron_concentration", "measured_fixed_detector","cycle_exposure"]

        state_groups = get_groups_with_prefix(file_name = file_name, prefix = "set_", num_procs = num_procs)
        states       = State.read_states_from_hdf5(file_name          = file_name,
                                                   features           = features_to_read,
                                                   states             = state_groups,
                                                   num_procs          = num_procs,
                                                   random_sample_size = random_sample_size)

        # Convert state data to correct format
        for state in states:
            state.features["average_exposure"]       = np.nan_to_num(state["2d_assembly_exposure"], nan=0.)
            state.features["assembly_enrichment"]    = np.nan_to_num(state["average_enrichment"], nan=0.)

        # Convert state data to list of state series
        series = [[state] for state in states]

        return series
