from typing import List
import os
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from ml_tools.utils.status_bar import StatusBar

def keys_with_prefix(keys: List[str], prefix: str) -> List[str]:
    """ Helper function for getting the keys within a list of key that start with the given prefix

    Parameters
    ----------
    keys : List[str]
        The list of keys to parse for keys starting with the given prefix
    prefix : str
        The prefix which to match

    Returns
    -------
    List[str]
        The list of keys that start with the given prefix
    """

    return [key for key in keys if key.startswith(prefix)]



def get_groups_with_prefix(file_name: str, prefix: str, num_procs: int = 1) -> List[str]:
    """ Helper function for getting the groups of an HDF5 file belonging to the set with a leading prefix

    Parameters
    ----------
    file_name : str
        The name of the file which to read the groups from
    prefix : str
        The prefix of the set which the groups must match
    num_procs : int
        The number of parallel processors to use when reading data from the HDF5

    Returns
    -------
    List[str]
        The list of groups that belong to the set
    """

    assert(os.path.exists(file_name))

    with h5py.File(file_name, 'r') as h5_file:
        keys = list(h5_file.keys())
        key_chunks = np.array_split(keys, num_procs)
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
                keys = list(executor.map(keys_with_prefix, key_chunks, prefix))
        keys = [key for sublist in keys for key in sublist]

    return keys
