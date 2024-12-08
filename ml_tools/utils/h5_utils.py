from typing import List, Union
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import h5py

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

    assert os.path.exists(file_name), f"File does not exist: {file_name}"

    with h5py.File(file_name, 'r') as h5_file:
        group_names = list(h5_file.keys())

        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            results = list(executor.map(partial(_check_group, prefix=prefix), group_names))
            groups = [group for group in results if group is not None]

    return groups

def _check_group(group: str, prefix: str) -> Union[str, None]:
    """ Private function for get_groups_with_prefix that is required to be a separate function by ProcessPoolExecutor
    """
    return group if group.startswith(prefix) else None
