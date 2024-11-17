from __future__ import annotations
from typing import List
import os
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
        def check_group(group):
            return group if group.startswith(prefix) else None

        with ThreadPoolExecutor(max_workers=num_procs) as executor:
            results = list(executor.map(check_group, h5_file))
            groups = [group for group in results if group is not None]

    return groups
