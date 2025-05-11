from typing import List, Union
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py

from ml_tools.utils.status_bar import StatusBar

def get_groups_with_prefix(file_name: str, prefix: str, num_procs: int = 1, silent: bool = False) -> List[str]:
    """ Helper function for getting the groups of an HDF5 file belonging to the set with a leading prefix

    Parameters
    ----------
    file_name : str
        The name of the file which to read the groups from
    prefix : str
        The prefix of the set which the groups must match
    num_procs : int
        The number of parallel processors to use when reading data from the HDF5
    silent : bool
        A flag indicating whether or not to display the progress bar to the screen

    Returns
    -------
    List[str]
        The list of groups that belong to the set
    """

    assert os.path.exists(file_name), f"File does not exist: {file_name}"

    with h5py.File(file_name, 'r') as h5_file:
        group_names = list(h5_file.keys())

    groups = []


    if not silent:
        print(f"Retrieving groups with prefix: '{prefix}' from: '{file_name}'")
        statusbar = StatusBar(len(group_names))

    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        jobs = {executor.submit(_check_group, group, prefix=prefix): group for group in group_names}

        completed = 0
        for job in as_completed(jobs):
            result = job.result()
            if result is not None:
                groups.append(result)
            if not silent:
                statusbar.update(completed)
                completed += 1

    if not silent:
        statusbar.finalize()

    return groups

def _check_group(group: str, prefix: str) -> Union[str, None]:
    """ Private function for get_groups_with_prefix that is required to be a separate function by ProcessPoolExecutor
    """
    return group if group.startswith(prefix) else None
