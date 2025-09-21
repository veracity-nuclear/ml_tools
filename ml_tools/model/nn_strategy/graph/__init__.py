from .graph import Graph
from .sage import SAGE, GraphSAGEConv

def load_graph_from_h5(group) -> Graph:
    """Load a Graph variant from an HDF5 group saved via ``Graph.save``.

    Parameters
    ----------
    group : h5py.Group
        Group containing the variant and its configuration.

    Returns
    -------
    Graph
        The reconstructed Graph variant instance.
    """
    graph_type = group['graph_type'][()].decode('utf-8')
    if graph_type == 'SAGE':
        return SAGE.from_h5(group)
    raise ValueError(f"Unknown graph variant '{graph_type}' in H5 group")

__all__ = [
    'Graph',
    'SAGE',
    'GraphSAGEConv',
    'load_graph_from_h5',
]
