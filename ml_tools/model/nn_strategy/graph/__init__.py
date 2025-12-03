from .graph import Graph
from .sage import SAGE, GraphSAGEConv
from .gat import GAT, GraphAttentionConv

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
    if graph_type == 'GAT':
        return GAT.from_h5(group)
    raise ValueError(f"Unknown graph variant '{graph_type}' in H5 group")

def build_graph_from_dict(data: dict) -> Graph:
    """ Build a Graph variant from a parameter dict.

    Parameters
    ----------
    data : dict
        Dictionary containing the graph configuration.

    Returns
    -------
    Graph
        The constructed Graph variant instance.
    """
    if 'variant' not in data:
        raise KeyError("Graph configuration must include 'variant' key")
    variant = data['variant']
    if variant == 'SAGE':
        return SAGE.from_dict(data)
    if variant == 'GAT':
        return GAT.from_dict(data)
    raise ValueError(f"Unknown graph variant '{variant}' in configuration")

__all__ = [
    'Graph',
    'SAGE',
    'GraphSAGEConv',
    'GAT',
    'GraphAttentionConv',
    'load_graph_from_h5',
    'build_graph_from_dict'
]
