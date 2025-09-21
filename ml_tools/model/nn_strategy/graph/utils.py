from __future__ import annotations
from typing import Tuple, Union, List, Literal
import numpy as np

# Pylint appears to not be handling the tensorflow imports correctly
# pylint: disable=import-error, no-name-in-module, no-member
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

ShapeType = Union[
    Tuple[int],           # 1D shape: (H,)
    Tuple[int, int],      # 2D shape: (H, W)
    Tuple[int, int, int]  # 3D shape: (H, W, D)
]


def _extend_shape(shape: ShapeType) -> Tuple[int, int, int]:
    """Extend a 1D/2D shape to 3D by appending ones.

    Parameters
    ----------
    shape : ShapeType
        Spatial shape as ``(H,)``, ``(H, W)``, or ``(H, W, D)``.

    Returns
    -------
    tuple[int, int, int]
        Extended shape ``(H, W, D)``.
    """
    if len(shape) == 1:
        return (shape[0], 1, 1)
    if len(shape) == 2:
        return (shape[0], shape[1], 1)
    if len(shape) == 3:
        return shape
    raise ValueError(f"Invalid shape {shape}. Expected a 1D, 2D, or 3D tuple.")


def _prod(t: Tuple[int, int, int]) -> int:
    """Product of a 3D shape tuple ``(H, W, D)``.

    Parameters
    ----------
    t : tuple[int, int, int]
        The shape triple.

    Returns
    -------
    int
        ``H * W * D``.
    """
    return int(t[0]) * int(t[1]) * int(t[2])


def _neighbors_offsets(dim: int,
                       connectivity: Literal['1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26']) -> List[Tuple[int, int, int]]:
    """Return neighbor offsets for a regular grid.

    Parameters
    ----------
    dim : int
        Dimensionality of the grid (1, 2, or 3).
    connectivity : {'1d-2','2d-4','2d-8','3d-6','3d-18','3d-26'}
        Neighborhood definition.

    Returns
    -------
    list[tuple[int, int, int]]
        Offsets as ``(dx, dy, dz)``.
    """
    if dim == 1:
        return [(1, 0, 0), (-1, 0, 0)]
    if dim == 2:
        offs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                manhattan = abs(dx) + abs(dy)
                if connectivity == '2d-4' and manhattan == 1:
                    offs.append((dx, dy, 0))
                if connectivity == '2d-8' and 1 <= manhattan <= 2:
                    offs.append((dx, dy, 0))
        return offs
    if dim == 3:
        offs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    manhattan = abs(dx) + abs(dy) + abs(dz)
                    if connectivity == '3d-6' and manhattan == 1:
                        offs.append((dx, dy, dz))
                    elif connectivity == '3d-18' and 1 <= manhattan <= 2:
                        offs.append((dx, dy, dz))
                    elif connectivity == '3d-26':
                        offs.append((dx, dy, dz))
        return offs
    raise ValueError(f"Unsupported dimension {dim}")


def _build_adjacency(shape: Tuple[int, int, int],
                     connectivity: Literal['1d-2', '2d-4', '2d-8', '3d-6', '3d-18', '3d-26'],
                     self_loops: bool,
                     normalize: bool,
                     distance_weighted: bool = False) -> np.ndarray:
    """Build a dense adjacency matrix for a grid graph.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Extended spatial shape ``(H, W, D)``.
    connectivity : {'1d-2','2d-4','2d-8','3d-6','3d-18','3d-26'}
        Neighborhood class.
    self_loops : bool
        Whether to include identity edges.
    normalize : bool
        Apply symmetric degree normalization.
    distance_weighted : bool, optional
        Use inverse Manhattan weights within the chosen neighborhood, by default False.

    Returns
    -------
    numpy.ndarray
        Adjacency matrix of shape ``(N, N)`` with ``N = H * W * D``.
    """
    H, W, D = shape
    dim = 1 if (W == 1 and D == 1) else 2 if (D == 1) else 3
    N = _prod(shape)
    A = np.zeros((N, N), dtype=np.float32)

    def idx(x: int, y: int, z: int) -> int:
        return z * (H * W) + y * H + x

    offsets = _neighbors_offsets(dim, connectivity)

    for z in range(D):
        for y in range(W):
            for x in range(H):
                i = idx(x, y, z)
                if self_loops:
                    A[i, i] = 1.0
                for dx, dy, dz in offsets:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                        j = idx(nx, ny, nz)

                        if distance_weighted:
                            manhattan = abs(dx) + abs(dy) + abs(dz)
                            w = float(1.0 / manhattan) if manhattan > 0 else 1.0
                        else:
                            w = 1.0
                        A[i, j] = w
                        A[j, i] = w

    if normalize:
        deg = np.sum(A, axis=1)
        inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        A = D_inv_sqrt @ A @ D_inv_sqrt

    return A


@register_keras_serializable()
def merge_batch_node(t: tf.Tensor) -> tf.Tensor:
    """Merge batch and node axes of a 4D tensor.

    Parameters
    ----------
    t : tf.Tensor
        Tensor of shape ``(B, T, N, F)``.

    Returns
    -------
    tf.Tensor
        Reshaped tensor of shape ``(B*N, T, F)``.
    """
    b = tf.shape(t)[0]
    n = tf.shape(t)[2]
    tm = tf.shape(t)[1]
    f = tf.shape(t)[3]
    return tf.reshape(t, (b * n, tm, f))


@register_keras_serializable()
def unmerge_batch_node(inputs):
    """Inverse of ``merge_batch_node`` given a reference tensor.

    Parameters
    ----------
    inputs : tuple[tf.Tensor, tf.Tensor] | list
        ``[t, ref]`` where ``t`` is ``(B*N, T, F)`` and ``ref`` is ``(B, T, N, _)``.

    Returns
    -------
    tf.Tensor
        Reshaped tensor of shape ``(B, T, N, F)``.
    """
    if isinstance(inputs, (list, tuple)):
        t, ref = inputs
    else:
        raise ValueError("unmerge_batch_node expects [t, ref]")
    b = tf.shape(ref)[0]
    n = tf.shape(ref)[2]
    tm = tf.shape(ref)[1]
    u = tf.shape(t)[2]
    return tf.reshape(t, (b, tm, n, u))


@register_keras_serializable()
def merge_batch_time(t: tf.Tensor) -> tf.Tensor:
    """Merge batch and time axes of a 4D tensor.

    Parameters
    ----------
    t : tf.Tensor
        Tensor of shape ``(B, T, N, F)``.

    Returns
    -------
    tf.Tensor
        Reshaped tensor of shape ``(B*T, N, F)``.
    """
    b = tf.shape(t)[0]
    tm = tf.shape(t)[1]
    n = tf.shape(t)[2]
    f = tf.shape(t)[3]
    return tf.reshape(t, (b * tm, n, f))


@register_keras_serializable()
def unmerge_batch_time(inputs):
    """Inverse of ``merge_batch_time`` given a reference tensor.

    Parameters
    ----------
    inputs : tuple[tf.Tensor, tf.Tensor] | list
        ``[t, ref]`` where ``t`` is ``(B*T, N, F)`` and ``ref`` is ``(B, T, N, _)``.

    Returns
    -------
    tf.Tensor
        Reshaped tensor of shape ``(B, T, N, F)``.
    """
    if isinstance(inputs, (list, tuple)):
        t, ref = inputs
    else:
        raise ValueError("unmerge_batch_time expects [t, ref]")
    b = tf.shape(ref)[0]
    tm = tf.shape(ref)[1]
    n = tf.shape(ref)[2]
    u = tf.shape(t)[2]
    return tf.reshape(t, (b, tm, n, u))
