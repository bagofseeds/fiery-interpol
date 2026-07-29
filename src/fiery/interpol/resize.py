"""
Resizing functions (equivalent to SciPy's `zoom` and PyTorch's
`interpolate`) based on `grid_pull`.
"""

__all__ = ['resize']

# dependencies
from typing import Optional, Sequence, Union

import torch

from . import backend, jitfields

# internals
from .api import grid_pull
from .utils import make_list, meshgrid_ij

Tensor = torch.Tensor
OrderLike = Union[int, str]


def resize(
    image: Tensor,
    factor: Optional[Union[float, Sequence[float]]] = None,
    shape: Optional[Sequence[int]] = None,
    anchor: Union[str, Sequence[str]] = 'c',
    interpolation: Union[OrderLike, Sequence[OrderLike]] = 1,
    prefilter: bool = True,
    **kwargs,
) -> Tensor:
    """Resize an image by a factor or to a specific shape.

    Notes
    -----
    - At least one of `factor` and `shape` must be specified.
    - If `anchor` is `'centers'` or `'edges'`, exactly one of `factor`
      or `shape` must be specified.
    - If `anchor` is `'first'` or `'last'`, `factor` must be provided,
      even if `shape` is specified.
    - Because of rounding, `resize(resize(x, f), 1/f)` is not guaranteed
      to have the same shape as `x`.

    The four anchor modes place the sampled points as follows (`e`, `c`,
    `f` and `l` mark the anchor points of each mode):

    ```
        edges          centers          first           last
    e - + - + - e   + - + - + - +   + - + - + - +   + - + - + - +
    | . | . | . |   | c | . | c |   | f | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | . | . | . |   | . | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | c | . | c |   | . | . | . |   | . | . | l |
    e _ + _ + _ e   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    ```

    Parameters
    ----------
    image : (batch, channel, *inshape) tensor
        Image to resize.
    factor : float or list[float], optional
        Resizing factor:

        - `> 1`: larger image <-> smaller voxels;
        - `< 1`: smaller image <-> larger voxels.
    shape : (ndim,) list[int], optional
        Output shape.
    anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
        - With `'centers'` or `'edges'`, the volume shape is multiplied
          by the zoom factor (and truncated if needed), and two anchor
          points are used to determine the voxel size.
        - With `'first'` or `'last'`, a single anchor point is used, so
          that the voxel size is exactly divided by the zoom factor.
          When `1/factor` is an integer, this is equivalent to
          subslicing the volume (`factor=1/2` gives `x[::2, ::2, ::2]`).
        - A list of anchors (one per dimension) can also be provided.
    interpolation : int or sequence[int], default=1
        Interpolation order.
    prefilter : bool, default=True
        Apply the spline pre-filter, so that the spline interpolates
        the input.
    **kwargs : dict
        Additional keyword arguments passed to `grid_pull`, such as
        `bound` or `extrapolate`.

    Returns
    -------
    resized : (batch, channel, *shape) tensor
        Resized image.

    Raises
    ------
    ValueError
        If neither `factor` nor `shape` is provided, or if `anchor` is
        not one of `'centers'`, `'edges'`, `'first'` or `'last'`.

    """
    if backend.jitfields and jitfields.available:
        return jitfields.resize(
            image, factor, shape, anchor, interpolation, prefilter, **kwargs
        )

    factor = make_list(factor) if factor else []
    shape = make_list(shape) if shape else []
    anchor = make_list(anchor)
    nb_dim = max(len(factor), len(shape), len(anchor)) or (image.dim() - 2)
    anchor = [a[0].lower() for a in make_list(anchor, nb_dim)]
    bck = dict(dtype=image.dtype, device=image.device)

    # compute output shape
    inshape = image.shape[-nb_dim:]
    if factor:
        factor = make_list(factor, nb_dim)
    elif not shape:
        raise ValueError('One of `factor` or `shape` must be provided')
    if shape:
        shape = make_list(shape, nb_dim)
    else:
        shape = [int(i * f) for i, f in zip(inshape, factor)]

    if not factor:
        factor = [o / i for o, i in zip(shape, inshape)]

    # compute transformation grid
    lin = []
    for anch, f, inshp, outshp in zip(anchor, factor, inshape, shape):
        if anch == 'c':  # centers
            lin.append(torch.linspace(0, inshp - 1, outshp, **bck))
        elif anch == 'e':  # edges
            scale = inshp / outshp
            shift = 0.5 * (scale - 1)
            lin.append(torch.arange(0.0, outshp, **bck) * scale + shift)
        elif anch == 'f':  # first voxel
            # scale = 1/f
            # shift = 0
            lin.append(torch.arange(0.0, outshp, **bck) / f)
        elif anch == 'l':  # last voxel
            # scale = 1/f
            shift = (inshp - 1) - (outshp - 1) / f
            lin.append(torch.arange(0.0, outshp, **bck) / f + shift)
        else:
            raise ValueError('Unknown anchor {}'.format(anch))

    # interpolate
    kwargs.setdefault('bound', 'nearest')
    kwargs.setdefault('extrapolate', True)
    kwargs.setdefault('interpolation', interpolation)
    kwargs.setdefault('prefilter', prefilter)
    grid = torch.stack(meshgrid_ij(*lin), dim=-1)
    resized = grid_pull(image, grid, **kwargs)

    return resized
