"""
Restriction functions (the adjoint of `resize`) based on `grid_push`.
"""

__all__ = ['restrict']

# dependencies
from typing import Optional, Sequence, Union

import torch

from . import backend, jitfields

# internals
from .api import grid_push
from .utils import make_list, meshgrid_ij

Tensor = torch.Tensor
OrderLike = Union[int, str]


def restrict(
    image: Tensor,
    factor: Optional[Union[float, Sequence[float]]] = None,
    shape: Optional[Sequence[int]] = None,
    anchor: Union[str, Sequence[str]] = 'c',
    interpolation: Union[OrderLike, Sequence[OrderLike]] = 1,
    reduce_sum: bool = False,
    **kwargs,
) -> Tensor:
    """Restrict an image by a factor or to a specific shape.

    Restriction is the adjoint of resizing: instead of sampling the
    input at the output coordinates, each input voxel is splatted into
    the output lattice.

    Notes
    -----
    - At least one of `factor` and `shape` must be specified.
    - If `anchor` is `'centers'` or `'edges'`, exactly one of `factor`
      or `shape` must be specified.
    - If `anchor` is `'first'` or `'last'`, `factor` must be provided,
      even if `shape` is specified.
    - Because of rounding, `restrict(resize(x, f), f)` is not guaranteed
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
        Image to restrict.
    factor : float or list[float], optional
        Restriction factor:

        - `> 1`: smaller image <-> larger voxels;
        - `< 1`: larger image <-> smaller voxels.
    shape : (ndim,) list[int], optional
        Output shape.
    anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
        - With `'centers'` or `'edges'`, the volume shape is divided by
          the restriction factor (and truncated if needed), and two
          anchor points are used to determine the voxel size.
        - With `'first'` or `'last'`, a single anchor point is used, so
          that the voxel size is exactly multiplied by the restriction
          factor.
        - A list of anchors (one per dimension) can also be provided.
    interpolation : int or sequence[int], default=1
        Interpolation order.
    reduce_sum : bool, default=False
        Return the accumulated values without normalizing them by the
        change of voxel size.
    **kwargs : dict
        Additional keyword arguments passed to `grid_push`, such as
        `bound` or `extrapolate`.

    Returns
    -------
    restricted : (batch, channel, *shape) tensor
        Restricted image.

    Raises
    ------
    ValueError
        If neither `factor` nor `shape` is provided, or if `anchor` is
        not one of `'centers'`, `'edges'`, `'first'` or `'last'`.

    """
    if backend.jitfields and jitfields.available:
        return jitfields.restrict(
            image, factor, shape, anchor, interpolation, reduce_sum, **kwargs
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
        shape = [int(i / f) for i, f in zip(inshape, factor)]

    if not factor:
        factor = [i / o for o, i in zip(shape, inshape)]

    # compute transformation grid
    lin = []
    fullscale = 1
    for anch, f, inshp, outshp in zip(anchor, factor, inshape, shape):
        if anch == 'c':  # centers
            lin.append(torch.linspace(0, outshp - 1, inshp, **bck))
            fullscale *= (inshp - 1) / (outshp - 1)
        elif anch == 'e':  # edges
            scale = outshp / inshp
            shift = 0.5 * (scale - 1)
            fullscale *= scale
            lin.append(torch.arange(0.0, inshp, **bck) * scale + shift)
        elif anch == 'f':  # first voxel
            # scale = 1/f
            # shift = 0
            fullscale *= 1 / f
            lin.append(torch.arange(0.0, inshp, **bck) / f)
        elif anch == 'l':  # last voxel
            # scale = 1/f
            shift = (outshp - 1) - (inshp - 1) / f
            fullscale *= 1 / f
            lin.append(torch.arange(0.0, inshp, **bck) / f + shift)
        else:
            raise ValueError('Unknown anchor {}'.format(anch))

    # scatter
    kwargs.setdefault('bound', 'nearest')
    kwargs.setdefault('extrapolate', True)
    kwargs.setdefault('interpolation', interpolation)
    kwargs.setdefault('prefilter', False)
    grid = torch.stack(meshgrid_ij(*lin), dim=-1)
    resized = grid_push(image, grid, shape, **kwargs)
    if not reduce_sum:
        resized /= fullscale

    return resized
