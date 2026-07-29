# fiery-interpol
High-order spline interpolation in PyTorch

`fiery-interpol` is a [`fiery`](https://bagofseeds.github.io/fiery/) match: it
installs on its own and imports as `fiery.interpol`.

## Description

This package contains a pure Python implementation of **high-order spline
interpolation** for ND tensors (including 2D and 3D images). It uses the
just-in-time capabilities of TorchScript and explicitly implements the
forward and backward passes of all functions, which makes it **fast** and
**memory-efficient**.

All the functions available in this (small) package were originally
implemented in [NITorch](https://github.com/balbasty/nitorch), a larger
PyTorch-based package dedicated to NeuroImaging and Medical Image Computing.

## Installation

### Dependencies

- `python >= 3.6`
- `torch >= 1.3`

### Pip

```shell
pip install fiery-interpol
```

## Usage

```python
from fiery import interpol
```

**See our [example notebooks](https://bagofseeds.github.io/fiery-interpol/examples/)**

## Quick doc

The `interpolation` and `bound` arguments below are shared by most
functions in the package:

```text
Notes
-----

`interpolation` can be an int, a string or an InterpolationType.
Possible values are:
    - 0 or 'nearest'
    - 1 or 'linear'
    - 2 or 'quadratic'
    - 3 or 'cubic'
    - 4 or 'fourth'
    - 5 or 'fifth'
    - etc.
A list of values can be provided, in the order [W, H, D],
to specify dimension-specific interpolation orders.

`bound` can be an int, a string or a BoundType.
Possible values are:
    - 'replicate'  or 'nearest'     :  a  a  a  |  a  b  c  d  |  d  d  d
    - 'dct1'       or 'mirror'      :  d  c  b  |  a  b  c  d  |  c  b  a
    - 'dct2'       or 'reflect'     :  c  b  a  |  a  b  c  d  |  d  c  b
    - 'dst1'       or 'antimirror'  : -b -a  0  |  a  b  c  d  |  0 -d -c
    - 'dst2'       or 'antireflect' : -c -b -a  |  a  b  c  d  | -d -c -b
    - 'dft'        or 'wrap'        :  b  c  d  |  a  b  c  d  |  a  b  c
    - 'zero'       or 'zeros'       :  0  0  0  |  a  b  c  d  |  0  0  0
A list of values can be provided, in the order [W, H, D],
to specify dimension-specific boundary conditions.
Note that
- `dft` corresponds to circular padding
- `dct2` corresponds to Neumann boundary conditions (symmetric)
- `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
See https://en.wikipedia.org/wiki/Discrete_cosine_transform
    https://en.wikipedia.org/wiki/Discrete_sine_transform
```

```python
interpol.grid_pull(
    input,
    grid,
    interpolation='linear',
    bound='zero',
    extrapolate=False,
    prefilter=False,
)
"""
Sample an image with respect to a deformation field.

If the input dtype is not a floating point type, the input image is
assumed to contain labels. In that case, the unique labels are
extracted and resampled individually, which turns them into soft
labels. The label map is then reconstructed from the individual soft
labels by assigning, in each output voxel, the label with the largest
soft value.

Parameters
----------
input : (..., [channel], *inshape) tensor
    Input image.
grid : (..., *outshape, dim) tensor
    Transformation field.
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : BoundType or sequence[BoundType], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=False
    Extrapolate out-of-bound data.
prefilter : bool, default=False
    Apply the spline pre-filter, so that the spline interpolates
    the input.

Returns
-------
output : (..., [channel], *outshape) tensor
    Deformed image.
"""
```

```python
interpol.grid_push(
    input,
    grid,
    shape=None,
    interpolation='linear',
    bound='zero',
    extrapolate=False,
    prefilter=False,
)
"""
Splat an image with respect to a deformation field (pull adjoint).

Parameters
----------
input : (..., [channel], *inshape) tensor
    Input image.
grid : (..., *inshape, dim) tensor
    Transformation field.
shape : sequence[int], default=inshape
    Output shape.
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : BoundType or sequence[BoundType], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=False
    Extrapolate out-of-bound data.
prefilter : bool, default=False
    Apply the spline pre-filter to the splatted image.

Returns
-------
output : (..., [channel], *shape) tensor
    Splatted image.
"""
```

```python
interpol.grid_grad(
    input,
    grid,
    interpolation='linear',
    bound='zero',
    extrapolate=False,
    prefilter=False,
)
"""
Sample spatial gradients of an image along a deformation field.

The returned gradients are taken with respect to the coordinates of
the *input* lattice: this function returns (∇f)∘φ, not ∇(f∘φ). To
obtain gradients with respect to the output lattice, multiply the
sampled gradients by the Jacobian field of the transformation.

Parameters
----------
input : (..., [channel], *inshape) tensor
    Input image.
grid : (..., *outshape, dim) tensor
    Transformation field.
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : BoundType or sequence[BoundType], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=False
    Extrapolate out-of-bound data.
prefilter : bool, default=False
    Apply the spline pre-filter, so that the spline interpolates
    the input.

Returns
-------
output : (..., [channel], *outshape, dim) tensor
    Sampled gradients.
"""
```

```python
interpol.spline_coeff_nd(
    input,
    interpolation='linear',
    bound='dct2',
    dim=None,
    inplace=False,
)
"""
Compute the interpolating spline coefficients, for a given spline order
and boundary conditions, along the last `dim` dimensions.

References
----------
1. M. Unser, A. Aldroubi and M. Eden.
   "B-Spline Signal Processing: Part I-Theory,"
   IEEE Transactions on Signal Processing 41(2):821-832 (1993).
2. M. Unser, A. Aldroubi and M. Eden.
   "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
   IEEE Transactions on Signal Processing 41(2):834-848 (1993).
3. M. Unser.
   "Splines: A Perfect Fit for Signal and Image Processing,"
   IEEE Signal Processing Magazine 16(6):22-38 (1999).

Parameters
----------
input : (..., *spatial) tensor
    Input image.
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : BoundType or sequence[BoundType], default='dct2'
    Boundary conditions.
dim : int, default=input.dim()
    Number of trailing dimensions to process.
inplace : bool, default=False
    Process the volume in place.

Returns
-------
output : (..., *spatial) tensor
    Coefficient image.
"""
```

```python
interpol.resize(
    image,
    factor=None,
    shape=None,
    anchor='c',
    interpolation=1,
    prefilter=True,
    **kwargs,
)
"""Resize an image by a factor or to a specific shape.

Notes
-----
- At least one of `factor` and `shape` must be specified.
- If `anchor` is 'centers' or 'edges', exactly one of `factor` or
  `shape` must be specified.
- If `anchor` is 'first' or 'last', `factor` must be provided, even
  if `shape` is specified.
- Because of rounding, `resize(resize(x, f), 1/f)` is not guaranteed
  to have the same shape as `x`.

        edges          centers          first           last
    e - + - + - e   + - + - + - +   + - + - + - +   + - + - + - +
    | . | . | . |   | c | . | c |   | f | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | . | . | . |   | . | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | c | . | c |   | . | . | . |   | . | . | l |
    e _ + _ + _ e   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +

Parameters
----------
image : (batch, channel, *inshape) tensor
    Image to resize.
factor : float or list[float], optional
    Resizing factor
    * > 1 : larger image <-> smaller voxels
    * < 1 : smaller image <-> larger voxels
shape : (ndim,) list[int], optional
    Output shape.
anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
    * With 'centers' or 'edges', the volume shape is multiplied by
      the zoom factor (and truncated if needed), and two anchor
      points are used to determine the voxel size.
    * With 'first' or 'last', a single anchor point is used, so that
      the voxel size is exactly divided by the zoom factor. When
      1/factor is an integer, this is equivalent to subslicing the
      volume (factor=1/2 gives x[::2, ::2, ::2]).
    * A list of anchors (one per dimension) can also be provided.
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
"""
```

## License

fiery-interpol is released under the MIT license.
