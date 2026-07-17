"""High-order spline interpolation in PyTorch."""

from . import backend  # noqa: F401
from .api import *  # noqa: F401, F403
from .resize import *  # noqa: F401, F403
from .restrict import *  # noqa: F401, F403

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0+unknown"
