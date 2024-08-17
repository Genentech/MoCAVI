from importlib.metadata import version

from ._model import MoCAVIModel
from ._module import MoCAVIModule

__all__ = ["MoCAVIModel", "MoCAVIModule"]

__version__ = version("mocavi")
