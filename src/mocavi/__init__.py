from importlib.metadata import version

from ._ebmodel import EBMoCAVIModel
from ._ebmodule import EBMoCAVIModule
from ._model import MoCAVIModel
from ._module import MoCAVIModule

__all__ = ["MoCAVIModel", "MoCAVIModule", "EBMoCAVIModel", "EBMoCAVIModule"]

__version__ = version("mocavi")
