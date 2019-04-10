from .detector import detect
from .training import Model
from os import path


with open(path.join(path.abspath(path.dirname(__file__)),"VERSION"),"r") as f:
    __version__ = f.read().strip()
