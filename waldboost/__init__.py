from .training import Learner, DStump, DTree
from .samples import Pool
from .model import Model
from .channels import grad_mag, grad_hist
from . import fpga, channels


# with open(path.join(path.abspath(path.dirname(__file__)),"VERSION"),"r") as f:
#     __version__ = f.read().strip()


load_model = Model.load


def save_model(model, filename):
    model.save(filename)
