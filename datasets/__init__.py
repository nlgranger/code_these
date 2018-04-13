import os

from .ch14dataset.dataset import Chalearn2014Dataset
from .ch17dataset.dataset import Chalearn2017Dataset
from .devisign.dataset import DEVISIGNDataset

try:
    chalearn2014 = Chalearn2014Dataset(
        os.path.join(os.path.dirname(__file__), "ch14dataset", "data"))
except FileNotFoundError:
    pass

try:
    chalearn2017 = Chalearn2017Dataset(
        os.path.join(os.path.dirname(__file__), "ch17dataset", "data"))
except FileNotFoundError:
    pass

try:
    devisign = DEVISIGNDataset(
        os.path.join(os.path.dirname(__file__),
                     "devisign", "data"),
        os.path.join(os.path.dirname(__file__), "devisign", "cache"))
except FileNotFoundError:
    pass
