from numpy.random import default_rng
from numpy.random import BitGenerator


def create_rng(seed=None):
    return default_rng(seed)
