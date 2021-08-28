import os
import numpy as np


def load_data(path):
    return np.load(path, allow_pickle=True)
