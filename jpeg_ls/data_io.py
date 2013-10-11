
from __future__ import division, print_function, unicode_literals

import PIL
import PIL.Image
import numpy as np


def read_PIL(fname):
    """Read image file using PIL."""

    img = PIL.Image.open(fname)
    data = np.asarray(img)

    return data


def write_PIL(fname, data):
    """Write image file using PIL."""

    img = PIL.Image.fromarray(data)
    img.save(fname)
