
import PIL.Image
import numpy as np


def read(fp):
    """Read image data from file-like object using PIL.  Return Numpy array.
    """
    with open(fp, 'rb') as fpp:
        img = PIL.Image.open(fpp)
        data = np.asarray(img)

    return data



def write(fp, data, fmt=None, **kwargs):
    """Write image data from Numpy array to file-like object.

    File format is automatically determined from fp if it's a filename, otherwise you
    must specify format via fmt keyword, e.g. fmt = 'png'.

    Parameter options: http://pillow.readthedocs.io/en/4.2.x/handbook/image-file-formats.html
    """
    img = PIL.Image.fromarray(data)
    img.save(fp, format=fmt, **kwargs)
