


import numpy as np

import _CharLS

#################################################

def read(fname):
    """
    Read image data from JPEG-LS file.
    """
    with open(fname, 'rb') as fo:
        string_buffer = fo.read()

    data_buffer = np.fromstring(string_buffer, dtype=np.uint8)

    data_image = _CharLS.decode(data_buffer)

    # Done.
    return data_image


def write(fname, data_image):
    """
    Write compressed image data to JPEG-LS file.
    """
    data_buffer = _CharLS.encode(data_image)

    with open(fname, 'wb') as fo:
        fo.write(data_buffer.tostring())

    # Done.

#################################################


def encode(data_image):
    """
    Encode grey-scale image via JPEG-LS using CharLS implementation.
    """

    if data_image.dtype == np.uint16 and np.max(data_image) <= 255:
        data_image = data_image.astype(np.uint8)

    data_buffer = _CharLS.encode(data_image)

    return data_buffer



def decode(data_buffer):
    """
    Decode grey-scale image via JPEG-LS using CharLS implementation.
    """

    data_image = _CharLS.decode(data_buffer)

    return data_image

#################################################


if __name__ == '__main__':
    """
    Development and testing.
    """

    import os
    from . import data_io

    # # data prep.
    # path = os.path.dirname(os.path.abspath(__file__))
    # fname = 'IMG_20120129_120644.jpg'

    # f = os.path.join(path, fname)
    # data, meta = io.read(f)

    # data = np.mean(data, axis=2)

    # data -= data.min()
    # data /= data.max()

    # data = (data * 255).astype(np.uint8)

    # data_io.write('gray_raw.dat', data)
    # data_io.write('gray_raw.png', data)

    path = os.path.dirname(os.path.abspath(__file__))
    fname = 'gray_raw.dat'

    f = os.path.join(path, fname)
    image_gray, meta = data_io.read(f)

    image_gray = image_gray.squeeze()

    buff = encode(image_gray)

    with open('file.jls', 'wb') as fo:
        fo.write(buff.tostring())

