import numpy as np

import _CharLS


def read(fname):
    """Read image data from JPEG-LS file."""
    with open(fname, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)

    return _CharLS.decode(arr)


def write(fname, data_image):
    """Write compressed image data to JPEG-LS file."""
    data_buffer = _CharLS.encode(data_image)

    with open(fname, "wb") as f:
        f.write(data_buffer.tobytes())


def encode(data_image):
    """Encode grey-scale image via JPEG-LS using CharLS implementation."""
    if data_image.dtype == np.uint16 and np.max(data_image) <= 255:
        data_image = data_image.astype(np.uint8)

    return _CharLS.encode(data_image)


def decode(data_buffer):
    """Decode grey-scale image via JPEG-LS using CharLS implementation."""
    return _CharLS.decode(data_buffer)
