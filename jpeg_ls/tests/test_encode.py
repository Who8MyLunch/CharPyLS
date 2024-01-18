from tempfile import TemporaryDirectory
from pathlib import Path

import pytest
import numpy as np

from jpeg_ls import decode, encode, write, read
from _CharLS import encode_to_buffer, decode_from_buffer


DATA = Path(__file__).parent / "jlsimV100"


@pytest.fixture
def TEST8():
    # 8-bit colour test image
    # p6, 256 x 256 x 3, uint8, RGB
    p = DATA / "TEST8.PPM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((256, 256, 3))


@pytest.fixture
def TEST8R():
    # Red component of TEST8
    # p5, 256 x 256 x 1, 255, uint8, greyscale
    p = DATA / "TEST8R.PGM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((256, 256))


@pytest.fixture
def TEST8G():
    # Green component of TEST8
    # p5, 256 x 256 x 1, 255, uint8, greyscale
    p = DATA / "TEST8G.PGM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((256, 256))


@pytest.fixture
def TEST8B():
    # Blue component of TEST8
    # p5, 256 x 256 x 1, 255, uint8, greyscale
    p = DATA / "TEST8B.PGM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((256, 256))


@pytest.fixture
def TEST8BS2():
    # Blue component of TEST8, subsampled 2X in horizontal and vertical
    # p5, 128 x 128 x 1, 255, uint8, greyscale
    p = DATA / "TEST8BS2.PGM"
    arr = np.fromfile(p, dtype="u1", offset=15)
    return arr.reshape((128, 128))


@pytest.fixture
def TEST8GR4():
    # Gren component of TEST8, subsampled 4X in the vertical
    # p5, 256 x 64 x 1, 255, uint8, greyscale
    p = DATA / "TEST8GR4.PGM"
    arr = np.fromfile(p, dtype="u1", offset=14)
    return arr.reshape((64, 256))


@pytest.fixture
def TEST16():
    # 12-bit greyscale, big-endian
    # p5, 256 x 256 x 1, 4095, uint16, greyscale
    p = DATA / "TEST16.PGM"
    arr = np.fromfile(p, dtype=">u2", offset=16)
    return arr.reshape((256, 256)).astype("<u2")


class TestEncode:
    """Tests for encode()"""

    def test_invalid_dtype_raises(self):
        msg = "Invalid input data type 'float64', expecting np.uint8 or np.uint16"
        with pytest.raises(Exception, match=msg):
            encode(np.empty((2, 2), dtype=float))

    def test_invalid_nr_components_raises(self):
        msg = (
            "Encoding error: The component count argument is outside the range"
            "[1, 255]"
        )
        with pytest.raises(Exception, match=msg):
            encode(np.empty((2, 2, 256), dtype="u1"))

    def test_invalid_shape_raises(self):
        msg = "Invalid data shape"
        with pytest.raises(Exception, match=msg):
            encode(np.empty((2,), dtype="u1"))

        with pytest.raises(Exception, match=msg):
            encode(np.empty((2, 2, 2, 2), dtype="u1"))

    def test_TEST8(self, TEST8):
        buffer = encode(TEST8)
        assert isinstance(buffer, np.ndarray)  # weird choice
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8)

    def test_TEST8R(self, TEST8R):
        buffer = encode(TEST8R)
        assert isinstance(buffer, np.ndarray)
        assert buffer.shape[0] < TEST8R.shape[0] * TEST8R.shape[1]
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8R)

    def test_TEST8G(self, TEST8G):
        buffer = encode(TEST8G)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8G)

    def test_TEST8B(self, TEST8B):
        buffer = encode(TEST8B)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8B)

    def test_TEST8BS2(self, TEST8BS2):
        buffer = encode(TEST8BS2)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8BS2)

    def test_TEST8GR4(self, TEST8GR4):
        buffer = encode(TEST8GR4)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST8GR4)

    def test_TEST16(self, TEST16):
        buffer = encode(TEST16)
        assert isinstance(buffer, np.ndarray)
        arr = decode(buffer)
        assert np.array_equal(arr, TEST16)


class TestEncodeBytes:
    def test_invalid_lossy_raises(self, TEST8):
        msg = (
            "Encoding error: The near lossless argument is outside the range" "[0, 255]"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_to_buffer(TEST8, lossy_error=-1)

        with pytest.raises(RuntimeError, match=msg):
            encode_to_buffer(TEST8, lossy_error=256)

    def test_TEST8_lossy(self, TEST8):
        buffer = encode_to_buffer(TEST8, lossy_error=5)
        assert isinstance(buffer, bytearray)
        arr = np.frombuffer(decode_from_buffer(buffer), dtype="u1")
        arr = arr.reshape(TEST8.shape)
        assert np.allclose(arr, TEST8, atol=5)

    def test_TEST8R_lossy(self, TEST8R):
        buffer = encode_to_buffer(TEST8R, lossy_error=3)
        assert isinstance(buffer, bytearray)
        arr = np.frombuffer(decode_from_buffer(buffer), dtype="u1")
        arr = arr.reshape(TEST8R.shape)
        assert np.allclose(arr, TEST8R, atol=3)


def test_write(TEST8R):
    """Test write()"""
    with TemporaryDirectory() as tdir:
        p = Path(tdir)
        write(p / "test.jls", TEST8R)

        arr = read(p / "test.jls")
        assert np.array_equal(arr, TEST8R)
