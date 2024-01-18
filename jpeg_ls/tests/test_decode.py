from pathlib import Path

import pytest
import numpy as np

from jpeg_ls import decode, read
from _CharLS import read_header, decode_from_buffer

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
    return arr.reshape((256, 256))


class TestRead:
    """Tests for read()"""

    @pytest.mark.xfail
    def test_T8C0E0(self):
        # Decodes, but the output is weird
        # TEST8 in colour mode 0, lossless
        arr = read(DATA / "T8C0E0.JLS")
        assert arr.shape == (256, 256)
        assert np.array_equal(arr, TEST8)

    @pytest.mark.xfail
    def test_T8C0E3(self):
        # Decodes, but the output is weird
        # TEST8 in colour mode 0, lossy
        arr = read(DATA / "T8C0E0.JLS")
        assert arr.shape == (256, 256)
        assert np.allclose(arr, TEST8, atol=3)

    def test_T8C1E0(self, TEST8):
        # TEST8 in colour mode 1, lossless
        arr = read(DATA / "T8C1E0.JLS")
        assert np.array_equal(arr, TEST8)

    def test_T8C1E3(self, TEST8):
        # TEST8 in colour mode 1, lossy
        arr = read(DATA / "T8C1E3.JLS")
        assert np.allclose(arr, TEST8, atol=3)

    def test_T8C2E0(self, TEST8):
        # TEST8 in colour mode 2, lossless
        arr = read(DATA / "T8C2E0.JLS")
        assert np.array_equal(arr, TEST8)

    def test_T8C2E3(self, TEST8):
        # TEST8 in colour mode 2, lossy
        arr = read(DATA / "T8C2E3.JLS")
        assert np.allclose(arr, TEST8, atol=3)

    def test_T8NDE0(self, TEST8BS2):
        # TEST8 lossless, non-default parameters
        arr = read(DATA / "T8NDE0.JLS")
        assert np.array_equal(arr, TEST8BS2)

    def test_T8NDE3(self, TEST8BS2):
        # TEST8 lossy, non-default parameters
        arr = read(DATA / "T8NDE3.JLS")
        assert np.allclose(arr, TEST8BS2, atol=3)

    @pytest.mark.xfail
    def test_T8SSE0(self, TEST8):
        # The JPEG-LS stream is encoded with a parameter value that is not
        #   supported by the CharLS decoder
        # TEST8 lossless
        arr = read(DATA / "T8SSE0.JLS")
        assert np.array_equal(arr, TEST8)

    @pytest.mark.xfail
    def test_T8SSE3(self, TEST8):
        # The JPEG-LS stream is encoded with a parameter value that is not
        #   supported by the CharLS decoder
        # TEST8 lossy
        arr = read(DATA / "T8SSE3.JLS")
        assert np.allclose(arr, TEST8, atol=3)

    def test_T16E0(self, TEST16):
        # TEST16 lossless
        arr = read(DATA / "T16E0.JLS")
        assert np.array_equal(arr, TEST16)

    def test_T16E3(self, TEST16):
        # TEST16 lossy
        arr = read(DATA / "T16E3.JLS")
        assert np.allclose(arr, TEST16, atol=3)

    def test_decode_failure(self):
        msg = (
            "The JPEG-LS stream is encoded with a parameter value that is not "
            "supported by the CharLS decoder"
        )
        with pytest.raises(RuntimeError, match=msg):
            read(DATA / "T8SSE0.JLS")


def test_decode(TEST8):
    """Test decode()"""
    with open(DATA / "T8C1E0.JLS", "rb") as f:
        arr = decode(np.frombuffer(f.read(), dtype="u1"))
        assert np.array_equal(arr, TEST8)


def test_decode_buffer(TEST8):
    with open(DATA / "T8C1E0.JLS", "rb") as f:
        buffer = decode_from_buffer(f.read())
        arr = np.frombuffer(buffer, dtype="u1")
        arr = arr.reshape(256, 256, 3)
        assert np.array_equal(arr, TEST8)


class TestReadHeader:
    """Tests for _CharLS.read_header()"""

    def test_read_header(self):
        """Test read_header()"""
        with open(DATA / "T8C1E0.JLS", "rb") as f:
            buffer = f.read()

        info = read_header(buffer)
        assert info["width"] == 256
        assert info["height"] == 256
        assert info["bits_per_sample"] == 8
        assert info["stride"] == 768
        assert info["components"] == 3
        assert info["allowed_lossy_error"] == 0
        assert info["interleave_mode"] == 1
        assert info["colour_transformation"] == 0

    def test_read_header2(self):
        """Test read_header()"""
        with open(DATA / "T8C0E0.JLS", "rb") as f:
            buffer = f.read()

        info = read_header(buffer)
        assert info["width"] == 256
        assert info["height"] == 256
        assert info["bits_per_sample"] == 8
        assert info["stride"] == 256
        assert info["components"] == 3
        assert info["allowed_lossy_error"] == 0
        assert info["interleave_mode"] == 0
        assert info["colour_transformation"] == 0

    def test_read_header_raises(self):
        """Test decoding error with read_header()"""
        with open(DATA / "T8SSE0.JLS", "rb") as f:
            buffer = f.read()

        msg = (
            "Decoding error: The JPEG-LS stream is encoded with a parameter "
            "value that is not supported by the CharLS decoder"
        )
        with pytest.raises(RuntimeError, match=msg):
            read_header(buffer)
