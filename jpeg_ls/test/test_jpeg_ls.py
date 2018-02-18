


import os
import unittest
import numpy as np
from jpeg_ls import data_io

import jpeg_ls as jls
import jpeg_ls.CharLS

class Test_Jpeg_LS(unittest.TestCase):

    def setUp(self):
        self.path_module = os.path.dirname( os.path.abspath( __file__ ) )
        self.fname = os.path.join(self.path_module, 'gray_raw.dat')
        self.fname16 = os.path.join(self.path_module, 'gray16_raw.dat')
        self.fname16_254 = os.path.join(self.path_module, 'gray16_254.dat')
        self.fname_resid = os.path.join(self.path_module, 'band_resid.dat')

    def tearDown(self):

        f = 'data_temp.jls'
        if os.path.isfile(f):
            os.remove(f)

    def test_encode_uint8(self):
        data, meta = data_io.read(self.fname)

        data_comp = jls.encode(data)

        msg = 'oops size={:d}'.format(data_comp.size)
        self.assertTrue(data_comp.size < 2090000, msg)

    def test_encode_uint16(self):
        data, meta = data_io.read(self.fname16)

        data_comp = jls.encode(data)

        msg = 'oops size={:d}'.format(data_comp.size)
        self.assertTrue(data_comp.size < 2740000, msg)

    def test_encode_uint16_squeeze(self):
        data, meta = data_io.read(self.fname16)

        data = data.squeeze()
        data_comp = jls.encode(data)

        msg = 'oops size={:d}'.format(data_comp.size)
        self.assertTrue(data_comp.size < 2740000, msg)


    def test_encode_band_resid(self):
        data, meta = data_io.read(self.fname_resid)

        data = data.squeeze()
        data_comp = jls.encode(data)

        msg = 'oops size={:d}'.format(data_comp.size)
        self.assertTrue(data_comp.size < 24000, msg)


    def test_encode_to_file(self):
        data, meta = data_io.read(self.fname)

        fname_temp = os.path.join(self.path_module, 'data_temp.jls')
        jls.write(fname_temp, data)

        file_size = os.path.getsize(fname_temp)
        msg = 'oops file_size={:d}'.format(file_size)
        self.assertTrue(file_size < 2090000, msg)


    def test_read_header(self):
        data, meta = data_io.read(self.fname)
        data_comp = jls.encode(data)

        header = jls.CharLS._CharLS.read_header(data_comp)

        self.assertTrue(header['width'] == 2592)
        self.assertTrue(header['height'] == 1944)
        self.assertTrue(header['bitspersample'] == 8)
        self.assertTrue(header['bytesperline'] == 2592)
        self.assertTrue(header['components'] == 1)
        self.assertTrue(header['allowedlossyerror'] == 0)
        self.assertTrue(header['ilv'] == 0)


    def test_encode_decode_compare_uint8(self):
        data, meta = data_io.read(self.fname)

        # Compress, decompress.
        data_comp = jls.encode(data)

        data_image = jls.decode(data_comp)

        diff = np.sum( (data.squeeze().astype(np.int) - data_image.astype(np.int))**2)
        self.assertTrue(diff == 0)


    def test_encode_decode_compare_uint16(self):
        data, meta = data_io.read(self.fname16)

        # Compress, decompress.
        data_comp = jls.encode(data)

        data_image = jls.decode(data_comp)

        diff = np.sum( (data.squeeze().astype(np.int) - data_image.astype(np.int))**2)
        self.assertTrue(diff == 0)


    def test_encode_decode_compare_uint16_254(self):
        data, meta = data_io.read(self.fname16_254)

        self.assertTrue(data.max() < 255)

        # Compress, decompress.
        data_comp = jls.encode(data)

        data_image = jls.decode(data_comp)

        diff = np.sum( (data.squeeze().astype(np.int) - data_image.astype(np.int))**2)
        self.assertTrue(diff == 0)


# Standalone.
if __name__ == '__main__':
    unittest.main(verbosity=2)
