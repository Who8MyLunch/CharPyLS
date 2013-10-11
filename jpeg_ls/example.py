

from __future__ import division, print_function, unicode_literals

import os

import data_io
import jpeg_ls

# Read in an image from an existing PNG file.
fname_img = 'test/image.png'
data_image = data_io.read_PIL(fname_img)

# Compress image data to a sequence of bytes.
data_buffer = jpeg_ls.encode(data_image)

# Sizes.
size_png = os.path.getsize(fname_img)
print('Size of RGB 8-bit image data:  {:n}'.format(len(data_image.tostring())))
print('Size of PNG encoded data file: {:n}'.format(size_png))
print('Size of JPEG-LS encoded data:  {:n}'.format(len(data_buffer)))

# Decompress.
data_image_b = jpeg_ls.decode(data_buffer)

# Compare.
is_same = (data_image == data_image_b).all()
print('Restored data is identical to original: {:s}'.format(str(is_same)))
