



import os

from . import data_io
import jpeg_ls

# Read in an image from an existing PNG file.
fname_img = 'test/gray_raw.png'
data_image = data_io.read_PIL(fname_img)

# This input image should be a numpy array.
print('\nData properties:')
print('Type:  {:s}'.format(data_image.dtype))
print('Shape: {:s}'.format(data_image.shape))

# Compress image data to a sequence of bytes.
data_buffer = jpeg_ls.encode(data_image)

# Sizes.
size_png = os.path.getsize(fname_img)

print('\nSize of uncompressed image data: {:n}'.format(len(data_image.tostring())))
print('Size of PNG encoded data file:   {:n}'.format(size_png))
print('Size of JPEG-LS encoded data:    {:n}'.format(len(data_buffer)))

# Decompress.
data_image_b = jpeg_ls.decode(data_buffer)

# Compare image data, before and after.
is_same = (data_image == data_image_b).all()
print('\nRestored data is identical to original? {:s}\n'.format(str(is_same)))
