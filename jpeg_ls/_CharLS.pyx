
# from __future__ import division, print_function, unicode_literals

import numpy as np
cimport numpy as np

JLS_ERROR_MESSAGES = {0: 'OK',
                      1: 'Invalid Jls Parameters',
                      2: 'Parameter Value Not Supported',
                      3: 'Uncompressed Buffer Too Small',
                      4: 'Compressed Buffer Too Small',
                      5: 'Invalid Compressed Data',
                      6: 'Too Much Compressed Data',
                      7: 'Image Type Not Supported',
                      8: 'Unsupported Bit Depth For Transform',
                      9: 'Unsupported Color Transform'}



cdef extern from 'define_charls_dll.h':
    pass

cdef extern from 'publictypes.h':
    cdef enum JLS_ERROR:
        pass

    cdef enum interleavemode:
        pass
        # ILV_NONE = 0
        # ILV_LINE = 1
        # ILV_SAMPLE = 2

    cdef struct JfifParameters:
        int Ver
        char units
        int XDensity
        int YDensity
        short Xthumb
        short Ythumb
        char* pdataThumbnail

    cdef struct JlsCustomParameters:
        int MAXVAL
        int T1
        int T2
        int T3
        int RESET

    cdef struct JlsParameters:
        int width
        int height
        int bitspersample
        int bytesperline
        int components
        int allowedlossyerror
        interleavemode ilv
        int colorTransform
        char outputBgr
        JlsCustomParameters custom
        JfifParameters jfif

cdef extern from 'interface.h':
    cdef JLS_ERROR JpegLsEncode(char* compressedData, size_t compressedLength, size_t* byteCountWritten,
                                char* uncompressedData, size_t uncompressedLength, JlsParameters* info)

    cdef JLS_ERROR JpegLsReadHeader(char* compressedData, size_t compressedLength, JlsParameters* info)

    cdef JLS_ERROR JpegLsDecode(char* uncompressedData, size_t uncompressedLength,
                                char* compressedData, size_t compressedLength, JlsParameters* info)


######################################


cdef JlsParameters build_parameters():

    cdef JlsCustomParameters info_custom
    info_custom.MAXVAL = 0
    info_custom.T1 = 0
    info_custom.T2 = 0
    info_custom.T3 = 0
    info_custom.RESET = 0

    cdef JfifParameters jfif
    jfif.Xthumb = 0
    jfif.Ythumb = 0

    cdef JlsParameters info

    info.width = 0
    info.height = 0
    info.bitspersample = 0
    info.bytesperline = 0
    info.components = 1    # number of components (RGB = 3, RGBA = 4, monochrome = 1;
    info.allowedlossyerror = 0   #  0 means lossless
    info.ilv = <interleavemode>0   # For monochrome images, always use ILV_NONE.
    info.colorTransform = 0   # 0 means no color transform
    info.outputBgr = 0  # when set to true, CharLS will reverse the normal RGB
    info.custom = info_custom
    info.jfif = jfif

    # Done.
    return info



def encode(data_image):
    """
    Encode grey-scale image via JPEG-LS using CharLS implementation.
    """

    data_image = np.asarray(data_image)

    if data_image.dtype == np.uint8:
        Bpp = 1
    elif data_image.dtype == np.uint16:
        Bpp = 2
    else:
        msg = 'Invalid input data type {}, expecting np.uint8 or np.uint16.'.format(data_image.dtype)
        raise Exception(msg)

    if len(data_image.shape) < 2 or len(data_image.shape) > 3:
        raise Exception('Invalid data shape')

    # Size of input image data.
    cdef int num_lines = data_image.shape[0]
    cdef int num_samples = data_image.shape[1]
    cdef int num_bands
    if len(data_image.shape) == 2:
        num_bands = 1
    else:
        num_bands = data_image.shape[2]

    if num_bands != 1:
        raise Exception('Invalid number of bands {}'.format(num_bands))


    cdef int max_val = np.max(data_image)
    cdef int max_bits = 0
    max_bits = int(np.ceil( np.log2(max_val + 1) ))

    if max_bits <= 1:
        max_bits = 2

    # Setup parameter structure.
    cdef JlsParameters info = build_parameters()

    info.width = num_samples
    info.height = num_lines
    info.components = num_bands
    info.ilv = <interleavemode>0

    info.bytesperline = num_samples * Bpp
    info.bitspersample = max_bits

    info.allowedlossyerror = 0

    # Buffer to store compressed data results.
    cdef size_t size_buffer = num_samples*num_lines*Bpp*2

    data_buffer = np.zeros(size_buffer, dtype=np.uint8)
    cdef char* data_buffer_ptr = <char*>np.PyArray_DATA(data_buffer)

    cdef size_t size_work

    # Pointers to input and output data.
    cdef char* data_image_ptr = <char*>np.PyArray_DATA(data_image)
    cdef JlsParameters* info_ptr = &info
    cdef size_t* size_work_ptr = &size_work

    # Call encoder function.
    cdef size_t size_data = num_samples*num_lines*Bpp
    cdef JLS_ERROR err
    err = JpegLsEncode(data_buffer_ptr, size_buffer, size_work_ptr,
                       data_image_ptr, size_data, info_ptr)

    if err != 0:
        raise Exception('Error calling CharLS: {}'.format(JLS_ERROR_MESSAGES[err]))

    # Finish.
    data_buffer = data_buffer[:size_work]

    # All done.
    return data_buffer



cdef JlsParameters _read_header(np.ndarray[np.uint8_t, ndim=1] data_buffer):
    """
    Decode grey-scale image via JPEG-LS using CharLS implementation.
    """

    # Size of input image, point to buffer.
    cdef int size_buffer = data_buffer.shape[0]

    #
    # Read compressed data header.
    #

    # Setup parameter structure.
    cdef JlsParameters info = build_parameters()

    # Pointers to input and output data.
    cdef char* data_buffer_ptr = <char*>np.PyArray_DATA(data_buffer)
    cdef JlsParameters* info_ptr = &info

    # Read the header.
    cdef JLS_ERROR err
    err = JpegLsReadHeader(data_buffer_ptr, size_buffer, info_ptr)

    if err != 0:
        raise Exception('Error calling CharLS: {}'.format(JLS_ERROR_MESSAGES[err]))

    # Done.
    return info #data_image



def read_header(np.ndarray[np.uint8_t, ndim=1] data_buffer):
    info = _read_header(data_buffer)

    header = {'width': info.width,
              'height': info.height,
              'bitspersample': info.bitspersample,
              'bytesperline': info.bytesperline,
              'components': info.components,
              'allowedlossyerror': info.allowedlossyerror,
              'ilv': info.ilv}

    return header



def decode(np.ndarray[np.uint8_t, ndim=1] data_buffer):
    """
    Decode compressed data into image array.
    """

    # Read header info.
    info = _read_header(data_buffer)

    # Sizes of data.
    if 2 <= info.bitspersample <= 8:
        Bpp = 1
    elif 9 <= info.bitspersample <= 16:
        Bpp = 2
    else:
        raise Exception('Invalid bitspersample: {}'.format(info.bitspersample))

    cdef int size_buffer = data_buffer.shape[0]

    cdef size_data = info.width * info.height * info.components * Bpp
    data_image = np.zeros(size_data, dtype=np.uint8)

    # Pointers to input and output data.
    cdef char* data_buffer_ptr = <char*>np.PyArray_DATA(data_buffer)
    cdef JlsParameters* info_ptr = &info
    cdef char* data_image_ptr = <char*>np.PyArray_DATA(data_image)

    # Decode compressed data.
    cdef JLS_ERROR err
    err = JpegLsDecode(data_image_ptr, size_data,
                       data_buffer_ptr, size_buffer, info_ptr)

    if err != 0:
        raise Exception('Error calling CharLS: {}'.format(JLS_ERROR_MESSAGES[err]))

    # Finish.
    num_samples = info.width
    num_lines = info.height
    num_bands = info.components

    if Bpp == 2:
        data_image = data_image.view(dtype=np.uint16)
        data_image = data_image.reshape(num_lines, num_samples, num_bands)
    else:
        data_image = data_image.reshape(num_lines, num_samples, num_bands)

    data_image = data_image.squeeze()

    # Done.
    return data_image
