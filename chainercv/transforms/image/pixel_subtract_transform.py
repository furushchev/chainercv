#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from __future__ import print_function
import numpy as np

# Const pixel values
VGG_PIXEL_SUBTRACT_BGR = np.array([103.939, 116.779, 123.68], np.float32)


def pixel_subtract(pixel_value=VGG_PIXEL_SUBTRACT_BGR):
    """Subtract each pixel of image. This is used for averaging images.

    Args:
        x (~numpy.ndarray): array to be transformed. Ths is in CHW format.
        pixel_value (~numpy.ndarray): The pixel value which is subtracted from each pixels of x.
    """

    if not isinstance(pixel_value, np.ndarray):
        pixel_value = np.asarray(pixel_value)

    if pixel_value.ndim == 0:
        pixel_value = pixel_value[None, None, None]
    elif pixel_value.ndim == 1:
        pixel_value = pixel_value[:, None, None]
    elif pixel_value.ndim != 3:
        raise ValueError('Only 0d, 1d or 3d array is accepted for pixel value: %s' % str(pixel_value))

    x = x - pixel_value

    return x

if __name__ == '__main__':
    from skimage.data import astronaut
    data = astronaut().astype(np.float32)
    out = pixel_subtract(data.transpose(2,0,1))

    mean = data - VGG_PIXEL_SUBTRACT_BGR

    assert mean.mean() - out.transpose(1,2,0).mean() == 0.0
