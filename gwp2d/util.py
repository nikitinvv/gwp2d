import numpy as np
import cupy as cp
import os
import struct



def rotate(x, theta, reverse=False):
    """Rotate coordinates with respect to the angle theta
    """
    R = np.array([[cp.cos(theta), -cp.sin(theta)],
                  [cp.sin(theta), cp.cos(theta)]])
    if(reverse):
        R = R.swapaxes(0, 1)
    xr = cp.zeros(x.shape, dtype='float32')
    xr[:, 0] = R[1,0]*x[:, 1] + R[1, 1]*x[:, 0]
    xr[:, 1] = R[0,0]*x[:, 1] + R[0, 1]*x[:, 0]
    return xr

def checkerboard(array, inverse=False):
    """In-place FFTshift for even sized grids only.
    If and only if the dimensions of `array` are even numbers, flipping the
    signs of input signal in an alternating pattern before an FFT is equivalent
    to shifting the zero-frequency component to the center of the spectrum
    before the FFT.
    """
    def g(x):
        return 1 - 2 * (x % 2)

    for i in range(2):
        array = cp.moveaxis(array, i, -1)
        array *= g(cp.arange(array.shape[-1]) + 1)
        if inverse:
            array *= g(array.shape[-1] // 2)
        array = cp.moveaxis(array, -1, i)
    return array
