from gwp2d import solver
import numpy as np
import struct
import dxchange
import scipy.ndimage as ndimage

f = dxchange.read_tiff('data/lens_srt_128-128-128.tiff')[64]
f = ndimage.zoom(f,2)
dxchange.write_tiff(f,'data/lens_srt_256-256.tiff')


