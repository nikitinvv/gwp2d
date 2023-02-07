import tomopy
import dxchange
import numpy as np
import tomovector as pt
import cupy as cp

# tomopy
objx = dxchange.read_tiff('data/phantom_00012/M4R1_mx.tif').astype('float32')
objy = dxchange.read_tiff('data/phantom_00012/M4R1_my.tif').astype('float32')
objz = dxchange.read_tiff('data/phantom_00012/M4R1_mz.tif').astype('float32')

npad = ((182, 182), (64, 64), (0, 0))
objx = np.pad(objx, npad, mode='constant', constant_values=0)
objy = np.pad(objy, npad, mode='constant', constant_values=0)
objz = np.pad(objz, npad, mode='constant', constant_values=0)

objx = tomopy.downsample(objx, level=1, axis=0)
objx = tomopy.downsample(objx, level=1, axis=1)
objx = tomopy.downsample(objx, level=1, axis=2)
objy = tomopy.downsample(objy, level=1, axis=0)
objy = tomopy.downsample(objy, level=1, axis=1)
objy = tomopy.downsample(objy, level=1, axis=2)
objz = tomopy.downsample(objz, level=1, axis=0)
objz = tomopy.downsample(objz, level=1, axis=1)
objz = tomopy.downsample(objz, level=1, axis=2)


theta = tomopy.angles(31, 90, 270)
prj1 = tomopy.project3(objx, objy, objz, theta, axis=0, pad=False)

dxchange.write_tiff(prj1,'data/prj1',overwrite=True)


# Fourier based


# init variables
ntheta = len(theta)
nz = objx.shape[0]
n = objx.shape[-1]
pnz = nz
center = n/2
ngpus = 1
theta = theta.astype('float32')


# obj
obj = np.zeros([3,nz,n,n],dtype='complex64')
obj[0] = np.array(objx)
obj[1] = np.array(objy)
obj[2] = np.array(objz)

# p
p = np.zeros([3,ntheta,1,1],dtype='float32')
p[0] = np.cos(theta)[:,np.newaxis,np.newaxis]
p[1] = -np.sin(theta)[:,np.newaxis,np.newaxis]
p[2] = 0#

with pt.SolverTomo(theta, p, ntheta, nz, n, pnz, center, ngpus) as slv:
    # FWD operator
    prj = slv.fwd_tomo_batch(obj)    
    dxchange.write_tiff(prj.real,'data/prj11',overwrite=True)
    # ADJ operator    
    aobj = slv.adj_tomo_batch(prj)

    t1 = np.sum(prj*np.conj(prj))    
    t2 = np.sum(aobj*np.conj(obj))        
    print(t1,t2)







# rec1, rec2, rec3 = tomopy.vector3(prj1, prj2, prj3, theta, theta, theta, axis1=0, axis2=1, axis3=2, num_iter=100)
# dxchange.write_tiff(rec1,'rec/rec1',overwrite=True)
# dxchange.write_tiff(rec2,'rec/rec2',overwrite=True)
# dxchange.write_tiff(rec3,'rec/rec3',overwrite=True)