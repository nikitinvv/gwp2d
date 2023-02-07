import cupy as cp
import dxchange
import numpy as np
import tomovector as pt

if __name__ == "__main__":

    # Load object
    u = np.load('data/obj.npy')
    #init sizes and parameters
    [nz,n] = u.shape[1:3]
    ntheta = 100    
    ngpus = 1
    pnz = nz
    niter = 64
    
    # init angles
    theta = np.linspace(0,np.pi,ntheta).astype('float32')
    # init p
    p = np.zeros([3,ntheta,1,1],dtype='float32')
    p[0] = np.cos(theta)[:,np.newaxis,np.newaxis]
    p[1] = -np.sin(theta)[:,np.newaxis,np.newaxis]
    p[2] = 0#
    
    with pt.SolverTomo(theta, p, ntheta, nz, n, pnz, n/2, ngpus) as slv:
        # generate data
        data = slv.fwd_tomo_batch(u)
        # initial guess
        ur = slv.cg_tomo_batch(data, u*0, niter)
    
    # save results
    dxchange.write_tiff(ur[0].real,  'rec/urx', overwrite=True)
    dxchange.write_tiff(ur[1].real,  'rec/ury', overwrite=True)
    dxchange.write_tiff(ur[2].real,  'rec/urz', overwrite=True)
    dxchange.write_tiff(u[0].real,  'rec/ux', overwrite=True)
    dxchange.write_tiff(u[1].real,  'rec/uy', overwrite=True)
    dxchange.write_tiff(u[2].real,  'rec/uz', overwrite=True)            
    dxchange.write_tiff(data.real,  'data/d', overwrite=True)
    