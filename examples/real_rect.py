from gwp2d import solver
import numpy as np
import dxchange
import matplotlib.pyplot as plt

n = 64
nangles = 128
alpha = 1
beta = 3
ep = 1e-3

# init coefficients
cl = solver.Solver(n, nangles, alpha, beta, ep)

f = np.zeros([1024,64],dtype='float32') 
f[:,:50] = np.load('data/seism.npy').swapaxes(0,1)[1500:1500+1024]
dxchange.write_tiff(f,'data/seism.tiff', overwrite=True)

coeffs = cl.fwdmany(f)
fr = cl.adjmany(coeffs)
rname = 'data/rec'
dxchange.write_tiff(fr.real, rname, overwrite=True)
