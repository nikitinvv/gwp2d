from gwp2d import solver
import numpy as np
import dxchange
import matplotlib.pyplot as plt

n = 128
nangles = 12
alpha = 1
beta = 3
ep = 1e-3

# init coefficients
cl = solver.Solver(n, nangles, alpha, beta, ep)
f = dxchange.read_tiff('data/lens_srt_256-256.tiff')[:,:]
coeffs = cl.fwdmany(f)
fr = cl.adjmany(coeffs)
rname = 'data/rec'
dxchange.write_tiff(fr.real, rname, overwrite=True)
print('data saved in'+rname+'.tiff')
