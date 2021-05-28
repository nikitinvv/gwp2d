from gwp2d import solver
import numpy as np
import dxchange
import matplotlib.pyplot as plt

n = 128
nangles = 14
alpha = 1
beta = 3
ep = 1e-3

# pick angle
ang = 12
# pick level
level = 1

# init coefficients
cl = solver.Solver(n, nangles, alpha, beta, ep)
f = np.random.random([2*n, 2*n]).astype('float32')

coeffs = cl.fwdmany(f)

# zero coefficients and set some as 1
cshape = coeffs[0][0][level][0].shape
for k0 in range(len(coeffs)):
    for k1 in range(len(coeffs[k0])):
        for k2 in range(len(coeffs[k0][k1])):
            coeffs[k0][k1][k2][:] = 0

coeffs[0][1][level][ang, cshape[0]//2-8, cshape[1]//2-12] = 1
coeffs[0][0][level][ang, cshape[0]//2+6, cshape[1]//2+23] = 1
coeffs[1][0][level][ang, cshape[0]//2-6, cshape[1]//2+20] = 1
coeffs[0][0][level][ang, cshape[0]//2+5, cshape[1]//2-24] = 1

fr = cl.adjmany(coeffs)

rname = f'data/fr{ang}_{level}'
iname = f'data/fi{ang}_{level}'

dxchange.write_tiff(fr.real, rname, overwrite=True)
dxchange.write_tiff(fr.imag, iname, overwrite=True)

print('gwp real part saved in'+rname+'.tiff')
print('gwp imag part saved in'+iname+'.tiff')