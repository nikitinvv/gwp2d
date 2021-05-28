from gwp2d import solver
import numpy as np
import dxchange

n = 256
nangles = 16
alpha = 1
beta = 3
ep = 1e-3

# pick angle
ang = 14
# pick level
level = 2

# init coefficients
cl = solver.Solver(n, nangles, alpha, beta, ep)
f = np.random.random([n, n]).astype('float32')
coeffs = cl.fwd(f)

# zero coefficients and set only the one as 1
cshape = coeffs[level][0].shape
for k in range(len(coeffs)):
    coeffs[k][:] = 0
# coeffs[level][ang, cshape[0]//2, cshape[1]//2] = 1
coeffs[level][ang, cshape[0]//2, cshape[1]//2+16] = 1

# recover gwp
fr = cl.adj(coeffs)

rname = 'data/fr'+str(ang)+'_'+str(level)
iname = 'data/fi'+str(ang)+'_'+str(level)

dxchange.write_tiff(fr.real, rname, overwrite=True)
dxchange.write_tiff(fr.imag, iname, overwrite=True)

print('gwp real part saved in'+rname+'.tiff')
print('gwp imag part saved in'+iname+'.tiff')
