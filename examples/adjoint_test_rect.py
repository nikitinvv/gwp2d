from gwp2d import solver
import numpy as np

n = 256
nangles = 16
alpha = 1
beta = 3
ep = 1e-2
cl = solver.Solver(n, nangles, alpha, beta, ep)
f = np.random.random([n, 2*n]).astype('float32')
coeffs = cl.fwdmany(f)
fr = cl.adjmany(coeffs)

s0 = np.sum(f*np.conj(fr))
s1 = np.float32(0)
for k0 in range(len(coeffs)):
    for k1 in range(len(coeffs[0])):
        for k2 in range(len(coeffs[0][0])):
            s1 += np.sum(coeffs[k0][k1][k2]*np.conj(coeffs[k0][k1][k2]))

print('Adjoint test <fwd(f),fwd(f)> ?= <f,adj(fwd(f))> :', s0, '?=', s1)
