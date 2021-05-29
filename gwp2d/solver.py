import numpy as np
import cupy as cp
from gwp2d import util
from gwp2d import usfft
from collections import defaultdict
import matplotlib.pyplot as plt

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


class Solver():
    """Provides forward and adjoint operators for Gaussian Wave-packet (GWP) decompositon on
    GPU with using cupy library. For details see http://www.mathnet.ru/links/f0cbda0c8155c9c4d0ff6dd015c9ec78/vmp881.pdf
    """

    def __init__(self, n, nangles, alpha, beta, eps, levels=None):
        # init box parameters for covering the spectrum (see paper)
        nf_start = np.int32(np.log2(n/64)+0.5)
        K = 3*nf_start
        step = (nf_start+1)/(K-1)
        if(levels == None):
            nf = 2**(nf_start-range(K)*step)
        else:
            levels = np.array(levels)
            levels = np.sort(levels[levels < K])
            nf = 2**(nf_start-levels*step)
            K = len(levels)
            print('Set manual levels', levels)
        if K == 0:
            nf = np.array([0.5])
            K = 1

        xi_cent = np.int32(n/nf/(2*np.pi)/2)*2
        lam1 = 4*np.log(2)*(xi_cent/alpha)**2
        lam0 = lam1/beta**2
        lambda0 = np.pi**2/lam0
        lambda1 = np.pi**2/lam1
        # box sizes in the frequency domain
        L0 = 4*np.int32(np.round(np.sqrt(-np.log(eps)/lambda0)))
        L1 = 4*np.int32(np.round(np.sqrt(-np.log(eps)/lambda1)))

        boxshape = np.array([L0, L1]).swapaxes(0, 1)
        fgridshape = np.array([2 * n] * 2)
        # init angles
        theta = cp.linspace(0, 2*np.pi, nangles, endpoint=False)

        # box grid in frequency
        y = ((cp.arange(-L0[-1]//2, L0[-1]//2)))
        x = ((cp.arange(-L1[-1]//2, L1[-1]//2)))
        [y, x] = cp.meshgrid(y, x, indexing='ij')
        x = cp.array([y.flatten(), x.flatten()]).astype(
            'float32').swapaxes(0, 1)

        # init gaussian wave-packet basis for each layer
        gwpf = [None]*K
        for k in range(K):
            xn = (cp.arange(-L1[k]/2, L1[k]/2))/2
            yn = (cp.arange(-L0[k]/2, L0[k]/2))/2
            [yn, xn] = cp.meshgrid(yn, xn, indexing='ij')
            gwpf[k] = cp.exp(-lambda1[k]*xn*xn-lambda0[k]*yn*yn)
            gwpf[k] = gwpf[k].astype('float32').flatten()

        # find grid index for extracting boxes on layers>1 (small ones)
        # from the box on the the last layer (large box)
        inds = [None]*K
        phase = [None]*K
        phasemanyx = [None]*K
        phasemanyy = [None]*K

        for k in range(K):
            xi_centk = np.int32(xi_cent[k])
            xi_centK = np.int32(xi_cent[-1])
            xst = xi_centk-L1[k]//2+L1[-1]//2-xi_centK
            yst = -L0[k]//2+L0[-1]//2
            indsy, indsx = cp.meshgrid(
                cp.arange(yst, yst+L0[k]), cp.arange(xst, xst+L1[k]), indexing='ij')
            inds[k] = (indsx+indsy*L1[-1]).astype('int32').flatten()
            # to understand
            phase[k] = -2*np.pi*cp.arange(-L1[k]//2, L1[k]//2)/L1[k]*xi_centk
            phasemanyx[k] = -2*np.pi*cp.arange(-L1[k]//2, L1[k]//2)
            phasemanyy[k] = -2*np.pi*cp.arange(-L0[k]//2, L0[k]//2)

        # 2D USFFT plan
        U = usfft.Usfft(n, eps)

        # find spectrum subregions for each rotated box
        subregion = np.zeros([nangles, 4], dtype='int32')
        for ang in range(nangles):
            xr = x.copy()
            xr[:, 1] += xi_cent[-1]
            xr = util.rotate(xr, theta[ang])
            ell = np.round(xr + cp.array(fgridshape)/2).astype(np.int32)
            # borders in y
            subregion[ang, 0] = cp.min(ell[:, 0])-U.m
            subregion[ang, 1] = cp.max(ell[:, 0])+U.m
            # borders in x
            subregion[ang, 2] = cp.min(ell[:, 1])-U.m
            subregion[ang, 3] = cp.max(ell[:, 1])+U.m

        print('number of levels:', K)
        for k in range(K):
            print('box size on level', k, ':', boxshape[k])
            print('box center in frequency for level', k, ':', xi_cent[k])
        self.U = U
        self.nangles = nangles
        self.boxshape = boxshape
        self.fgridshape = fgridshape
        self.K = K
        self.theta = theta
        self.x = x
        self.gwpf = gwpf
        self.inds = inds
        self.xi_cent = xi_cent
        self.subregion = subregion
        self.phase = phase
        self.phasemanyx = phasemanyx
        self.phasemanyy = phasemanyy
        # free cupy pools
        # mempool.free_all_blocks()
        # pinned_mempool.free_all_blocks()

    def subregion_ids(self, ang):
        """
        Calculate indices for the subregion in the global grid with wrapping 
        for the gathering operation in the forward transform
        Parameters
        ----------
        ang : float32
            Box orientation angle
        Returns
        -------        
        yg,xg: float32
            y,x arrays of coordinates
        """
        # extract subregion
        yg = cp.arange(
            self.subregion[ang, 0], self.subregion[ang, 1]) % self.fgridshape[0]
        xg = cp.arange(
            self.subregion[ang, 2], self.subregion[ang, 3]) % self.fgridshape[1]
        return yg, xg

    def coordinates_fwd(self, ang):
        """
        Compute coordinates of the local box grid on the last layer 
        for the gathering operation in the adjoint transform
        Parameters
        ----------
        ang : float32
            Box orientation angle
        Returns
        -------        
        xr: [Ns,2]: float32
            (y,x) array of coordinates
        """
        # shift and rotate box on the last layer
        xr = self.x.copy()
        xr[:, 1] += self.xi_cent[-1]
        xr = util.rotate(xr, self.theta[ang])

        # plot boxes
        # plt.plot(xr[:,1].get(),xr[:,0].get(),'.')

        # move to the subregion
        xr[:, 0] -= self.subregion[ang, 0]
        xr[:, 1] -= self.subregion[ang, 2]
        # switch to [-1,1) interval w.r.t. global grid
        xr /= cp.array(self.fgridshape)
        return xr

    def coordinates_adj(self, ang):
        """
        Compute coordinates of the global grid 
        for the gathering operation with respect to the box on the last layer.
        Parameters
        ----------
        ang : float32
            Box orientation angle
        Returns
        -------        
        xr: [Ns,2]: float32
            (y,x) array of coordinates
        """
        # form 1D array of indeces
        xr = cp.indices((self.subregion[ang, 1]-self.subregion[ang, 0],
                         self.subregion[ang, 3]-self.subregion[ang, 2]))
        xr[0] += self.subregion[ang, 0]-self.fgridshape[0]//2
        xr[1] += self.subregion[ang, 2]-self.fgridshape[1]//2
        xr = xr.reshape(2, -1).swapaxes(0, 1)
        # rotate and shift
        xr = util.rotate(xr, self.theta[ang], reverse=True)
        xr[:, 1] -= self.xi_cent[-1]
        # switch to [-1/2,1/2) interval w.r.t. box
        xr /= cp.array(self.boxshape[-1])
        return xr

    def takeshifts(self, k, j):
        n = self.fgridshape[0]//2
        shifts = cp.zeros(
            [self.nangles, self.boxshape.shape[0], 2], dtype='float32')
        for ang in range(self.nangles):
            for l in range(self.boxshape.shape[0]):
                # use any point in the box [-8,-12] - example
                xrm = cp.array(
                    [[-8/self.boxshape[l, 0], -12/self.boxshape[l, 1]]])
                xrm1 = xrm.copy()
                # switch to the first box coordinate system
                xrm = util.rotate(xrm, self.theta[ang])
                xrm += cp.array([[0.5*k, 0.5*j]])
                xrm = util.rotate(xrm, -self.theta[ang])
                xrm[0, 0] *= self.boxshape[l, 0]
                xrm[0, 1] *= self.boxshape[l, 1]
                # find closest point in the first region
                xrm = cp.round(xrm)
                #print(f'{ang},{l},closest point in the first box {xrm=}')
                # return to the current box coordinate system
                xrm[0, 0] /= self.boxshape[l, 0]
                xrm[0, 1] /= self.boxshape[l, 1]
                xrm = util.rotate(xrm, self.theta[ang])
                xrm -= cp.array([[0.5*k, 0.5*j]])
                xrm = util.rotate(xrm, -self.theta[ang])
                # shift in space
                shifts[ang, l] = xrm[0]-xrm1[0]
        return shifts

    def mergecoeffs(self,coeffs):
        for l in range(self.boxshape.shape[0]):
            y = cp.arange(-self.boxshape[l, 0]//2,self.boxshape[l, 0]//2)/self.boxshape[l, 0]
            x = cp.arange(-self.boxshape[l, 1]//2,self.boxshape[l, 1]//2)/self.boxshape[l, 1]
            [y, x] = cp.meshgrid(y, x, indexing='ij')
            yx = cp.array([y.flatten(), x.flatten()]).astype('float32').swapaxes(0, 1)
            # create a big array for all coeffs with coordinates of the first region
            coeffsall = np.zeros([len(coeffs)*len(coeffs[0])*self.boxshape[l, 0], len(
                coeffs)*len(coeffs[0])*self.boxshape[l, 1]], dtype='complex64')# could be done smaller
            for ang in range(self.nangles):                
                # fill the big array
                coeffsall[:] = 0
                for k in range(len(coeffs)):
                    for j in range(len(coeffs[0])):
                        # switch to the first box coordinate system
                        xrm = util.rotate(yx, self.theta[ang])
                        xrm += cp.array([[0.5*k, 0.5*j]])
                        xrm = util.rotate(xrm, -self.theta[ang])
                        xrm[:, 0] *= self.boxshape[l, 0]
                        xrm[:, 1] *= self.boxshape[l, 1]
                        # find closest point in the first region
                        xrm = cp.round(xrm).astype('int32').get()                        
                        coeffsall[coeffsall.shape[0]//2+xrm[:, 0],
                                  coeffsall.shape[1]//2+xrm[:, 1]] += coeffs[k][j][l][ang].flatten()
                # broadcast from the big array                                  
                for k in range(len(coeffs)):
                    for j in range(len(coeffs[0])):
                        xrm = util.rotate(yx, self.theta[ang])
                        # switch to the first box coordinate system                        
                        xrm += cp.array([[0.5*k, 0.5*j]])
                        xrm = util.rotate(xrm, -self.theta[ang])
                        xrm[:, 0] *= self.boxshape[l, 0]
                        xrm[:, 1] *= self.boxshape[l, 1]
                        # find closest point in the first region
                        xrm = cp.round(xrm).astype('int32').get()
                        coeffs[k][j][l][ang] = coeffsall[coeffsall.shape[0]//2+xrm[:, 0],coeffsall.shape[1]//2+xrm[:, 1]].reshape(self.boxshape[l]) 
        return coeffs                

    def fwdmany(self, f):
        n = self.fgridshape[0]//2
        coeffs = [[0 for x in range(f.shape[1]//n)]
                  for x in range(f.shape[0]//n)]
        # process data by splitting into parts of the size nxn
        for k in range(f.shape[0]//n):
            for j in range(f.shape[1]//n):
                print(f'Processing region ({k},{j})')
                # take shifts for merging grids
                shifts = self.takeshifts(k, j)
                # decompostion
                coeffs[k][j] = self.fwd(f[k*n:(k+1)*n, j*n:(j+1)*n], shifts)
        coeffs = self.mergecoeffs(coeffs)
        return coeffs

    def adjmany(self, coeffs):
        n = self.fgridshape[0]//2
        f = np.zeros([len(coeffs)*n, len(coeffs[0])*n], dtype='complex64')
        for k in range(f.shape[0]//n):
            for j in range(f.shape[1]//n):
                print(f'Processing region ({k},{j})')
                # take shifts for merging grids
                shifts = self.takeshifts(k, j)
                # decompostion
                f[k*n:(k+1)*n, j*n:(j+1)*n] = self.adj(coeffs[k][j], shifts)
        return f

    def fwd(self, f, shifts=None):
        """Forward operator for GWP decomposition
        Parameters
        ----------
        f : [N,N] complex64
            2D function in the space domain
        Returns
        -------
        coeffs : [K](Nangles,L3,L0,L1) complex64
            Decomposition coefficients for box levels 0:K, angles 0:Nangles, 
            defined on box grids with sizes [L3[k],L0[k],L1[k]], k=0:K
        """
        print('fwd transform')
        # 1) Compensate for the USFFT kernel function in the space domain and apply 2D FFT
        F = self.U.compfft(cp.array(f))

        # allocate memory for coefficeints
        coeffs = [None]*self.K
        for k in range(self.K):
            coeffs[k] = np.zeros(
                [self.nangles, *self.boxshape[k]], dtype='complex64')

        # loop over box orientations
        for ang in range(0, self.nangles):
            print('angle', ang)
            # 2) Interpolation to the local box grid.
            # Gathering operation from the global to local grid.
            # extract ids of the global spectrum subregion contatining the box
            [idsy, idsx] = self.subregion_ids(ang)
            # calculate box coordinates in the space domain
            xr = self.coordinates_fwd(ang)
            # gather values to the box grid
            g = self.U.gather(F[cp.ix_(idsy, idsx)], xr, F.shape)

            # 3) IFFTs on each box.
            # find coefficients on each box
            for k in range(self.K):
                # broadcast values to smaller boxes, multiply by the gwp kernel function
                fcoeffs = self.gwpf[k]*g[self.inds[k]]
                fcoeffs = fcoeffs.reshape(self.boxshape[k])
                # shift wrt xi_cent
                fcoeffs = util.checkerboard(cp.fft.ifftn(
                    util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                fcoeffs *= cp.exp(-1j*self.phase[k])

                if(shifts is not None):
                    # shift wrt region in rectangular grid
                    fcoeffs = util.checkerboard(cp.fft.fftn(
                        util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                    [py, px] = cp.meshgrid(
                        self.phasemanyy[k]*shifts[ang, k, 0], self.phasemanyx[k]*shifts[ang, k, 1], indexing='ij')
                    fcoeffs *= cp.exp(-1j*(px+py))
                    fcoeffs = util.checkerboard(cp.fft.ifftn(
                        util.checkerboard(fcoeffs), norm='ortho'), inverse=True)

                # normalize
                fcoeffs /= (np.prod(self.boxshape[-1]))
                coeffs[k][ang] = fcoeffs.get()

        # free cupy pools
        # mempool.free_all_blocks()
        # pinned_mempool.free_all_blocks()
        return coeffs

    def adj(self, coeffs, shifts=None):
        """Adjoint operator for GWP decomposition
        Parameters
        ----------
        coeffs : [K](Nangles,L0,L1) complex64
            Decomposition coefficients for box levels 0:K, angles 0:Nangles, 
            defined box grids with sizes [L0[k],L1[k]], k=0:K        
        Returns
        -------
        f : [N,N] complex64
            2D function in the space domain        
        """
        print('adj transform')
        # build spectrum by using gwp coefficients
        F = cp.zeros(self.fgridshape, dtype="complex64")
        # loop over box orientations
        for ang in range(0, self.nangles):
            print('angle', ang)
            g = cp.zeros(int(np.prod(self.boxshape[-1])), dtype='complex64')
            for k in range(self.K):
                fcoeffs = cp.array(coeffs[k][ang])
                # normalize
                fcoeffs /= (np.prod(self.boxshape[-1]))
                if(shifts is not None):
                    # shift wrt region in rectangular grid
                    fcoeffs = util.checkerboard(cp.fft.fftn(
                        util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                    [py, px] = cp.meshgrid(
                        self.phasemanyy[k]*shifts[ang, k, 0], self.phasemanyx[k]*shifts[ang, k, 1], indexing='ij')
                    fcoeffs *= cp.exp(1j*(px+py))
                    fcoeffs = util.checkerboard(cp.fft.ifftn(
                        util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                # shift wrt xi_cent
                fcoeffs *= cp.exp(1j*self.phase[k])
                fcoeffs = util.checkerboard(cp.fft.fftn(
                    util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                # broadcast values to smaller boxes, multiply by the gwp kernel function
                g[self.inds[k]] += self.gwpf[k]*fcoeffs.flatten()
            g = g.reshape(self.boxshape[-1])
            # 2) Interpolation to the global grid
            # Conventional scattering operation from the global to local grid is replaced
            # by an equivalent gathering operation.
            # calculate global grid coordinates in the space domain
            xr = self.coordinates_adj(ang)
            # extract ids of the global spectrum subregion contatining the box
            [idsy, idsx] = self.subregion_ids(ang)
            # gathering to the global grid
            F[cp.ix_(idsy, idsx)] += self.U.gather(g, xr,
                                                   g.shape).reshape(len(idsy), len(idsx))
           # util.mplt(F)
        # 3) apply 2D IFFT, compensate for the USFFT kernel function in the space domain
        f = self.U.ifftcomp(
            F.reshape(self.fgridshape)).get().astype('complex64')

        # free cupy pools
        # mempool.free_all_blocks()
        # pinned_mempool.free_all_blocks()
        return f
