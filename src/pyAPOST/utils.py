import numpy as np

from pyscf.dft.numint import eval_ao, eval_rho


# Note: if basis set big, for memory efficiency might be better to access already evaluated AO on the grid
# insteadof calling eval_ao every time
def eval_dens(mol, dm, coords):
    """Utility to evaluate density on a grid of points. Equivalent to pyscf/dft/numint.py eval_rho()"""
    dens_ao = eval_ao(mol, coords)
    dens = np.zeros((dens_ao.shape[0]))
    for mu in range(dens_ao.shape[1]):
        for nu in range(dens_ao.shape[1]):
            dens += dm[mu, nu] * dens_ao.T[mu, :] * dens_ao.T[nu, :]

    return dens