import warnings as warnings

import numpy as np

from scipy.signal import argrelmin, argrelmax

from pyscf.data.radii import BRAGG
from pyscf.lib.parameters import BOHR
from pyscf.dft.numint import eval_ao, eval_rho

from pyAPOST.utils import eval_dens

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

# def _eval_dens(mol, dm, coords):
#     """Utility to evaluate density on a grid of points. Equivalent to pyscf/dft/numint.py eval_rho()"""
#     dens_ao = eval_ao(mol, coords)
#     dens = np.zeros((dens_ao.shape[0]))
#     for mu in range(dens_ao.shape[1]):
#         for nu in range(dens_ao.shape[1]):
#             dens += dm[mu, nu] * dens_ao.T[mu, :] * dens_ao.T[nu, :]
#
#     return dens

# todo: code the error function erf(x/sqrt(1-x**2))

def _f_k(mu_cord, k=4):
    for k in range(k):
        mu_cord = (3/2)*mu_cord - (1/2)*mu_cord**3
    return mu_cord

def mu_cord(grid_cords, atm_coords):
    "WIP: Function to obtain mu_ab for Becke and TFVC schemes"

    # 1. Calculate mu_ab (both becke and TFVC)
    mu_ab = np.linalg.norm(atm_coords[:, None, :] - grid_cords , axis=-1)
    # Fixed bug info: the sign was wrong because we were putting the wrong order of None axis and then the following transpose didn't match
    # mu_ab = mu_ab[None, :] - mu_ab[:, None, :] # Wrong None axis for transpose [2,0,1]
    mu_ab = mu_ab[:, None, :] - mu_ab[None, :]
    # Compute atm_coords distance R_AB. symmetric array Natm, Natm
    r_ab = np.linalg.norm(atm_coords[:, None, :] - atm_coords[None, :, :], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu_ab = mu_ab / r_ab[:, :, None]  # Expecting Nans for A=B
    # Option: we could keep the natom_natom_mu
    mu_ab = mu_ab.transpose([2, 0, 1]) # transpose to have mu_ab values as first dimension
    return mu_ab
#todo: use atm_radii when passed
def becke_w(grid_cords, atm_coords, atnums, atm_grid_idx, atm_radii=None, log='info'):
    """WIP: Becke weights for AIM."""

    print('DEV: Becke weights for AIM')
    print('grid_cords: ', grid_cords.shape)
    print('atm_coords: ', atm_coords.shape)

    if log == 'debug':
        logger.setLevel(level=logging.DEBUG)

    mu_ab = mu_cord(grid_cords, atm_coords)
    # # 2. get alpha_ab with Bragg-Slater radii values. Stored values are in A.U (Bohr)
    # From Becke paper: H = 0.35A/0.66 Bohr
    # v_ab coordinate with alpha_ab and Bragg slater radii (Particular to becke)
    idx_radii = []
    for z in atnums:
        if z != 1:
            idx_radii.append(z -1)
        else:
            idx_radii.append(z)
    logger.info(f'Idx Bragg radi (Python index): {idx_radii}',) # Hydrogen is stored at the second index
    atm_radii = BRAGG[idx_radii]
    logger.info(f'Selected Bragg radi: {atm_radii}', )
    # DEV: u_ab is equivalent to the 2 lines below
    # chi = atm_radii[:, None] / atm_radii[None, :]
    # u_ab_1 = (chi -1) / (chi+1)
    u_ab = (atm_radii[:, None] - atm_radii) / (atm_radii[:, None] + atm_radii)
    # Below codi pyscf
    # chi = atm_radii[:, None] / atm_radii[None, :]
    # u_ab = 0.25 * (chi.T - chi)
    # Note: Grid uses cutoff = 0.45
    alpha = u_ab / (np.power(u_ab, 2) - 1)
    logger.info('Calculated alpha: ')
    logger.info(alpha.shape)
    # print('alpha: ', alpha.shape)
    alpha[alpha < -0.5] = -0.5
    alpha[alpha > 0.5] = 0.5
    logger.info('After cutoff 0.5 ')
    # logger.info(alpha)
    v_ab = mu_ab + alpha*(1 -np.power(mu_ab,2))
    # f_k function (same with TFVC)
    f_k = _f_k(v_ab)
    # s_a_k (same with TFVC)
    s_a = 0.5 * (1 - f_k)
    # Substitute Nans by 1. Spurious just to keep the number of dimensions
    s_a[np.isnan(s_a)] = 1
    pa = np.prod(s_a, axis=-1)
    # transfer weights from the corresponding atom grid point
    w = np.zeros(pa.shape[0])
    for ia in range(len(atnums)):
        w[atm_grid_idx==ia] += pa[atm_grid_idx==ia, ia] / np.sum(pa[atm_grid_idx==ia, :], axis=1)

    return w


def tfvc_w(mol, dm, grid_cords, atm_coords, atnums, atm_grid_idx, atm_radii=None):
    """WIP: TFVC weights for AIM. This function creates the grid exactly as APOST3D code sources/wat.f subroutine khi"""


    natom = atm_coords.shape[0]

    # 1. compute mu_ab and get brag radi
    mu_ab = mu_cord(grid_cords, atm_coords)
    # 2. Start computing atomic radii (chi)
    logger.info('  Using density along atom pairs to set atomic radii.')
    # 2.1 Compute xdist to know what atom pairs to do
    mpoint_coords = (atm_coords[:, None, :] + atm_coords[None, :, :])/2
    dist_mp = np.linalg.norm(atm_coords[:, None, :] - atm_coords[None, :, :], axis=-1)/2
    xdist = np.linalg.norm(mpoint_coords[:, :, None, :] - atm_coords[None, None, :, :], axis=-1)

    chi = np.zeros((atm_coords.shape[0], atm_coords.shape[0]))

    # Only lower triangle
    # Compute differences between atom coords components
    rij_mat = atm_coords[:, None, :] - atm_coords[None, :, :]
    # Filter
    for i in range(natom):
        for j in range(i+1, natom):
            if np.any(np.round(xdist[i,j],4) < np.round(dist_mp[i,j],4)):
                chi[i,j] = 1
                chi[j,i] = 1
            else:
                # Todo: Complete cases: 2.Deal with pseudopotentials
                # 3. Compute atomic radii as min density along the bond
                # Create a radial grid with a setup step.
                # step*dfloat(k-1)*rijx : percentage of the rijx we are moving from ri
                rij = rij_mat[j,i]

                # Note: For now hard coded div = 50
                ndiv = 50 #APOST3d value
                rad_dens_x = np.zeros(ndiv)
                rad_dens_y = np.zeros(ndiv)
                rad_dens_z = np.zeros(ndiv)
                step = 1/ndiv

                for k in range(ndiv):
                    rad_dens_x[k] = atm_coords[i, 0] + (step * k)*rij[0]
                    rad_dens_y[k] = atm_coords[i, 1] + (step * k)*rij[1]
                    rad_dens_z[k] = atm_coords[i, 2] + (step * k)*rij[2]

                # Combine coord grids into a radial grid
                rad_dens_grid = np.column_stack((rad_dens_x, rad_dens_y, rad_dens_z))
                # In APOST3d there is a function that if a point is passed it gives you back the density. Here we should call eval_ao/eval_rho
                # Implemented _eval_dens function to double-check we are using eval_rho correctly
                radial_dens_ao = eval_ao(mol, rad_dens_grid)
                radial_dens = eval_rho(mol, radial_dens_ao, dm)
                # check comparing with calculating ourselves the density
                check_dens_radial = eval_dens(mol, dm, rad_dens_grid)
                assert np.allclose(radial_dens, check_dens_radial)
                n_min_idx = argrelmin(radial_dens, order=1)[0]
                n_max_idx = argrelmax(radial_dens, order=1)[0]
                logger.info(f'  Max : {n_max_idx} , Min : {n_min_idx}')
                maxf = False
                if n_min_idx.shape[0] > 0:
                    if n_min_idx.shape[0] == 1 and n_max_idx.shape[0] == 0:
                        min_index = n_min_idx[0]
                        logger.info(f'  Only 1 minima found : {n_min_idx}')
                        sub_index = min_index
                    elif n_min_idx.shape[0] == 2 and (n_max_idx.shape[0] == 1 or n_max_idx.shape[0] == 3):
                        # Note: For now only dealing with 1 maximum between 2 minims
                        logger.info('  Inconsistent number of extrema')
                        # Non-nuclear attractor maxima
                        if n_max_idx.shape[0] == 3:
                            n_max_idx = n_max_idx[1]
                        diff = n_min_idx - n_max_idx
                        if diff[0] < 0 < diff[1]:
                            logger.info(f'  The central maxima selected as dividing point: {n_max_idx}')
                            sub_index = n_max_idx
                            maxf = True
                        else:
                            raise TypeError("Couldn't sort the extrema closer to the midpoint")
                    elif n_min_idx.shape[0] >= 1 and n_max_idx:
                        logger.info('  Inconsistent number of extrema')
                        # Selecting one closest to the midpoint
                        kc = np.argmin(np.linalg.norm(mpoint_coords[i,j] - rad_dens_grid[n_min_idx], axis=-1))
                        min_index = n_min_idx[kc]
                        logger.info(f'  Selecting minimum closest to midpoint : {min_index}')
                        sub_index = min_index
                    else:
                        raise TypeError("Couldn't sort the extrema closer to the midpoint")


                # Second round of radial dens evaluation
                nsubdiv = 40 #APOST3d value

                rad_dens_x_2 = np.zeros(nsubdiv)
                rad_dens_y_2 = np.zeros(nsubdiv)
                rad_dens_z_2 = np.zeros(nsubdiv)

                step2 = 2 / (ndiv*nsubdiv)

                for k in range(nsubdiv):
                    rad_dens_x_2[k] = atm_coords[i, 0] + (step*(sub_index-2+1) + step2*k)*rij[0]
                    rad_dens_y_2[k] = atm_coords[i, 1] + (step*(sub_index-2+1) + step2*k)*rij[1]
                    rad_dens_z_2[k] = atm_coords[i, 2] + (step*(sub_index-2+1) + step2*k)*rij[2]
                rad_dens_grid_2 = np.column_stack((rad_dens_x_2, rad_dens_y_2, rad_dens_z_2))

                # Get density on the radial points
                radial_dens_ao_2 = eval_ao(mol, rad_dens_grid_2)
                radial_dens_2 = eval_rho(mol, radial_dens_ao_2, dm)
                # Searching for min/max based on extrema type sorted above
                if maxf:
                    index_2 = np.argmax(radial_dens_2)
                else:
                    index_2 = np.argmin(radial_dens_2)

                # ri = (step * dfloat(kc - 2) + substep * dfloat(kc1 - 1))
                ri = step*(sub_index-2+1) + step2*index_2
                rj = 1 - ri
                logger.debug(f'ri: {ri}, rj: {rj}')
                chi[j,i] = rj/ri
                chi[i,j] = ri/rj

    for i in range(chi.shape[0]-1):
        for j in range(i+1,chi.shape[0]):
            # print(i,j)
            logger.info(f'  Atom pair {i+1} and {j+1} : {chi[i,j] : .4f}')

    v_ab = (1 + mu_ab - chi*(1-mu_ab))/(1 + mu_ab + chi*(1-mu_ab))
    # f_k function (same with TFVC)
    f_k = _f_k(v_ab)
    # s_a_k (same with TFVC)
    s_a = 0.5 * (1 - f_k)
    # Substitute Nans by 1. Spurious just to keep the number of dimensions
    s_a[np.isnan(s_a)] = 1
    pa = np.prod(s_a, axis=-1)
    # transfer weights from the corresponding atom grid point
    w = np.zeros(pa.shape[0])
    for ia in range(len(atnums)):
        w[atm_grid_idx == ia] += pa[atm_grid_idx == ia, ia] / np.sum(pa[atm_grid_idx == ia, :], axis=1)
    return w
