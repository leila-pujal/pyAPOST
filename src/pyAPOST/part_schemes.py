import numpy as np

from pyscf.data.radii import BRAGG
from pyscf.lib.parameters import BOHR

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

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
    mu_ab = mu_ab / r_ab[:, :, None]  # Expecting Nans for A=B
    # Option: we could keep the natom_natom_mu
    mu_ab = mu_ab.transpose([2, 0, 1]) # transpose to have mu_ab values as first dimension
    return mu_ab

def becke_w(grid_cords, atm_coords, atnums, atm_grid_idx):
    """WIP: Becke weights for AIM."""

    print('DEV: Becke weights for AIM')
    print('grid_cords: ', grid_cords.shape)
    print('atm_coords: ', atm_coords.shape)

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
    logger.debug('Idx Bragg radi: ', idx_radii)
    atm_radii = BRAGG[idx_radii]
    logger.debug('Selected Bragg radi: ', atm_radii)
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
