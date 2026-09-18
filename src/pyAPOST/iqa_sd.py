# IQA module Single Determinant
import copy as copy
import time as time

import multiprocessing as mp
from multiprocessing import set_start_method, get_start_method
import numba as numba
from numba import njit

import numpy as np
import logging

from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.libxc import eval_xc, is_hybrid_xc, hybrid_coeff

from pyAPOST.energy_analyt import compute_energy_terms
from pyAPOST.utils import out_summary
from pyAPOST.bond_order import bo_dens

@njit(nopython=True)
def _coul_integral(id_p1, p1, rho1, rho2, atgrid_idx_b, a_grid_coords, b_grid_coords, wq, aim_w1, aim_w2, rij_v=1e-5):
    """WIP: coulomb integral from rho"""
    ther_rij = rij_v
    if id_p1 == 0:
        print('Threshold rij', ther_rij)
    coul_dens = np.zeros((rho2[atgrid_idx_b].shape[0]))
    coul_dens += rho1[p1] * rho2[atgrid_idx_b]
    rij = a_grid_coords[id_p1, :] - b_grid_coords
    rij = np.sqrt(np.sum(np.square(rij), axis=-1))
    if id_p1 == 0:
        print('Shape rij', rij.shape)
        print('Filter rij', rij[rij<ther_rij].shape)
    coul_dens = coul_dens / rij
    # filter values of the points wit rij < 1e-3. If not filtered they add spurious big values
    coul_dens[rij < ther_rij] = 0
    # Now integration starts. Get aim weights for p1 atgridA and all points in atgridB
    coul_b = np.sum(coul_dens * wq[atgrid_idx_b] * aim_w2[atgrid_idx_b])  # integrated for atB
    coul = coul_b * wq[p1] * aim_w1[p1]

    return coul, rij[rij < ther_rij].shape[0]

# @njit(nopython=True, parallel=True)
@njit(nopython=True)
def _exchange_integral(id_p1, p1, mo_grid1, mo_grid2, atgrid_idx_b, a_grid_coords, b_grid_coords, wq, aim_w1, aim_w2, rij_v=1e-10, occs=None):
    """ WIP: Auxiliary function for parallel calculation of 2e integrals"""
    # print('inside _two_elect_integral')
    # Loop only for occupied orbitals
    ther_rij = rij_v
    if id_p1 == 0:
        print('Threshold rij', ther_rij)
    # coul_dens = np.zeros((mo_grid2[0, atgrid_idx_b].shape[0]))
    exch_dens = np.zeros((mo_grid2[0, atgrid_idx_b].shape[0]))
    # Todo: Maybe we pass tuples for mo_grid1/2 we could do alpha and beta in the same call
    for i in range(mo_grid1.shape[0]):
        for j in range(mo_grid2.shape[0]):
            # If occs included then 1/4
            # exch_dens += occs[i] * mo_grid1[i, p1] * mo_grid1[j, p1] * occs[j] * mo_grid2[j, atgrid_idx_b] * mo_grid2[i, atgrid_idx_b]
            exch_dens += mo_grid1[i, p1] * mo_grid2[j, atgrid_idx_b]* mo_grid2[i, atgrid_idx_b] * mo_grid1[j, p1]
    # Compute rij between i point index (p1) and atom b atgrid points
    # rij = np.linalg.norm(a_grid_coords[id_p1, :] - b_grid_coords, axis=-1)
    rij = a_grid_coords[id_p1, :] - b_grid_coords
    rij = np.sqrt(np.sum(np.square(rij), axis=-1))
    if id_p1 == 0:
        print('Shape rij', rij.shape)
        print('Filter rij', rij[rij<ther_rij].shape)


    # coul_dens = coul_dens / rij
    exch_dens = exch_dens / rij
    # filter values of the points wit rij < 1e-3. If not filtered they add spurious big values
    # coul_dens[rij<ther_rij] = 0
    exch_dens[rij<ther_rij] = 0
    # order is 1e-15
    # rij_small += rij[rij<1e-9].shape[0]
    # Now integration starts. Get aim weights for p1 atgridA and all points in atgridB
    # coul_b = np.sum(coul_dens* qw[atgrid_idx_b] * aim_w2[atgrid_idx_b]) # integrated for atB
    exch_b = np.sum(exch_dens* wq[atgrid_idx_b] * aim_w2[atgrid_idx_b]) # integrated for atB
    exch = exch_b * wq[p1] * aim_w1[p1]

    return exch, rij[rij<ther_rij].shape[0]




def nn_energy(at_nums, at_coords):
    r"""Compute nuclear-nuclear repulsion energy.

    math::
        \sum_{A>B} \frac{Z_{A}Z_{B}}{R_{A}-R_{B}}

    Where Z_{A} and Z_{B} are nuclear charges.

    Returns
    -------
    nn_e: np.array()
        Total value for nuclear-nuclear repulsion energy
    """
    # Compute Nucleus-Nucleus repulsion
    logging.info("CALCULATING NUCLEAR REPULSION ENERGY")
    rab = np.triu(np.linalg.norm(at_coords[:, None] - at_coords, axis=-1))
    atomic_charges = np.triu(at_nums[:, None] * at_nums)[np.where(rab > 0)]
    rab = rab[rab > 0]
    print('VERBOSE NN')
    print(atomic_charges / rab)
    nn_e = np.sum(atomic_charges / rab)
    print("NN ENERGY: ", nn_e)
    print()

    return nn_e

def en_energy(at_nums, at_coords, grid_coords, dens, int_w, aim_w, atgrid_idx):
    r"""Compute IQA's electron-nuclear attraction energy.

     math::
        \sum_{A, B}\int_{A} \rho(r_{1}) \frac{Z_{B}}{r_{1}-R_{B}} dr_{1}

    Where \rho(r) is the electron density and Z_{B} are nuclear charges.

    Parameters
    ----------
    share_factor: float
        How much of the energy is given to atom i when its density interacts with other nuclei j(x)
        and how much energy is given to atom i when its nucleus interacts with other atomic densities j(1-x)

    Returns
    -------
    en_t: np.array()
        Total value for electron-nuclear attraction energy
    en_atomic_matrix: np.array()
        Decomposition en energy array
    en_cond_en: np.array(natoms)
        Atomic condensed electron-nuclear attraction energies

    """


    # Compute Electron-Nucleus attraction energy
    logging.info("CALCULATING ELECTRON-NUCLEI ATTRACTION ENERGY")
    natoms = at_nums.shape[0]



    rij = np.linalg.norm(at_coords[:, None, :] - grid_coords, axis=-1)
    print('rij', rij.shape)
    en_values = np.sum((-at_nums[:, None] * (dens / rij)), axis=0)
    t_en = np.sum(en_values * int_w * aim_w)
    print("TOTAL EN ENERGY: ", t_en)

    # condense
    # Getting atomic/fragment weights
    print(dens.shape)
    print(dens[atgrid_idx == 0].shape)

    en_atomic_matrix = np.zeros((at_nums.size, at_nums.size))
    for i in range(natoms):
        at_dens = dens[atgrid_idx == i] * aim_w[atgrid_idx == i]
        for j in range(natoms):
             en_at = -at_nums[j] * at_dens / rij[j][atgrid_idx == i]
             en_at = np.sum(en_at * int_w[atgrid_idx == i])
             en_atomic_matrix[i][j] = en_at
    # en_atomic_matrix_dic = copy.copy(en_atomic_matrix)
    # share_matrix
    share_factor = 0.5
    share_matrix = np.zeros((natoms, natoms, natoms))
    for i in range(natoms):
        share_matrix[i, i, :] = share_factor
        share_matrix[i, :, i] = 1 - share_factor
        share_matrix[i, i, i] = 1

    en_cond_en = np.sum(en_atomic_matrix[None, :, :] * share_matrix, axis=(1, 2))
    assert np.allclose(np.sum(en_cond_en), t_en, atol=1e-6)

    return t_en, en_atomic_matrix, en_cond_en

def kin_energy(at_nums, at_coords, mol, mo_coeff, occs, grid_coords, wq, aim_w, atgrid_idx ):
    r"""Compute IQA's kinetic energy.

        Returns
        -------
        t_kin: np.array()
            Total value for kinetic energy
        at_kin: np.array(natoms)
            Atomic kinetic energies.

        """
    print('DEV: COMPUTE GRADIENTS')
    # 1. Compute positive def kinetic
    # 2. Compute alpha part

    # Todo: Did we mention to compute the kinetic in molecular?
    # Get the gradients.
    eval_d_ao = eval_ao(mol, grid_coords, deriv=1)
    # eval_d_ao = eval_ao(mol, grid_coords)
    print(eval_d_ao.shape)
    # Build MO gradients
    # 100/100, 010/010, 001/001
    eval_d_ao = eval_d_ao.transpose(0, 2, 1)
    # Start with  restricted hf closed shell so only one set of orbitals
    eval_d_mo = np.einsum('dmn, mi -> din', eval_d_ao, mo_coeff)

    kin = np.zeros(grid_coords.shape[0])
    # Below loop is the same as:
    # dens_100 = np.trace(np.dot(dm, (eval_d_ao[1, :, None, :] * eval_d_ao[1, None, :, :])))
    # dens_010 = np.trace(np.dot(dm, (eval_d_ao[2, :, None, :] * eval_d_ao[2, None, :, :])))
    # dens_001 = np.trace(np.dot(dm, (eval_d_ao[3, :, None, :] * eval_d_ao[3, None, :, :])))
    for i_dv in range(1,4):
        kin += np.trace(np.dot(occs, (eval_d_mo[i_dv, :, None, :] * eval_d_mo[i_dv, None, :, :])))


    # for i_dv in range(1,4):
    #     kin += np.trace(np.dot(dm, (eval_d_ao[i_dv, :, None, :] * eval_d_ao[i_dv, None, :, :])))
    kin *= 0.5
    kin_t = np.sum(kin * wq * aim_w)
    print(f'TOTAL INTEGRATED KINETIC ENERGY = {kin_t: .6f}', )

    # To get the general kinetic energy now compute the laplacian
    # Investigate: PySCF does not return the expected derivative components for deriv=2
    # Expected 6 but got 10 as if I was requesting the 3rd order derivative
    # eval_l_ao = eval_ao(mol, grid_coords, deriv=2)

    # Getting atomic contributions
    at_kin = np.zeros((len(at_nums)))
    for at in range(len(at_nums)):
        at_k = np.sum(kin[atgrid_idx == at] * wq[atgrid_idx == at] * aim_w[atgrid_idx == at])
        at_kin[at] = at_k

    print('DEV ATOMIC VALUES FOR KINETIC ENERGY')
    print(at_kin)
    print(f'CHECK: SUM OF ATOMIC VALUES : {np.sum(at_kin): .6f}')

    return kin_t, at_kin


def ee_hf_energy(at_nums, at_coords, mol, dm, grid_coords, eval_ao_grid, mo_coeff, occ, wq, aim_w, atgrid_idx, coul=True, x=True):
    r"""Compute Hartree Fock electron-electron interaction energy."""

    # Follow libcint approach, loop for each charge
    # Compute molecular orbitals numerically
    # Note: mo_coeff filtered by occ
    # if isinstance(mo_coeff, tuple) and len(mo_coeff) == 2:
    #     dens_grid = np.zeros((eval_ao_grid.shape[0]))
    #     mo_grid = np.zeros((mo_coeff[0].shape[1] +mo_coeff[1].shape[1], eval_ao_grid.shape[0]))
    #     mo_grid[:occ[0].shape[0]] = np.einsum('pa,ai->ip ', eval_ao_grid, mo_coeff[0])
    #     mo_grid[occ[0].shape[0]:] = np.einsum('pa,ai->ip ', eval_ao_grid, mo_coeff[1])
    #     for s in range(2):
    #         for i in range(mo_coeff[s].shape[1]):
    #             dens_grid += mo_grid[i, :] * mo_grid[i, :] * occ[s][i]
    #
    #     total_elect_int = np.sum(dens_grid * qw *aim_w)
    # elif isinstance(mo_coeff, np.ndarray):

    # Initialize returned variables to be consistent
    t_coul, t_exch, at_coul, at_exch = None, None, None, None


    mo_grid = np.einsum('pa,ai->ip ', eval_ao_grid, mo_coeff)
    dens_grid = np.zeros((mo_grid.shape[1]))
    for i in range(mo_grid.shape[0]):
        dens_grid += mo_grid[i, :] * mo_grid[i, :] * occ[i]

    total_elect_int = np.sum(dens_grid * wq * aim_w)


    print(f'  Check integrated elect: {total_elect_int: .5f}\n')

    # For now explicit loop but if it works look into applying a funct to a list of points
    chunk_size = 250000000 // (mol.nao_nr() ** 2)
    print('chunk_size', chunk_size)
    istart = 0

    while istart < grid_coords.shape[0]:
        iend = min(istart + chunk_size, grid_coords.shape[0])
        chunk_pc = np.zeros((mol.nao_nr(), mol.nao_nr(), iend))
        chunk_dens = dens_grid[istart:iend]
        for ip, p in enumerate(grid_coords[istart:iend, :]):
            with mol.with_rinv_origin(p):
                pc_int = mol.intor('int1e_rinv')
                # Note: Exchange negative introduced with the one charge
                pc_int *= -1
                chunk_pc[:, :, ip] = pc_int


        if coul:
            t_coul_num = np.zeros((grid_coords.shape[0]))
            pc_rho =  np.trace(np.tensordot(dm, (chunk_pc * chunk_dens), axes=(1, 0)))
            t_coul_num[istart:iend] = -0.5 * pc_rho
        if x:
            t_exch_num = np.zeros((grid_coords.shape[0]))
            t1 = np.einsum("abn,ai->ibn", chunk_pc, mo_coeff)
            t1 = np.einsum("ibn,bj->ijn", t1, mo_coeff)
            exch_chunk = np.einsum("ijn,in,jn->n", t1, mo_grid, mo_grid)
            print('occ', occ)
            if occ[0] == 2:
                t_exch_num[istart:iend] =  exch_chunk
            elif occ[0] == 1:
                t_exch_num[istart:iend] = 0.5 * exch_chunk
            del t1


        istart = iend



    if coul:
        t_coul = np.sum(t_coul_num * wq * aim_w)
        print(f'TOTAL COULOMB {t_coul: .5f}')
        at_coul = np.zeros((at_nums.size))
        for at in range(len(at_nums)):
            at_c = np.sum(t_coul_num[atgrid_idx == at] * wq[atgrid_idx == at] * aim_w[atgrid_idx == at])
            at_coul[at] = at_c
        print('DEV ATOMIC VALUES for semianalitic ee terms')
        print(at_coul)
        print('CHECK: SUM OF ATOMIC VALUES')
        print(np.sum(at_coul))


    if x:
        t_exch = np.sum(t_exch_num * wq * aim_w)
        print(f'TOTAL EXCHANGE {t_exch: .5f}')
        at_exch = np.zeros((at_nums.size))
        for at in range(len(at_nums)):
            at_x = np.sum(t_exch_num[atgrid_idx == at] * wq[atgrid_idx == at] * aim_w[atgrid_idx == at])
            at_exch[at] = at_x
        print('DEV ATOMIC VALUES for semianalitic ee terms')
        print(at_exch)
        print('CHECK: SUM OF ATOMIC VALUES')
        print(np.sum(at_exch))


    # Compute atomic values

    # assert 5 == 7

    return t_coul, at_coul, t_exch, at_exch


def ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, wq, aim_w, atgrid_idx, spin=0):
    """"WIP: DFT energy to obtain AA contributions

    Parameters:
        rho: Has to include all elements of the density needed to call Libxc and compute the energy density
        For Restricted calculations:
            LDA: np.array (1,Npoints)  --> Density
            GGA: np.array (4, Npoints) ---> Density, dx, dy, dz
        For unrestricted calculations:
            LDA tuple (alpha, beta):
                alpha: np.array (1,Npoints)  --> Density
                beta: np.array (1,Npoints)  --> Density
            GGA: tuple (alpha, beta):
                alpha: np.array (4, Npoints) ---> Density, dx, dy, dz
                beta: np.array (4, Npoints) ---> Density, dx, dy, dz

    """



    print('DEV dft exchange-correlation')

    # 1. get the functional string
    # 2. call eval_xc to get the energy density
    # 3. Check for an example of each type if we compute the correct total energy density integrating
    # correctly

    print('XC CODE AND TYPE :', xc_code, xc_type)
    # prepare rho to pass to eval_xc. rho contains necessary components e.g. dens, grad
    # todo: we need to sort xc, x and c individually. Check if PySCF allows for them to be defined separately
    # rho includes necessary density quantities for each type of functional
    # LDA: density
    # GGA: density, grad_x, grad_y, grad_z
    # There are several functions whose goal is to evaluate the DFT exchange-correlation energy density but ultimately
    # all of them end up on the libxc module interface calling eval_xc which calls the protected method _eval_xc (calls the C++ routine)
    # eval_xc returns ex(energy density), vxc(first derivatives of the xc energy with respect density, gradient, lap and kin, fxc(second derivs), kxc(third derivs)
    # shape of ex, vxc, fxc and kxc different when spin=0 or 1
    r = eval_xc(xc_code, rho, spin=spin)
    xc_e_dens = r[0]

    # Todo: Double check if we always pass (,Npoints) or we could pass (1, Npoints)
    # Atomic contributions
    at_dft = np.zeros((at_nums.size))
    # Unrestriced
    if isinstance(rho, tuple):
        print('unrestricted')
        if len(rho[0].shape) == 1:
            # LDA
            print('lda')
            xc_e = np.sum(xc_e_dens * (rho[0] + rho[1]) * wq * aim_w)
            for at in range(len(at_nums)):
                at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] *(rho[0] + rho[1])[atgrid_idx == at]* wq[atgrid_idx == at] * aim_w[atgrid_idx == at])

        else:
            # GGA
            print('gga')
            xc_e = np.sum(xc_e_dens * (rho[0][0] + rho[1][0]) * wq * aim_w)
            for at in range(len(at_nums)):
                at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] *(rho[0][0] + rho[1][0])[atgrid_idx == at]* wq[atgrid_idx == at] * aim_w[atgrid_idx == at])
    # Restricted
    elif isinstance(rho, np.ndarray):
        print('restricted')
        if len(rho.shape) == 1:
            # LDA
            print('lda')
            xc_e = np.sum(xc_e_dens * rho *  wq * aim_w)
            for at in range(len(at_nums)):
                at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] *rho[atgrid_idx == at]* wq[atgrid_idx == at] * aim_w[atgrid_idx == at])

        else:
            print('gga')
            xc_e = np.sum(xc_e_dens * rho[0] *  wq * aim_w)
            for at in range(len(at_nums)):
                at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] * rho[0][atgrid_idx == at] * wq[atgrid_idx == at] * aim_w[atgrid_idx == at])


    print('TOTAL DFT EXCHANGE-CORRELATION ENERGY : ', xc_e)
    # get HF exchange contribution and alpha, beta and omega parameters if necessary
    # Todo: for now only dealing with hybrid but not range separated
    coeff_hf = None
    if is_hybrid_xc(xc_code):
        coeff_hf = hybrid_coeff(xc_code, spin=spin)

    print(f'DEV ATOMIC VALUES for DFT functional {xc_code}')
    print(at_dft)
    print('CHECK: SUM OF ATOMIC VALUES')
    print(np.sum(at_dft))

    return xc_e, at_dft, coeff_hf



def ee_hf_energy_interatomic(at_nums, grid_coords1, grid_coords2, mo_grid1, mo_grid2, occ, wq, aim_w1, aim_w2, atgrid_idx, rij_v, ncpus, coul=True, x=True, aa=False):
    """Function to compute Coulomb and HF exchange numerically"""
    print('DEV ee hf energy interatomic')

    start = time.time()
    # generating rho1 and rho2 for coulomb
    rho1 = np.zeros(mo_grid1.shape[1])
    rho2 = np.zeros(mo_grid2.shape[1])
    print(mo_grid2.shape)

    for i in range(mo_grid2.shape[0]):
        rho1 += occ[i] * mo_grid1[i, :] * mo_grid1[i, :]
        rho2 += occ[i] * mo_grid2[i, :] * mo_grid2[i, :]
    print('CHECK RHO1', np.sum(rho1*wq*aim_w1))
    print('CHECK RHO2', np.sum(rho2*wq*aim_w2))

    coul_mat = np.zeros((len(at_nums), len(at_nums)))
    t_exch_mat = np.zeros((len(at_nums), len(at_nums)))

    # ## ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
    if get_start_method(allow_none=True) is None:
        set_start_method('fork')

    logging.info(f"USING {ncpus} CPUS")

    # start = time.time()
    # end = time.time()
    # print("compilation time = ", end - start)

    for ia in range(len(at_nums)):
        # Get atomic grid for at A
        a_grid_coords = grid_coords1[atgrid_idx == ia, :]
        atgrid_idx_a = np.where(atgrid_idx == ia)[0]
        for ib in range(ia+1):
            if ia != ib and aa:
                print('Pass ia != ib', aa)
                continue
            else:
                b_grid_coords = grid_coords2[atgrid_idx == ib, :]
                atgrid_idx_b = np.where(atgrid_idx == ib)[0]
                # id_p1 = index for the atomic grid a
                # p1 = index of the atomic grid a within the molecular grid
                # rij_small = 0
                pool_coul= mp.Pool(processes=ncpus)
                pool_x= mp.Pool(processes=ncpus)
                segments_coul = []
                segments_x = []
                for id_p1, p1 in enumerate(atgrid_idx_a):
                    if coul:
                        segments_coul.append((id_p1, p1, rho1, rho2, atgrid_idx_b, a_grid_coords, b_grid_coords, wq,aim_w1, aim_w2, rij_v))
                    if x:
                        segments_x.append((id_p1, p1, mo_grid1, mo_grid2, atgrid_idx_b, a_grid_coords, b_grid_coords, wq, aim_w1, aim_w2, rij_v, occ))
                # print(len(segments))
                # Parallel execution of _aux_6n_integral
                if coul:
                    print('CALL COULOMB')
                    results_coul = pool_coul.starmap_async(_coul_integral, segments_coul).get()
                    pool_coul.close()
                    pool_coul.join()

                    integral_coul = 0
                    rij_small = 0
                    for out in results_coul:
                        integral_coul += out[0]
                        rij_small += out[1]

                    if ia != ib:
                        coul_mat[ia, ib] += integral_coul
                        coul_mat[ib, ia] = coul_mat[ia, ib]
                    else:
                        coul_mat[ia, ib] += integral_coul / 2
                if x:
                    print('CALL HF X')
                    results_x = pool_x.starmap_async(_exchange_integral, segments_x).get()
                    pool_x.close()
                    pool_x.join()

                    integral_exch = 0
                    rij_small = 0
                    for out in results_x:
                        integral_exch += out[0]
                        rij_small += out[1]

                    if ia != ib:
                        t_exch_mat[ia, ib] += -2*integral_exch
                        t_exch_mat[ib, ia] = t_exch_mat[ia, ib]
                    else:
                        t_exch_mat[ia, ib] += -integral_exch

                    # if occ is not None:
                    #     t_exch_mat = 0.25 * t_exch_mat

    print(f'DEBUG: TOTAL NUMBER OF POINTS WITH Rij < {rij_v} : {rij_small}')
    print('SUMMARY TWO ELECTRON NUMERICAL INTEGRATION')
    if coul:
        print(f'*** COULOMB PART ***')
        print(coul_mat)
        print(np.sum(np.triu(coul_mat)))
    if x:
        if occ[0] == 1:
            t_exch_mat = 0.5 * t_exch_mat
        # else:
        #     t_exch_mat = 0.25 * t_exch_mat
        print(f'*** HF EXCHANGE PART ***')
        print(t_exch_mat)
        print(np.sum(np.triu(t_exch_mat)))
        print()
        end = time.time()
        print("computation time = ", end - start)

    return coul_mat, t_exch_mat,


def ee_dft_energy_interatomic(at_nums, xc_code, xc_type, eval_mo_grid1, eval_mo_grad_grid1, wq, aim_w, atgrid_idx, aim_w_allp, spin=0, bod_type='bod'):

    print('DEV dft exchange-correlation interatomic')

    # Todo: Check passed objects and write dimensions of arrays

    # WARNING: mo_grid1/2 come as tuples when unrestricted

    # Initializer final array here
    dft_pairwise = np.zeros((len(at_nums), len(at_nums)))
    # Fixme: Update checks taking into account the MOs and MOs grad can be passed as tuple for unrestricted
    # GGA and hybrid  GGA with BOD gradient
    if xc_type == 'GGA':
        if spin == 1:
            grad_bod = np.zeros((len(at_nums), len(at_nums), 2, 3, eval_mo_grad_grid1[0].shape[2]))
        elif spin == 0:
            grad_bod = np.zeros((len(at_nums), len(at_nums), 3, eval_mo_grad_grid1.shape[2]))

    #     # Check MOs gradient
    #     if not isinstance(eval_mo_grad_grid1, np.ndarray) or not isinstance(eval_mo_grad_grid1, tuple):
    #         raise TypeError('MOs gradient should be passed as a numpy array')
        # elif eval_mo_grad_grid1.shape[0] != 3 and eval_mo_grad_grid1.shape[-1] != eval_mo_grid1.shape[0]:
        #     raise ValueError(
        #         'MOs gradient should be passed as a numpy array with dimensions (deriv coordinate, points, MOs)')
        # Initialize grad_bod

    # BOD density and GRAD BOD
    # Because number of alpha and beta occupied orbitals might differ the for loop block needs to be executed separately
    # WARNING: Assuming alpha/beta order for tuple eval_mod_grid1
    if spin == 1:
        bod = np.zeros((len(at_nums), len(at_nums), 2, eval_mo_grid1[0].shape[1]))
        for ia in range(len(at_nums)):
            for ib in range(len(at_nums)):
                # Loop over occupied i and j molecular orbitals
                for i in range(eval_mo_grid1[0].shape[0]):
                    for j in range(eval_mo_grid1[0].shape[0]):
                        sij_a = np.sum(eval_mo_grid1[0][i] * eval_mo_grid1[0][j]  * wq * aim_w_allp[ia] * aim_w)
                        sij_b = np.sum(eval_mo_grid1[0][j] * eval_mo_grid1[0][i] * wq* aim_w_allp[ib]* aim_w)
                        bod[ia, ib][0] +=  (aim_w_allp[ia]* sij_b + aim_w_allp[ib] * sij_a)* eval_mo_grid1[0][i] * eval_mo_grid1[0][j]
                        if xc_type == 'GGA':
                            grad_bod[ia, ib][0] += 2*(aim_w_allp[ia] * sij_b + aim_w_allp[ib] * sij_a)[None, :] * eval_mo_grad_grid1[0][:, i, :] * eval_mo_grid1[0][None, j, :]
                for i in range(eval_mo_grid1[1].shape[0]):
                    for j in range(eval_mo_grid1[1].shape[0]):
                        sij_a = np.sum(eval_mo_grid1[1][i] * eval_mo_grid1[1][j] * wq * aim_w_allp[ia] * aim_w)
                        sij_b = np.sum(eval_mo_grid1[1][j] * eval_mo_grid1[1][i] * wq * aim_w_allp[ib] * aim_w)
                        bod[ia, ib][1] += (aim_w_allp[ia] * sij_b + aim_w_allp[ib] * sij_a) * eval_mo_grid1[1][i] * eval_mo_grid1[1][j]
                        if xc_type == 'GGA':
                            grad_bod[ia, ib][1] += 2*(aim_w_allp[ia] * sij_b + aim_w_allp[ib] * sij_a)[None, :] * eval_mo_grad_grid1[1][:, i, :] * eval_mo_grid1[1][None, j, :]
    elif spin == 0:
        bod = np.zeros((len(at_nums), len(at_nums), eval_mo_grid1.shape[1]))

        for ia in range(len(at_nums)):
            for ib in range(len(at_nums)):
                # Loop over occupied i and j molecular orbitals
                for i in range(eval_mo_grid1.shape[0]):
                    for j in range(eval_mo_grid1.shape[0]):
                        sij_a = np.sum(eval_mo_grid1[i] * eval_mo_grid1[j]  * wq * aim_w_allp[ia] * aim_w)
                        sij_b = np.sum(eval_mo_grid1[j] * eval_mo_grid1[i] * wq* aim_w_allp[ib]* aim_w)
                        # 2 factor because restricted
                        bod[ia, ib] +=  2 *(aim_w_allp[ia]* sij_b + aim_w_allp[ib] * sij_a)* eval_mo_grid1[i] * eval_mo_grid1[j]
                        if xc_type == 'GGA':
                            grad_bod[ia, ib] += 4*(aim_w_allp[ia] * sij_b + aim_w_allp[ib] * sij_a)[None, :] * eval_mo_grad_grid1[:, i, :] * eval_mo_grid1[None, j, :]



    # Computing from BOD-3RDM
    if spin==1:
        if xc_type == 'GGA':
            bod_3rdm_a, bod_grad_3rdm_a = bo_dens(at_nums, eval_mo_grid1[0], wq, aim_w, atgrid_idx, aim_w_allp, eval_mo_grad_grid=eval_mo_grad_grid1[0], spin=spin)
            bod_3rdm_b, bod_grad_3rdm_b = bo_dens(at_nums, eval_mo_grid1[1], wq, aim_w, atgrid_idx, aim_w_allp, eval_mo_grad_grid=eval_mo_grad_grid1[1], spin=spin)

            bod_3rdm = np.zeros((2, bod_3rdm_a.shape[0], bod_3rdm_a.shape[1], bod_3rdm_a.shape[-1]))
            bod_grad_3rdm = np.zeros((2, bod_grad_3rdm_a.shape[0], bod_grad_3rdm_a.shape[1], 3, bod_grad_3rdm_a.shape[-1]))

            bod_3rdm[0] = bod_3rdm_a
            bod_3rdm[1] = bod_3rdm_b
            bod_grad_3rdm[0] = bod_grad_3rdm_a
            bod_grad_3rdm[1] = bod_grad_3rdm_b

            bod_3rdm = bod_3rdm.transpose(1, 2, 0, 3)
            bod_grad_3rdm = bod_grad_3rdm.transpose(1, 2, 0, 3, 4)
        else:
            bod_3rdm_a = bo_dens(at_nums, eval_mo_grid1[0], wq, aim_w, atgrid_idx, aim_w_allp, eval_mo_grad_grid=None, spin=spin)
            bod_3rdm_b = bo_dens(at_nums, eval_mo_grid1[1], wq, aim_w, atgrid_idx, aim_w_allp, eval_mo_grad_grid=None, spin=spin)
            bod_3rdm = np.zeros((2, bod_3rdm_a.shape[0], bod_3rdm_a.shape[1], bod_3rdm_a.shape[2]))
            bod_3rdm[0] = bod_3rdm_a
            bod_3rdm[1] = bod_3rdm_b
            bod_3rdm = bod_3rdm.transpose(1,2,0,3)
    else:
        if xc_type == 'GGA':
            bod_3rdm, bod_grad_3rdm = bo_dens(at_nums, eval_mo_grid1, wq, aim_w, atgrid_idx, aim_w_allp, eval_mo_grad_grid=eval_mo_grad_grid1, spin=0)
        else:
            bod_3rdm = bo_dens(at_nums, eval_mo_grid1, wq, aim_w, atgrid_idx, aim_w_allp, eval_mo_grad_grid=None, spin=0)

    # Add threshold between atoms?
    int_bod = np.zeros((len(at_nums), len(at_nums)))
    for ia, a in enumerate(at_nums):
        for ib in range(ia + 1):
            if ia != ib:
                print('BOD int', ia, ib)
                if bod_type == 'bod':
                    if spin == 0:
                        int_bod[ia, ib] = np.sum(bod[ia, ib] * wq * aim_w, axis=-1)
                    elif spin == 1:
                        int_bod[ia, ib] = np.sum(bod[ia, ib][0] * wq * aim_w, axis=-1) + np.sum(bod[ia, ib][1] * wq * aim_w, axis=-1)
                    print(f"Integration BOD",   np.sum(bod[ia,ib][0] * wq* aim_w, axis=-1) + np.sum(bod[ia,ib][1] * wq* aim_w, axis=-1) )
                elif bod_type == 'bod_3rdm':
                    if spin == 0:
                        int_bod[ia, ib] = np.sum(bod_3rdm[ia,ib] * wq* aim_w, axis=-1)
                    elif spin == 1:
                        int_bod[ia, ib] = np.sum(bod_3rdm[ia,ib][0] * wq* aim_w, axis=-1) + np.sum(bod_3rdm[ia,ib][1] * wq* aim_w, axis=-1)
                    print(f"Integration BOD from 3RDM",   np.sum(bod_3rdm[ia,ib][0] * wq* aim_w, axis=-1) + np.sum(bod_3rdm[ia,ib][1] * wq* aim_w, axis=-1))

    # assert 5 == 6
    dft_pairwise = np.zeros((len(at_nums), len(at_nums)))
    if xc_type == 'LDA':
        # LDA with BOD density
        for ia in range(len(at_nums)):
            for ib in range(len(at_nums)):
                if ia != ib:
                    if spin == 1:
                        r = eval_xc(xc_code, (bod[ia,ib][0],bod[ia,ib][1]) , spin=spin)
                        r_3rdm = eval_xc(xc_code, (bod_3rdm[ia, ib][0], bod_3rdm[ia, ib][1]), spin=spin)
                        if bod_type == 'bod':
                            dft_pairwise[ia, ib] = np.sum(r[0] *(bod[ia,ib][0] + bod[ia,ib][1])* wq * aim_w**2)
                        elif bod_type == 'bod_3rdm':
                            dft_pairwise[ia, ib] = np.sum(r_3rdm[0] * (bod_3rdm[ia, ib][0] + bod_3rdm[ia, ib][1]) * wq * aim_w ** 2)
                    elif spin == 0:
                        r = eval_xc(xc_code, bod[ia, ib], spin=spin)
                        r_3rdm = eval_xc(xc_code, bod_3rdm[ia, ib], spin=spin)
                        if bod_type == 'bod':
                            dft_pairwise[ia, ib] = np.sum(r[0] *bod[ia,ib]* wq * aim_w**2)
                        else:
                            dft_pairwise[ia, ib] = np.sum(r_3rdm[0] *bod_3rdm[ia,ib]* wq * aim_w**2)


    # GGA with BOD gradient
    elif xc_type == 'GGA':
        # Pass the bod and the grad_bod to libxc
        for ia in range(len(at_nums)):
            for ib in range(len(at_nums)):
                if ia != ib:
                    if spin == 1:
                        bod_data = np.zeros((2, 4, eval_mo_grid1[0].shape[1]))
                        bod_data_3rdm = np.zeros((2, 4, eval_mo_grid1[0].shape[1]))
                        bod_data[0, 0, :] = bod[ia, ib][0]
                        bod_data[1, 0, :] = bod[ia, ib][1]
                        bod_data_3rdm[0, 0, :] = bod_3rdm[ia, ib][0]
                        bod_data_3rdm[1, 0, :] = bod_3rdm[ia, ib][1]
                        bod_data[0, 1:, :] = grad_bod[ia, ib][0]
                        bod_data[1, 1:, :] = grad_bod[ia, ib][1]
                        bod_data_3rdm[0, 1:, :] = bod_grad_3rdm[ia, ib][0]
                        bod_data_3rdm[1, 1:, :] = bod_grad_3rdm[ia, ib][1]
                        r = eval_xc(xc_code, (bod_data[0], bod_data[1]), spin=1)
                        r_3rdm = eval_xc(xc_code, (bod_data_3rdm[0], bod_data_3rdm[1]), spin=1)
                        if bod_type == 'bod':
                            dft_pairwise[ia, ib] = np.sum(r[0] *(bod_data[0, 0, :] + bod_data[1, 0, :])* wq * aim_w**2)
                        elif bod_type == 'bod_3rdm':
                            dft_pairwise[ia, ib] = np.sum(r_3rdm[0] *(bod_data_3rdm[0, 0, :] + bod_data_3rdm[1, 0, :])* wq * aim_w**2)

                    elif spin == 0:
                        bod_data_o = np.zeros((4, eval_mo_grid1.shape[1]))
                        bod_data_3rdm = np.zeros((4, eval_mo_grid1.shape[1]))
                        bod_data_o[0, :] = bod[ia,ib]
                        bod_data_3rdm[0, :] = bod_3rdm[ia,ib]
                        bod_data_o[1:, :] = grad_bod[ia,ib]
                        bod_data_3rdm[1:, :] = bod_grad_3rdm[ia,ib]
                        r = eval_xc(xc_code, bod_data_o, spin=spin)
                        r_3rdm = eval_xc(xc_code, bod_data_3rdm, spin=spin)
                        if bod_type == 'bod':
                            dft_pairwise[ia, ib] = np.sum(r[0] * bod[ia, ib] * wq * aim_w**2)
                        elif bod_type == 'bod_3rdm':
                            dft_pairwise[ia, ib] = np.sum(r_3rdm[0] * bod_3rdm[ia, ib] * wq * aim_w**2)
    print(dft_pairwise)


    #metaGGA needs the laplacian: Not implemented

    return dft_pairwise, int_bod, bod_type

def iqa_interatomic(APOSTBase, ph2=0.162, zero_error=False, rij_v=1e-5, ncpus=1, bod_type='bod', out=True):
    """WIP: Function to obtain two electron dependent energy quantities and
    interatomic components for DFT xc energy based on the BOD method"""

    at_coords = APOSTBase.at_coords
    at_nums = np.array(APOSTBase.at_nums)
    # Grid info
    wq = APOSTBase.part.quadrature_weights
    mol_w = APOSTBase.part.aim_w
    atgrid_idx = APOSTBase.part.grid_to_atom_idx

    thresh = 1e-8
    occ_coeff_a = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
    occ_coeff_b = APOSTBase.mo_coeff_b[:, APOSTBase.occ_b > thresh]
    occ_a = APOSTBase.occ_a[APOSTBase.occ_a > thresh]
    occ_b = APOSTBase.occ_b[APOSTBase.occ_b > thresh]
    dm_a = APOSTBase.dm_a
    dm_b = APOSTBase.dm_b

    # Compute interatomic terms for HF exchange/coulomb and DFT
    # for each point in grid_1 (Atom A) integrate all points in grid_2 (Atom B).
    # We need to 1.Generate a second grid and 2.Atomic weights (quadrature) are the same for both grids as they depend only on the distance
    # 3. Regenerate AIM weights
    # Rotate specific angle in rads
    # 1. Get points and compute the polar angles theta and phi
    grid_coords1 = np.ascontiguousarray(copy.copy(APOSTBase.part.grid_coords))
    print('DEV HF 2 electron numerical integration')
    print(f'TOTAL NUMBER OF GRID1 POINTS: {grid_coords1.shape}')


    # First phb values
    # Calling method to rotate the grid coords
    # ph =  [(0,0.162), (0,0.18)]
    pha = 0
    #phb = 0.162
    if zero_error:
        # phb = 0.162
        # phb_2 = 0.144
        # phb = 0.188
        # phb_2 = 0.206
        phb = 0.206
        phb_2 = 0.188
    else:
        phb = ph2

    # Note for the code:
    # APOST3D code: APOST3D generates the rotated grid not by rotating grid1 but by starting from scratch.
    # It generates distance and theta,fix for the same radial and angular points and then instead of passing 0 for pha and phb
    # it passes the specified value. This second call of the prenumint rutine is only called inside enpart.f (and only if numerical 2e integration
    # is requested, not for semi analytical IQA). Once the theta and fix angles have been rotated the atomic coordinates are addedd
    # Here because we rely on the Grid pyscf machinery to get the points and the atomic(quadrature) weights before rotating the points
    # we substract the atomic coordinates, and then rotate them again. After that we generate a second set of AIM weights for the rotated grid but keep the same
    # atomi(quadrature) weights.
    # Note: Grids are computed for total density not spin separated
    # Todo: Check because dm_a should include all density for closed shell
    grid_coords2, mol_w_2 = APOSTBase.part.rotate_grid(at_coords, at_nums, pha, phb, dm_a + dm_b, mol=APOSTBase._mol)
    print(f'DEV: ROTATED GRID AND AIM : {grid_coords2.shape}, {mol_w_2.shape}')

    xc_code, xc_type = APOSTBase.xc_code
    if xc_type == 'LDA' or xc_type is None:
        ao_dens = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=0)
        eval_ao_grid1 = ao_dens
        eval_mo_grad_grid1_a = None
        eval_mo_grad_grid1_b = None
    elif xc_type == 'GGA':
        ao_dens = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=1)
        eval_ao_grid1 = ao_dens[0, :, :]
        eval_ao_grad_grid1 = ao_dens[1:, :, :]
        if APOSTBase.unrestr:
            eval_mo_grad_grid1_a = np.einsum('ni, opn->oip ', occ_coeff_a, eval_ao_grad_grid1)
            eval_mo_grad_grid1_b = np.einsum('ni, opn->oip ', occ_coeff_b, eval_ao_grad_grid1)
        else:
            eval_mo_grad_grid1_a = np.einsum('ni, opn->oip ', occ_coeff_a, eval_ao_grad_grid1)

    else:
        raise TypeError(f"Unknown XC type: {xc_type}")

    eval_ao_grid2 = eval_ao(APOSTBase._mol, grid_coords2, deriv=0)
    if zero_error:
        # Get the third set of coordinates
        grid_coords3, mol_w_3 = APOSTBase.part.rotate_grid(at_coords, at_nums, pha, phb_2, dm_a + dm_b, mol=APOSTBase._mol)
        print(f'DEV: ROTATED GRID AND AIM : {grid_coords2.shape}, {mol_w_2.shape}')
        eval_ao_grid3 = eval_ao(APOSTBase._mol, grid_coords3, deriv=0)

    if APOSTBase.unrestr:
        eval_mo_grid1_b = np.ascontiguousarray(np.einsum('pa,ai->ip ', eval_ao_grid1, occ_coeff_b))
        eval_mo_grid2_b = np.ascontiguousarray(np.einsum('pa,ai->ip ', eval_ao_grid2, occ_coeff_b))
        dens_grid_1_b = np.zeros((eval_mo_grid1_b.shape[1]))
        dens_grid_2_b = np.zeros((eval_mo_grid2_b.shape[1]))
        if zero_error:
            eval_mo_grid3_b = np.ascontiguousarray(np.einsum('pa,ai->ip ', eval_ao_grid3, occ_coeff_b))
            dens_grid_3_b = np.zeros((eval_mo_grid3_b.shape[1]))
        for i in range(eval_mo_grid1_b.shape[0]):
            dens_grid_1_b += eval_mo_grid1_b[i, :] * eval_mo_grid1_b[i, :] * occ_b[i]
            dens_grid_2_b += eval_mo_grid2_b[i, :] * eval_mo_grid2_b[i, :] * occ_b[i]
            if zero_error:
                dens_grid_3_b += eval_mo_grid3_b[i, :] * eval_mo_grid3_b[i, :] * occ_b[i]

    eval_mo_grid1_a = np.ascontiguousarray(np.einsum('pa,ai->ip ', eval_ao_grid1, occ_coeff_a))
    eval_mo_grid2_a = np.ascontiguousarray(np.einsum('pa,ai->ip ', eval_ao_grid2, occ_coeff_a))
    dens_grid_1_a = np.zeros((eval_mo_grid1_a.shape[1]))
    dens_grid_2_a = np.zeros((eval_mo_grid2_a.shape[1]))
    if zero_error:
        eval_mo_grid3_a = np.ascontiguousarray(np.einsum('pa,ai->ip ', eval_ao_grid3, occ_coeff_a))
        dens_grid_3_a = np.zeros((eval_mo_grid3_a.shape[1]))
    for i in range(eval_mo_grid1_a.shape[0]):
        dens_grid_1_a += eval_mo_grid1_a[i, :] * eval_mo_grid1_a[i, :] * occ_a[i]
        dens_grid_2_a += eval_mo_grid2_a[i, :] * eval_mo_grid2_a[i, :] * occ_a[i]
        if zero_error:
            dens_grid_3_a += eval_mo_grid3_a[i, :] * eval_mo_grid3_a[i, :] * occ_a[i]


    total_elect_int_1_a = np.sum(dens_grid_1_a * mol_w * wq)
    total_elect_int_2_a = np.sum(dens_grid_2_a * mol_w_2 * wq)
    if zero_error:
        total_elect_int_3_a = np.sum(dens_grid_3_a * mol_w_3 * wq)
    if APOSTBase.unrestr:
        print('CHECK ALPHA TOTAL ELECT INT 1', total_elect_int_1_a)
        print('CHECK ALPHA TOTAL ELECT INT 2', total_elect_int_2_a)
        print('CHECK ALPHA TOTAL ELECT INT 3 (ZERO ERROR)', total_elect_int_3_a)
        total_elect_int_1_b = np.sum(dens_grid_1_b * mol_w * wq)
        total_elect_int_2_b = np.sum(dens_grid_2_b * mol_w_2 * wq)
        total_elect_int_3_b = np.sum(dens_grid_3_b * mol_w_3 * wq)
        print('CHECK BETA TOTAL ELECT INT 1', total_elect_int_1_b)
        print('CHECK BETA TOTAL ELECT INT 2', total_elect_int_2_b)
        print('CHECK BETA TOTAL ELECT INT 3 (ZERO ERROR)', total_elect_int_3_b)
    else:
        print('CHECK TOTAL ELECT INT 1', total_elect_int_1_a)
        print('CHECK TOTAL ELECT INT 2', total_elect_int_2_a)
        if zero_error:
            print('CHECK TOTAL ELECT INT 3 (ZERO ERROR)', total_elect_int_3_a)

    # First one electron terms
    iqa_r_a, iqa_r_b = iqa_atomic(APOSTBase, ee_hf=False, out=False)

    if APOSTBase.dft:
        if APOSTBase.unrestr:
            aa_at_dft = copy.copy(iqa_r_b['at_dft_xc'])
            iqa_r_b['at_dft_xc'], iqa_r_b['bod'], iqa_r_b['bod_type'] = ee_dft_energy_interatomic(at_nums, xc_code, xc_type,
                                                                             (eval_mo_grid1_a, eval_mo_grid1_b),
                                                                             (eval_mo_grad_grid1_a, eval_mo_grad_grid1_b),
                                                                             wq,
                                                                             mol_w,
                                                                             atgrid_idx,
                                                                             APOSTBase.part.aim_w_allp,
                                                                             spin=1, bod_type=bod_type)
            if iqa_r_b['coeff_mix']:
                _, iqa_r_a['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                                 grid_coords1,
                                                                 grid_coords2,
                                                                 eval_mo_grid1_a,
                                                                 eval_mo_grid2_a,
                                                                 occ_a,
                                                                 wq,
                                                                 mol_w,
                                                                 mol_w_2,
                                                                 atgrid_idx,
                                                                 rij_v=rij_v,
                                                                 ncpus=ncpus,
                                                                 coul=False,
                                                                 x=True)
                _, iqa_r_b['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                                 grid_coords1,
                                                                 grid_coords2,
                                                                 eval_mo_grid1_b,
                                                                 eval_mo_grid2_b,
                                                                 occ_b,
                                                                 wq,
                                                                 mol_w,
                                                                 mol_w_2,
                                                                 atgrid_idx,
                                                                 rij_v=rij_v,
                                                                 ncpus=ncpus,
                                                                 coul=False,
                                                                 x=True)
                iqa_r_a['t_exch'] = np.sum(np.triu(iqa_r_a['at_exch']))
                iqa_r_b['t_exch'] = np.sum(np.triu(iqa_r_b['at_exch']))

                # Scale also atomic exchange matrix?
                iqa_r_b['t_exch'] = iqa_r_b['t_exch'] * iqa_r_b['coeff_mix']
                iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * iqa_r_b['coeff_mix']
            for ia in range(len(at_nums)):
                iqa_r_b['at_dft_xc'][ia, ia] = aa_at_dft[ia] - 0.5 * np.sum(iqa_r_b['at_dft_xc'][:, ia])
            iqa_r_b['t_dft_xc'] = np.sum(np.triu(iqa_r_b['at_dft_xc']))

        else:
            aa_at_dft = copy.copy(iqa_r_a['at_dft_xc'])
            iqa_r_a['at_dft_xc'], iqa_r_a['bod'], iqa_r_a['bod_type'] = ee_dft_energy_interatomic(at_nums, xc_code,
                                                                             xc_type,
                                                                             eval_mo_grid1_a,
                                                                             eval_mo_grad_grid1_a,
                                                                             wq,
                                                                             mol_w,
                                                                             atgrid_idx,
                                                                             APOSTBase.part.aim_w_allp,
                                                                             spin=0, bod_type=bod_type)
            if iqa_r_a['coeff_mix']:
                # Scaling exchange
                _ , iqa_r_a['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                                grid_coords1,
                                                                grid_coords2,
                                                                eval_mo_grid1_a,
                                                                eval_mo_grid2_a,
                                                                occ_a,
                                                                wq,
                                                                mol_w,
                                                                mol_w_2,
                                                                atgrid_idx,
                                                                rij_v=rij_v,
                                                                ncpus=ncpus,
                                                                coul=False,
                                                                x=True)

                iqa_r_a['t_exch'] = np.sum(np.triu(iqa_r_a['at_exch']))
                iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * iqa_r_a['coeff_mix']
            for ia in range(len(at_nums)):
                iqa_r_a['at_dft_xc'][ia, ia] = aa_at_dft[ia] - 0.5 * np.sum(iqa_r_a['at_dft_xc'][:, ia])
            iqa_r_a['t_dft_xc'] = np.sum(np.triu(iqa_r_a['at_dft_xc']))

    if APOSTBase.unrestr:
        eval_mo_grid1_all = np.zeros((eval_mo_grid1_a.shape[0] + eval_mo_grid1_b.shape[0], eval_mo_grid1_a.shape[1]))
        eval_mo_grid1_all[: eval_mo_grid1_a.shape[0], :] = eval_mo_grid1_a
        eval_mo_grid1_all[eval_mo_grid1_a.shape[0]: eval_mo_grid1_a.shape[0] + eval_mo_grid1_b.shape[0], :] = eval_mo_grid1_b

        eval_mo_grid2_all = np.zeros((eval_mo_grid2_a.shape[0] + eval_mo_grid2_b.shape[0], eval_mo_grid2_a.shape[1]))
        eval_mo_grid2_all[: eval_mo_grid2_a.shape[0], :] = eval_mo_grid2_a
        eval_mo_grid2_all[eval_mo_grid2_a.shape[0]: eval_mo_grid2_a.shape[0] + eval_mo_grid2_b.shape[0], :] = eval_mo_grid2_b

        occ_all = np.zeros(occ_a.shape[0] + occ_b.shape[0])
        occ_all[: occ_a.shape[0]] = occ_a
        occ_all[occ_a.shape[0]: occ_a.shape[0] + occ_b.shape[0]] = occ_b

        iqa_r_b['at_coul'], _ = ee_hf_energy_interatomic(at_nums,
                                                         grid_coords1,
                                                         grid_coords2,
                                                         eval_mo_grid1_all,
                                                         eval_mo_grid2_all,
                                                         occ_all,
                                                         wq,
                                                         mol_w,
                                                         mol_w_2,
                                                         atgrid_idx,
                                                         rij_v=rij_v,
                                                         ncpus=ncpus,
                                                         coul=True,
                                                         x=False)
        iqa_r_b['t_coul'] = np.sum(np.triu(iqa_r_b['at_coul']))
        if iqa_r_b.get('coeff_mix', None) is None and APOSTBase.dft is False:
            _, iqa_r_a['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                             grid_coords1,
                                                             grid_coords2,
                                                             eval_mo_grid1_a,
                                                             eval_mo_grid2_a,
                                                             occ_a,
                                                             wq,
                                                             mol_w,
                                                             mol_w_2,
                                                             atgrid_idx,
                                                             rij_v=rij_v,
                                                             ncpus=ncpus,
                                                             coul=False,
                                                             x=True)
            _, iqa_r_b['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                             grid_coords1,
                                                             grid_coords2,
                                                             eval_mo_grid1_b,
                                                             eval_mo_grid2_b,
                                                             occ_b,
                                                             wq,
                                                             mol_w,
                                                             mol_w_2,
                                                             atgrid_idx,
                                                             rij_v=rij_v,
                                                             ncpus=ncpus,
                                                             coul=False,
                                                             x=True)
            iqa_r_a['t_exch'] = np.sum(np.triu(iqa_r_a['at_exch']))
            iqa_r_b['t_exch'] = np.sum(np.triu(iqa_r_b['at_exch']))

    else:
        # DEV: zero error scheme: Get values with pha
        iqa_r_a['at_coul'], _ = ee_hf_energy_interatomic(at_nums,
                                                         grid_coords1,
                                                         grid_coords2,
                                                         eval_mo_grid1_a,
                                                         eval_mo_grid2_a,
                                                         occ_a,
                                                         wq,
                                                         mol_w,
                                                         mol_w_2,
                                                         atgrid_idx,
                                                         rij_v=rij_v,
                                                         ncpus=ncpus,
                                                         coul=True,
                                                         x=False)
        iqa_r_a['t_coul'] = np.sum(np.triu(iqa_r_a['at_coul']))

        if iqa_r_a.get('coeff_mix', None) is None and APOSTBase.dft is False:
            _, iqa_r_a['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                             grid_coords1,
                                                             grid_coords2,
                                                             eval_mo_grid1_a,
                                                             eval_mo_grid2_a,
                                                             occ_a,
                                                             wq,
                                                             mol_w,
                                                             mol_w_2,
                                                             atgrid_idx,
                                                             rij_v=rij_v,
                                                             ncpus=ncpus,
                                                             coul=False,
                                                             x=True)

            iqa_r_a['t_exch'] = np.sum(np.triu(iqa_r_a['at_exch']))
    # Todo: Interpolation of ee atomic terms
    # 1 Rotate grid
    # 2. Compute AA terms again for Coulomb and Exchange(if necessary)/we can't do that before checking the error
    # Save two electron error --> numeric - exact (totals)
    # Compute delta ee (For atomic terms) difference between pha1 and pha2
    # Compute new error and check if different sign
    # Compute new angle and obtain interpolated values for atomic terms
    print('DEV ZERO ERROR: FIRST SUMMARY')
    out_summary([iqa_r_a], at_nums, APOSTBase.energy)
    if zero_error:
        # Compute total error
        if APOSTBase.unrestr:
            pass
        else:
            vee_analytic = APOSTBase.energy - iqa_r_a['t_nn'] - iqa_r_a['t_kin'] - iqa_r_a['t_en']
            vee_iqa =  iqa_r_a['t_coul'] +  iqa_r_a['t_exch']
            # vee_analytic = iqa_r_a['ee_analyt']
            print('EE analytic:', vee_analytic)
            print('EE IQA', vee_iqa)
            diff = abs(vee_iqa) - abs(vee_analytic)
            twoelerr = diff*23.06*27.212
            print('DIFF (AU):', diff)
            print('DIFF (Kcal/mol):', twoelerr)
            if abs(twoelerr) > 0.25:
                ze = {}
                if APOSTBase.unrestr:
                    pass
                else:
                    ze['at_coul'], _ = ee_hf_energy_interatomic(at_nums,
                                                             grid_coords1,
                                                             grid_coords3,
                                                             eval_mo_grid1_a,
                                                             eval_mo_grid3_a,
                                                             occ_a,
                                                             wq,
                                                             mol_w,
                                                             mol_w_3,
                                                             atgrid_idx,
                                                             rij_v=rij_v,
                                                             ncpus=ncpus,
                                                             coul=True,
                                                             x=False,
                                                             aa=True)
                    ze['t_coul'] = np.sum(np.triu(ze['at_coul']))
                    _, ze['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                                     grid_coords1,
                                                                     grid_coords3,
                                                                     eval_mo_grid1_a,
                                                                     eval_mo_grid3_a,
                                                                     occ_a,
                                                                     wq,
                                                                     mol_w,
                                                                     mol_w_3,
                                                                     atgrid_idx,
                                                                     rij_v=rij_v,
                                                                     ncpus=ncpus,
                                                                     coul=False,
                                                                     x=True,
                                                                     aa=True)

                    ze['t_exch'] = np.sum(np.triu(ze['at_exch']))

                # Get deltaee
                e1_aa = np.trace(iqa_r_a['at_coul']) + np.trace(iqa_r_a['at_exch'])
                e2_aa = np.trace(ze['at_coul']) + np.trace(ze['at_exch'])
                # deltaee_alt =0
                # for i in range(len(at_nums)):
                #     deltaee_alt += iqa_r_a['at_coul'][i][i] - ze['at_coul'][i][i]
                #     deltaee_alt += iqa_r_a['at_exch'][i][i] - ze['at_exch'][i][i]
                #     print(i, iqa_r_a['at_coul'][i][i], ze['at_coul'][i][i], iqa_r_a['at_coul'][i][i] - ze['at_coul'][i][i])
                #     print(i, iqa_r_a['at_exch'][i][i], ze['at_exch'][i][i],  iqa_r_a['at_exch'][i][i] - ze['at_exch'][i][i])
                deltaee = (e1_aa-e2_aa)*23.06*27.212
                print('Delta E^{1}_AA-E^{2}_AA', deltaee)
                print('Error1 = ', twoelerr)
                print('Error2 = ', twoelerr -deltaee)
                pha_best = 1 - (twoelerr/deltaee)
                print('Pha Best: ', pha_best)
                check_signed = twoelerr-deltaee*twoelerr
                if check_signed > 0:
                    print('WARNING: SAME SIGN. EXTRAPOLATING')

                # Interpolating intraatomic values
                print('DEV Coulomb')
                print(iqa_r_a['at_coul'])
                print('DEV Exchange')
                print(iqa_r_a['at_exch'])
                for ia in range(len(at_nums)):
                    iqa_r_a['at_coul'][ia][ia] = pha_best*iqa_r_a['at_coul'][ia][ia] + (1-pha_best)*ze['at_coul'][ia][ia]
                    iqa_r_a['at_exch'][ia][ia] = pha_best*iqa_r_a['at_exch'][ia][ia] + (1-pha_best)*ze['at_exch'][ia][ia]
                print('DEV Coulomb after')
                print(iqa_r_a['at_coul'])
                print('DEV Exchange after')
                print(iqa_r_a['at_exch'])
                iqa_r_a['t_exch'] = np.sum(np.triu(iqa_r_a['at_exch']))
                iqa_r_a['t_coul'] = np.sum(np.triu(iqa_r_a['at_coul']))
                vee_iqa = iqa_r_a['t_coul'] + iqa_r_a['t_exch']
                # vee_analytic = iqa_r_a['ee_analyt']
                print('EE analytic:', vee_analytic)
                print('EE IQA', vee_iqa)
                diff = abs(vee_iqa) - abs(vee_analytic)
                twoelerr = diff * 23.06 * 27.212
                print('DIFF (AU):', diff)
                print('DIFF (Kcal/mol):', twoelerr)

    if out:
        if APOSTBase.unrestr:
            out_summary([iqa_r_a, iqa_r_b], at_nums, APOSTBase.energy)
        else:
            print('DEV ZERO ERROR: SECOND SUMMARY')
            out_summary([iqa_r_a], at_nums, APOSTBase.energy)



    # Todo: Multipolar approach
    return (iqa_r_a, iqa_r_b)


# Note: To add flexibility in the future and for testing purposes we could add an argument to specify if we only want the alpha/beta electrons computed
# independently of the calculation
def iqa_atomic(APOSTBase, ee_hf=True, out=True):

    """WIP: Function to obtain one electron dependent energy quantities for SD wavefunctions"""

    # NN energy: Coulomb formula of charge repulsion-Analytic (Done)
    # EN energy: Compute density (Done)
    # Kinetic energy: Obtain density and density gradients (Done)
    # Exchange and Coulomb HF (pseudo analytical with only one electron numerical): Use the point charge integral within pyscf (Done)
    # DFT exchange-correlation (Done)


    # Molecule info
    at_coords = APOSTBase.at_coords
    at_nums = np.array(APOSTBase.at_nums)



    # Grid info
    grid_coords = APOSTBase.part.grid_coords
    wq = APOSTBase.part._grid.quadrature_weights
    mol_w = APOSTBase.part.aim_w
    atgrid_idx = APOSTBase.part.grid_to_atom_idx
    # mg = APOSTBase.part.grid_weights

    #Orbitals and density
    # todo: Put a warning for the methods we load a calc and a post-hartre fock method objects in case there are discrepencies
    # between the number of MOs because of the threshold
    thresh = 1e-11
    if APOSTBase.cas or APOSTBase._fci:
        if APOSTBase.cas:
            norb = APOSTBase._cas_data['ncas'] + APOSTBase._cas_data['ncore']
        else:
            norb = APOSTBase._fci_data['norb']
        occ_coeff_a = APOSTBase._cas_mo_coeff[:, :norb]
        occ_coeff_b = APOSTBase._cas_mo_coeff[:, :norb]
    occ_a = APOSTBase.occ_a[APOSTBase.occ_a > thresh]
    occ_b = APOSTBase.occ_b[APOSTBase.occ_b > thresh]
    dm_a = APOSTBase.dm_a
    dm_b = APOSTBase.dm_b
    # Density/AO/MOs on the grid and DFT
    xc_code, xc_type = APOSTBase.xc_code
    if xc_type == 'LDA' or xc_type is None:
        ao_dens = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=0)
    elif xc_type == 'GGA':
        ao_dens = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=1)
    else:
        raise TypeError(f"Unknown XC type: {xc_type}")

    if APOSTBase.unrestr:
        if APOSTBase.dft:
            rho_a = eval_rho(APOSTBase._mol, ao_dens, dm_a, xctype=xc_type)
            rho_b = eval_rho(APOSTBase._mol, ao_dens, dm_b, xctype=xc_type)
        else:
            rho_a = eval_rho(APOSTBase._mol, ao_dens, dm_a)
            rho_b = eval_rho(APOSTBase._mol, ao_dens, dm_b)
    else:
        if APOSTBase.dft:
            rho_a = eval_rho(APOSTBase._mol, ao_dens, dm_a, xctype=xc_type)
        else:
            rho_a = eval_rho(APOSTBase._mol, ao_dens, dm_a)


    print('CHECK ATOMIC CHARGES')
    # For now only rho_a. Todo: Computing charges should be a standalone function called from different modules
    pop_all = []
    for ia in range(len(at_nums)):
        if xc_type == 'GGA':
            pop = np.sum(rho_a[0][atgrid_idx == ia] * wq[atgrid_idx == ia] * mol_w[atgrid_idx == ia])
        else:
            pop = np.sum(rho_a[atgrid_idx == ia] * wq[atgrid_idx == ia] * mol_w[atgrid_idx == ia])
        pop_all.append(pop)
        print(f'POPULATION AT {ia +1} = {pop: .5f}')
    print(f'CHECK SUM POPULATION = {np.sum(pop_all):.5f}')

    # Initialize dicts
    iqa_r_a = {}
    iqa_r_b = {}
    # Compute analytic terms
    ee_terms = compute_energy_terms(APOSTBase, out=False)
    iqa_r_a['t_nn'] = ee_terms['nn']

    if APOSTBase.unrestr:
        iqa_r_b['kin_analyt'] = ee_terms['kin']
        iqa_r_b['en_analyt'] = ee_terms['en']
        iqa_r_b['coul_analyt'] = ee_terms['coul']
        iqa_r_b['exch_analyt'] = ee_terms['x_b']
        iqa_r_a['exch_analyt'] = ee_terms['x_a']
    else:
        iqa_r_a['kin_analyt'] = ee_terms['kin']
        iqa_r_a['en_analyt'] = ee_terms['en']
        iqa_r_a['coul_analyt'] = ee_terms['coul']
        iqa_r_a['exch_analyt'] = ee_terms['x_a'] + ee_terms['x_b']
        iqa_r_a['ee_analyt'] = ee_terms['ee']

    if APOSTBase.unrestr:
        if xc_type == 'GGA':
            iqa_r_b['t_en'], iqa_r_b['at_en'], iqa_r_b['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, rho_a[0] + rho_b[0], wq, mol_w, atgrid_idx)
        else:
            iqa_r_b['t_en'], iqa_r_b['at_en'], iqa_r_b['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, rho_a + rho_b, wq, mol_w, atgrid_idx)

        if APOSTBase.sd:
            occ_mat_a = np.diag(occ_a)
            occ_mat_b = np.diag(occ_b)
            t_kin_b, at_kin_b =   kin_energy(at_nums, at_coords, APOSTBase._mol, occ_coeff_b, occ_mat_b, grid_coords, wq, mol_w, atgrid_idx)
            t_kin_a, at_kin_a =   kin_energy(at_nums, at_coords, APOSTBase._mol, occ_coeff_a, occ_mat_a, grid_coords, wq, mol_w, atgrid_idx)
        else:
            occ_mat_a = APOSTBase._dm1_a
            occ_mat_b = APOSTBase._dm1_b
            t_kin_b, at_kin_b = kin_energy(at_nums, at_coords, APOSTBase._mol, occ_coeff_b, occ_mat_b, grid_coords, wq, mol_w, atgrid_idx)
            t_kin_a, at_kin_a = kin_energy(at_nums, at_coords, APOSTBase._mol, occ_coeff_a, occ_mat_a, grid_coords, wq, mol_w, atgrid_idx)

        iqa_r_b['t_kin'], iqa_r_b['at_kin'] =  t_kin_b +t_kin_a ,  at_kin_a + at_kin_b
        if ee_hf:
            # first Coulomb because only need to compute it once with alpha and beta combined
            occ_coeff = np.zeros((occ_coeff_b.shape[0], occ_coeff_a.shape[1] + occ_coeff_b.shape[1]))
            occ_coeff[:, : occ_coeff_a.shape[1]] = occ_coeff_a
            occ_coeff[:, occ_coeff_a.shape[1]:] = occ_coeff_b
            occs = np.zeros((occ_a.shape[0] + occ_b.shape[0]))
            occs[: occ_a.shape[0]] = occ_a
            occs[occ_a.shape[0]:] = occ_b
            iqa_r_b['t_coul'], iqa_r_b['at_coul'], _, _ =  ee_hf_energy(at_nums,
                                                                        at_coords,
                                                                        APOSTBase._mol,
                                                                        dm_b+dm_a,
                                                                        grid_coords,
                                                                        APOSTBase.part.ao_eval_grid,
                                                                        occ_coeff,
                                                                        occs,
                                                                        wq,
                                                                        mol_w,
                                                                        atgrid_idx,
                                                                        coul=True,
                                                                        x=False)

            if xc_type != 'LDA': # Checking LDA because these type of functionals don't incorporate HF-exchange but we can't check omega
                # HF exchange for alpha and beta separately
                _, _, iqa_r_a['t_exch'], iqa_r_a['at_exch'] = ee_hf_energy(at_nums,
                                                                           at_coords,
                                                                           APOSTBase._mol,
                                                                           dm_a,
                                                                           grid_coords,
                                                                           APOSTBase.part.ao_eval_grid,
                                                                           occ_coeff_a,
                                                                           occ_a,
                                                                           wq,
                                                                           mol_w,
                                                                           atgrid_idx,
                                                                           coul=False,
                                                                           x=True)
                _, _, iqa_r_b['t_exch'], iqa_r_b['at_exch'] = ee_hf_energy(at_nums,
                                                                           at_coords,
                                                                           APOSTBase._mol,
                                                                           dm_b,
                                                                           grid_coords,
                                                                           APOSTBase.part.ao_eval_grid,
                                                                           occ_coeff_b,
                                                                           occ_b,
                                                                           wq,
                                                                           mol_w,
                                                                           atgrid_idx,
                                                                           coul=False,
                                                                           x=True)


    else:
        iqa_r_b['t_nn'] = nn_energy(at_nums, at_coords)
        if xc_type == 'GGA':
            iqa_r_a['t_en'], iqa_r_a['at_en'], iqa_r_a['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, rho_a[0], wq, mol_w, atgrid_idx)
        else:
            iqa_r_a['t_en'], iqa_r_a['at_en'], iqa_r_a['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, rho_a, wq, mol_w, atgrid_idx)
        if APOSTBase.sd:
            occ_mat = np.diag(occ_a)
        else:
            if APOSTBase._cas:
                norb = (APOSTBase._cas_data['ncas'] + APOSTBase._cas_data['ncore'])
            elif APOSTBase._fci:
                norb = APOSTBase._fci_data['norb']
            occ_mat = APOSTBase._dij[: norb, :norb]
        iqa_r_a['t_kin'], iqa_r_a['at_kin'] = kin_energy(at_nums, at_coords, APOSTBase._mol, occ_coeff_a, occ_mat, grid_coords, wq, mol_w, atgrid_idx)
        if ee_hf:
            iqa_r_a['t_coul'],  iqa_r_a['at_coul'], _, _ = ee_hf_energy(at_nums,
                                                                        at_coords,
                                                                        APOSTBase._mol,
                                                                        dm_a,
                                                                        grid_coords,
                                                                        APOSTBase.part.ao_eval_grid,
                                                                        occ_coeff_a,
                                                                        occ_a,
                                                                        wq,
                                                                        mol_w,
                                                                        atgrid_idx,
                                                                        coul=True,
                                                                        x=False)
            if xc_type != 'LDA' or xc_type is None:
                _, _, iqa_r_a['t_exch'], iqa_r_a['at_exch'] = ee_hf_energy(at_nums,
                                                                           at_coords,
                                                                           APOSTBase._mol,
                                                                           dm_a,
                                                                           grid_coords,
                                                                           APOSTBase.part.ao_eval_grid,
                                                                           occ_coeff_a,
                                                                           occ_a,
                                                                           wq,
                                                                           mol_w,
                                                                           atgrid_idx,
                                                                           coul=False,
                                                                           x=True)


    # Todo: fix this because we need to be able to separate both contributions and below just repeats the same calc
    # Fixme: For now commented but it might be needed to deal with custom DFT functionals
    # if f_corr and f_exch:
    #     iqa_r_a['t_coul'], iqa_r_a['t_exch'], iqa_r_a['at_coul'], iqa_r_a['at_exch'] = ee_hf_energy(at_nums, at_coords, APOSTBase._mol,dm, grid_coords, APOSTBase.part.ao_eval_grid, occ_coeff, occ, qw, mol_w, atgrid_idx)
    #     iqa_r_a['t_dft_c'], iqa_r_a['at_dft_c'], coeff_mix = ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, mol_w, atgrid_idx)
    #     iqa_r_a['t_dft_x'], iqa_r_a['at_dft_x'], coeff_mix = ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, mol_w, atgrid_idx)
    #     if coeff_mix:
    #         iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * coeff_mix
    #         iqa_r_a['at_coul'] = iqa_r_a['at_coul'] * coeff_mix
    #         # iqa_r_a['t_dft_x'] = iqa_r_a['t_dft_x'] * (1-coeff_mix)
    #         # iqa_r_a['at_dft_x'] = iqa_r_a['at_dft_x'] * (1-coeff_mix)
    #     else:
    #         iqa_r_a.pop('t_exch')
    #         iqa_r_a.pop('at_exch')

    if APOSTBase.dft:
        # First check unrestricted
        if APOSTBase.unrestr:
            iqa_r_b['t_dft_xc'], iqa_r_b['at_dft_xc'], iqa_r_b['coeff_mix'] = ee_dft_energy(at_nums, xc_code, xc_type,(rho_a, rho_b), ao_dens, wq, mol_w, atgrid_idx, spin=1)
            # The calculated energy density is on obtained with alpha and beta densities. Only populating results for beta dict and assigning 0 to alpha results dict
            # iqa_r_a['t_dft_xc'], iqa_r_a['at_dft_xc'], iqa_r_a['coeff_mix'] = 0, 0, 0
            if ee_hf:
                if iqa_r_b['coeff_mix']:
                    # HF exchange for alpha and beta separately
                    _, _, iqa_r_a['t_exch'], iqa_r_a['at_exch'] = ee_hf_energy(at_nums,
                                                                               at_coords,
                                                                               APOSTBase._mol,
                                                                               dm_a,
                                                                               grid_coords,
                                                                               APOSTBase.part.ao_eval_grid,
                                                                               occ_coeff_a,
                                                                               occ_a,
                                                                               wq,
                                                                               mol_w,
                                                                               atgrid_idx,
                                                                               coul=False,
                                                                               x=True)
                    _, _, iqa_r_b['t_exch'], iqa_r_b['at_exch'] = ee_hf_energy(at_nums,
                                                                               at_coords,
                                                                               APOSTBase._mol,
                                                                               dm_b,
                                                                               grid_coords,
                                                                               APOSTBase.part.ao_eval_grid,
                                                                               occ_coeff_b,
                                                                               occ_b,
                                                                               wq,
                                                                               mol_w,
                                                                               atgrid_idx,
                                                                               coul=False,
                                                                               x=True)
                    # Scaling exchange
                    # iqa_r_b['t_dft_xc'] = iqa_r_b['t_dft_xc'] * (1 - iqa_r_b['coeff_mix'])
                    # iqa_r_b['at_dft_xc'] = iqa_r_b['at_dft_xc'] * (1 - iqa_r_b['coeff_mix'])
                    iqa_r_b['t_exch'] = iqa_r_b['t_exch'] * iqa_r_b['coeff_mix']
                    iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * iqa_r_b['coeff_mix']
                # elif xc_type != 'LDA':
                #     iqa_r_b.pop('t_exch')
                #     iqa_r_b.pop('at_exch')
                #     iqa_r_a.pop('t_exch')
                #     iqa_r_a.pop('at_exch')


        else:
            iqa_r_a['t_dft_xc'], iqa_r_a['at_dft_xc'], iqa_r_a['coeff_mix'] = ee_dft_energy(at_nums, xc_code, xc_type,rho_a, ao_dens, wq, mol_w, atgrid_idx, spin=0)
            if ee_hf:
                if iqa_r_a['coeff_mix']:
                    _, _, iqa_r_a['t_exch'], iqa_r_a['at_exch'] = ee_hf_energy(at_nums,
                                                                               at_coords,
                                                                               APOSTBase._mol,
                                                                               dm_a,
                                                                               grid_coords,
                                                                               APOSTBase.part.ao_eval_grid,
                                                                               occ_coeff_a,
                                                                               occ_a,
                                                                               wq,
                                                                               mol_w,
                                                                               atgrid_idx,
                                                                               coul=False,
                                                                               x=True)
                    # Scaling exchange
                    # iqa_r_a['t_dft_xc'] = iqa_r_a['t_dft_xc'] * (1 - iqa_r_a['coeff_mix'])
                    # iqa_r_a['at_dft_xc'] = iqa_r_a['at_dft_xc'] * (1 - iqa_r_a['coeff_mix'])
                    iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * iqa_r_a['coeff_mix']
                # elif xc_type != 'LDA':
                #     iqa_r_a.pop('t_exch')
                #     iqa_r_a.pop('at_exch')

    # Note: we could add printing as optional since we return the dict at the end
    if out:
        if APOSTBase.unrestr:
            out_summary([iqa_r_a, iqa_r_b], at_nums, APOSTBase.energy)
        else:
            out_summary([iqa_r_a], at_nums, APOSTBase.energy)

    return (iqa_r_a, iqa_r_b)

