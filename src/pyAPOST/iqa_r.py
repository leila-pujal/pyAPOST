# IQA module in real space
import copy as copy
import time as time

import multiprocessing as mp
from multiprocessing import set_start_method, get_start_method
import os as os
import numba as numba
from numba import njit

import numpy as np
import logging

from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.libxc import eval_xc, is_hybrid_xc, hybrid_coeff

from pyAPOST.utils import out_summary


# @njit(nopython=True, parallel=True)
@njit(nopython=True)
def _two_elect_integral(id_p1, p1, mo_grid1, mo_grid2, atgrid_idx_b, a_grid_coords, b_grid_coords, qw, aim_w1, aim_w2, rij_v=1e-5):
    """ WIP: Auxiliary function for parallel calculation of 2e integrals"""
    # print('inside _two_elect_integral')
    # Loop only for occupied orbitals
    ther_rij = rij_v
    if id_p1 == 0:
        print('Threshold rij', ther_rij)
    coul_dens = np.zeros((mo_grid2[0, atgrid_idx_b].shape[0]))
    exch_dens = np.zeros((mo_grid2[0, atgrid_idx_b].shape[0]))
    for i in range(mo_grid1.shape[0]):
        for j in range(mo_grid2.shape[0]):
            coul_dens += 2 * mo_grid1[i, p1] * mo_grid1[i, p1] * mo_grid2[j, atgrid_idx_b ] * mo_grid2[j, atgrid_idx_b]
            exch_dens += mo_grid1[i, p1] * mo_grid1[j, p1] * mo_grid2[j, atgrid_idx_b] * mo_grid2[i, atgrid_idx_b]
    # Compute rij between i point index (p1) and atom b atgrid points
    # rij = np.linalg.norm(a_grid_coords[id_p1, :] - b_grid_coords, axis=-1)
    rij = a_grid_coords[id_p1, :] - b_grid_coords
    rij = np.sqrt(np.sum(np.square(rij), axis=-1))
    if id_p1 == 0:
        print('Shape rij', rij.shape)
        print('Filter rij', rij[rij<ther_rij].shape)


    coul_dens = coul_dens / rij
    exch_dens = exch_dens / rij
    # filter values of the points wit rij < 1e-3. If not filtered they add spurious big values
    coul_dens[rij<ther_rij] = 0
    exch_dens[rij<ther_rij] = 0
    # order is 1e-15
    # rij_small += rij[rij<1e-9].shape[0]
    # Now integration starts. Get aim weights for p1 atgridA and all points in atgridB
    coul_b = np.sum(coul_dens* qw[atgrid_idx_b] * aim_w2[atgrid_idx_b]) # integrated for atB
    exch_b = np.sum(exch_dens* qw[atgrid_idx_b] * aim_w2[atgrid_idx_b]) # integrated for atB
    coul = coul_b * qw[p1] * aim_w1[p1]
    exch = exch_b * qw[p1] * aim_w1[p1]

    return coul, exch, rij[rij<ther_rij].shape[0]




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
    print('en_atomic_matrix')
    print(en_atomic_matrix)
    # en_atomic_matrix_dic = copy.copy(en_atomic_matrix)
    # share_matrix
    share_factor = 0.5
    share_matrix = np.zeros((natoms, natoms, natoms))
    for i in range(natoms):
        share_matrix[i, i, :] = share_factor
        share_matrix[i, :, i] = 1 - share_factor
        share_matrix[i, i, i] = 1

    en_cond_en = np.sum(en_atomic_matrix[None, :, :] * share_matrix, axis=(1, 2))
    print(en_cond_en)
    assert np.allclose(np.sum(en_cond_en), t_en, atol=1e-6)

    return t_en, en_atomic_matrix, en_cond_en

def kin_energy(at_nums, at_coords, mol, dm, grid_coords, qw, aim_w, atgrid_idx ):
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

    # Get the gradients.
    eval_d_ao = eval_ao(mol, grid_coords, deriv=1)
    # eval_d_ao = eval_ao(mol, grid_coords)
    print(eval_d_ao.shape)
    # Build MO gradients
    # 100/100, 010/010, 001/001
    eval_d_ao = eval_d_ao.transpose(0, 2, 1)
    kin = np.zeros(grid_coords.shape[0])
    # Below loop is the same as:
    # dens_100 = np.trace(np.dot(dm, (eval_d_ao[1, :, None, :] * eval_d_ao[1, None, :, :])))
    # dens_010 = np.trace(np.dot(dm, (eval_d_ao[2, :, None, :] * eval_d_ao[2, None, :, :])))
    # dens_001 = np.trace(np.dot(dm, (eval_d_ao[3, :, None, :] * eval_d_ao[3, None, :, :])))
    for i_dv in range(1,4):
        kin += np.trace(np.dot(dm, (eval_d_ao[i_dv, :, None, :] * eval_d_ao[i_dv, None, :, :])))
    kin *= 0.5
    kin_t = np.sum(kin * qw * aim_w)
    print(f'TOTAL INTEGRATED KINETIC ENERGY = {kin_t: .6f}', )

    # To get the general kinetic energy now compute the laplacian
    # Investigate: PySCF does not return the expected derivative components for deriv=2
    # Expected 6 but got 10 as if I was requesting the 3rd order derivative
    # eval_l_ao = eval_ao(mol, grid_coords, deriv=2)

    # Getting atomic contributions
    at_kin = np.zeros((len(at_nums)))
    for at in range(len(at_nums)):
        at_k = np.sum(kin[atgrid_idx == at] * qw[atgrid_idx == at] * aim_w[atgrid_idx == at])
        at_kin[at] = at_k

    print('DEV ATOMIC VALUES FOR KINETIC ENERGY')
    print(at_kin)
    print(f'CHECK: SUM OF ATOMIC VALUES : {np.sum(at_kin): .6f}')

    return kin_t, at_kin


def ee_hf_energy(at_nums, at_coords, mol, dm, grid_coords, eval_ao_grid, mo_coeff, occ, qw, aim_w, atgrid_idx, coul=True, x=True):
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

    total_elect_int = np.sum(dens_grid * qw * aim_w)


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
        t_coul = np.sum(t_coul_num * qw * aim_w)
        print(f'TOTAL COULOMB {t_coul: .5f}')
        at_coul = np.zeros((at_nums.size))
        for at in range(len(at_nums)):
            at_c = np.sum(t_coul_num[atgrid_idx == at] * qw[atgrid_idx == at] * aim_w[atgrid_idx == at])
            at_coul[at] = at_c
        print('DEV ATOMIC VALUES for semianalitic ee terms')
        print(at_coul)
        print('CHECK: SUM OF ATOMIC VALUES')
        print(np.sum(at_coul))


    if x:
        t_exch = np.sum(t_exch_num * qw * aim_w)
        print(f'TOTAL EXCHANGE {t_exch: .5f}')
        at_exch = np.zeros((at_nums.size))
        for at in range(len(at_nums)):
            at_x = np.sum(t_exch_num[atgrid_idx == at] * qw[atgrid_idx == at] * aim_w[atgrid_idx == at])
            at_exch[at] = at_x
        print('DEV ATOMIC VALUES for semianalitic ee terms')
        print(at_exch)
        print('CHECK: SUM OF ATOMIC VALUES')
        print(np.sum(at_exch))


    # Compute atomic values

    # assert 5 == 7

    return t_coul, at_coul, t_exch, at_exch


def ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, aim_w, atgrid_idx, spin=0):
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
    # todo: we need to sort the spin too
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
            xc_e = np.sum(xc_e_dens * (rho[0] + rho[1]) * qw * aim_w)
            for at in range(len(at_nums)):
                at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] *(rho[0] + rho[1])[atgrid_idx == at]* qw[atgrid_idx == at] * aim_w[atgrid_idx == at])

        else:
            # GGA
            print('gga')
            xc_e = np.sum(xc_e_dens * (rho[0][0] + rho[1][0]) * qw * aim_w)
            for at in range(len(at_nums)):
                at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] *(rho[0][0] + rho[1][0])[atgrid_idx == at]* qw[atgrid_idx == at] * aim_w[atgrid_idx == at])
    # Restricted
    elif isinstance(rho, np.ndarray):
        print('restricted')
        if len(rho.shape) == 1:
            # LDA
            print('lda')
            xc_e = np.sum(xc_e_dens * rho *  qw * aim_w)
            for at in range(len(at_nums)):
                at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] *rho[atgrid_idx == at]* qw[atgrid_idx == at] * aim_w[atgrid_idx == at])

        else:
            print('gga')
            xc_e = np.sum(xc_e_dens * rho[0] *  qw * aim_w)
            for at in range(len(at_nums)):
                at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] * rho[0][atgrid_idx == at] * qw[atgrid_idx == at] * aim_w[atgrid_idx == at])


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



# auxiliary ful(id1, p1, a, b, eval_mo_g1, eval_mo_g2, part, part2, grid1, points2, atgrid_a, atgrid_b, occupied_mo):
def ee_hf_energy_interatomic(at_nums, grid_coords1, grid_coords2, mo_grid1, mo_grid2, qw, aim_w1, aim_w2, atgrid_idx, rij_v):
    """Function to compute Coulomb and HF exchange numerically"""
    print('DEV ee hf energy interatomic')

    start = time.time()
    coul_mat = np.zeros((len(at_nums), len(at_nums)))
    t_exch_mat = np.zeros((len(at_nums), len(at_nums)))

    # todo: change setting the cpus here
    ncpus =4
    # ## ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
    # if get_start_method(allow_none=True) is None:
    #     print('inside fork')
    #     set_start_method('fork')
    #
    # logging.info(f"USING {ncpus} CPUS")

    # start = time.time()
    # end = time.time()
    # print("compilation time = ", end - start)

    for ia in range(len(at_nums)):
        # Get atomic grid for at A
        a_grid_coords = grid_coords1[atgrid_idx == ia, :]
        atgrid_idx_a = np.where(atgrid_idx == ia)[0]
        # print('CHECK idx a', atgrid_idx_a[0:5], atgrid_idx_a[-5:-1], atgrid_idx_a.shape)
        for ib in range(ia+1):
            b_grid_coords = grid_coords2[atgrid_idx == ib, :]
            atgrid_idx_b = np.where(atgrid_idx == ib)[0]
            # id_p1 = index for the atomic grid a
            # p1 = index of the atomic grid a within the molecular grid
            # rij_small = 0
            pool = mp.Pool(processes=ncpus)
            segments = []
            for id_p1, p1 in enumerate(atgrid_idx_a):
                # if ia == 0 and ib == 0 and id_p1 == 0:
                #     start = time.time()
                #     _two_elect_integral(id_p1, p1, mo_grid1, mo_grid2, atgrid_idx_b[0:1], a_grid_coords[0:1], b_grid_coords[0:1], qw[0:1], aim_w1[0:1], aim_w2[0:1])
                #     end = time.time()
                #     print("compilation time = ", end - start)
                segments.append((id_p1, p1, mo_grid1, mo_grid2, atgrid_idx_b, a_grid_coords, b_grid_coords, qw, aim_w1, aim_w2, rij_v))
            # print(len(segments))
            # Parallel execution of _aux_6n_integral
            results = pool.starmap_async(_two_elect_integral, segments).get()
            pool.close()
            pool.join()



            # Add integral for r1
            integral_coul = 0
            integral_exch = 0
            rij_small = 0
            for out in results:
                integral_coul += out[0]
                integral_exch += out[1]
                rij_small += out[2]

            if ia != ib:
                coul_mat[ia, ib] += 2*integral_coul
                t_exch_mat[ia, ib] += 2*integral_exch
                coul_mat[ib, ia] = coul_mat[ia, ib]
                t_exch_mat[ib, ia] = t_exch_mat[ia, ib]
            else:
                coul_mat[ia, ib] += integral_coul
                t_exch_mat[ia, ib] += integral_exch

    print(f'DEBUG: TOTAL NUMBER OF POINTS WITH Rij < {rij_v} : {rij_small}')
    print('SUMMARY TWO ELECTRON NUMERICAL INTEGRATION')
    print(f'*** COULOMB PART ***')
    print(coul_mat)

    # Q: If pure functional theoretically we don't need to compute it. However the bootleneck right now should be
    # the `for` loop so for the same price we can integrate both
    print(f'*** HF EXCHANGE PART ***')
    print(-t_exch_mat)
    print()
    end = time.time()
    print("computation time = ", end - start)

    return coul_mat, -t_exch_mat,


def ee_dft_energy_interatomic(at_nums, xc_code, xc_type, eval_mo_grid1, eval_mo_grad_grid1, qw, aim_w, atgrid_idx, aim_w_allp, spin=0):

    print('DEV dft exchange-correlation interatomic')

    # Todo: Check passed objects and write dimensions of arrays

    # Here restricted/unrestriced. Development for Restricted only

    # Initializer final array here
    dft_pairwise = np.zeros((len(at_nums), len(at_nums)))
    # GGA and hybrid  GGA with BOD gradient
    if xc_type == 'GGA':
        # Check MOs gradient
        if not isinstance(eval_mo_grad_grid1, np.ndarray):
            raise TypeError('MOs gradient should be passed as a numpy array')
        elif eval_mo_grad_grid1.shape[0] != 3 and eval_mo_grad_grid1.shape[-1] != eval_mo_grid1.shape[0]:
            raise ValueError(
                'MOs gradient should be passed as a numpy array with dimensions (deriv coordinate, points, MOs)')
        # Initialize grad_bod
        grad_bod = np.zeros((len(at_nums), len(at_nums), 3, eval_mo_grad_grid1.shape[2]))

    print(eval_mo_grad_grid1.shape)
    # BOD density and GRAD BOD
    bod = np.zeros((len(at_nums), len(at_nums), eval_mo_grid1.shape[1]))
    for ia in range(len(at_nums)):
        for ib in range(len(at_nums)):
            # Loop over occupied i and j molecular orbitals
            for i in range(eval_mo_grid1.shape[0]):
                for j in range(eval_mo_grid1.shape[0]):
                    sij_a = np.sum(eval_mo_grid1[i] * eval_mo_grid1[j]  * qw * aim_w_allp[ia] * aim_w)
                    sij_b = np.sum(eval_mo_grid1[j] * eval_mo_grid1[i] * qw* aim_w_allp[ib]* aim_w)
                    # 2 factor because restricted
                    bod[ia, ib] +=  2 *(aim_w_allp[ia]* sij_b + aim_w_allp[ib] * sij_a)* eval_mo_grid1[i] * eval_mo_grid1[j]
                    if xc_type == 'GGA':
                        grad_bod[ia, ib] += 4*(aim_w_allp[ia] * sij_b + aim_w_allp[ib] * sij_a)[None, :] * eval_mo_grad_grid1[:, i, :] * eval_mo_grid1[None, j, :]

    # Integrate BOD
    # Add threshold between atoms?
    int_bod = np.zeros((len(at_nums), len(at_nums)))
    for ia, a in enumerate(at_nums):
        for ib in range(ia + 1):
            if ia != ib:
                print('BOD int', ia, ib)
                int_bod[ia, ib] = np.sum(bod[ia,ib] * qw* aim_w)
                print(f"Integration BOD",  int_bod[ia, ib])

    if xc_type == 'LDA':
        # LDA with BOD density
        dft_pairwise = np.zeros((len(at_nums), len(at_nums)))
        for ia in range(len(at_nums)):
            for ib in range(len(at_nums)):
                if ia != ib:
                    r = eval_xc(xc_code, bod[ia,ib], spin=0)
                    dft_pairwise[ia, ib] = np.sum(r[0] *bod[ia, ib]* qw * aim_w**2)


    # GGA with BOD gradient
    elif xc_type == 'GGA':
        # Pass the bod and the grad_bod to libxc
        for ia in range(len(at_nums)):
            for ib in range(len(at_nums)):
                if ia != ib:
                    bod_data = np.zeros((4, eval_mo_grid1.shape[1]))
                    bod_data[0, :] = bod[ia,ib]
                    bod_data[1:, :] = grad_bod[ia,ib]
                    r = eval_xc(xc_code, bod_data, spin=0)
                    dft_pairwise[ia, ib] = np.sum(r[0] * bod[ia, ib] * qw * aim_w**2)

        print(dft_pairwise)






    #metaGGA needs the laplacian: Not implemented



    return dft_pairwise, int_bod



def iqa_interatomic(APOSTBase, f_corr=None, f_exch=None, ph2=0.162, rij_v=1e-5):
    """WIP: Function to obtain two electron dependent energy quantities and
    interatomic components for DFT xc energy based on the BOD method"""

    at_coords = APOSTBase.at_coords
    at_nums = np.array(APOSTBase.at_nums)
    # Grid info
    # Todo: fix calling _grid object to get the quadrature weights and maybe change the name to wp
    qw = APOSTBase.part._grid.quadrature_weights
    mol_w = APOSTBase.part.aim_w
    atgrid_idx = APOSTBase.part.grid_to_atom_idx

    # Density info. For now only SD
    # Todo: extend to Post-Hartree Fock methods
    thresh = 1e-8
    occ_coeff_a = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
    occ_coeff_b = APOSTBase.mo_coeff_b[:, APOSTBase.occ_a > thresh]
    occ_a = APOSTBase.occ_a[APOSTBase.occ_a > thresh]
    occ_b = APOSTBase.occ_b[APOSTBase.occ_b > thresh]
    dm_a = APOSTBase.dm_a
    dm_b = APOSTBase.dm_b

    # First one electron terms
    iqa_r_a = iqa_atomic(APOSTBase, f_corr=None, f_exch=None, ee_energy=False)
    # Create holding variable for optional beta electrons results
    iqa_r_b = {}
    if APOSTBase.unrestr:
        iqa_r_b = iqa_atomic(APOSTBase, f_corr=None, f_exch=None, ee_energy=False)

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
    grid_coords2, mol_w_2 = APOSTBase.part.rotate_grid(at_coords, at_nums, pha, phb, dm_a + dm_b)
    print(f'DEV: ROTATED GRID AND AIM : {grid_coords2.shape}, {mol_w_2.shape}')

    # Evaluating molecular orbitals on Grid1 and Grid2
    xc_code, xc_type = APOSTBase.xc_code
    if xc_type == 'LDA' or xc_type is None:
        ao_dens = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=0)
        # Getting rho components for the AA DFT terms
        rho = eval_rho(APOSTBase._mol, ao_dens, dm_a, xctype=xc_type)
        eval_ao_grid1 = ao_dens
        eval_mo_grad_grid1 = None
    elif xc_type == 'GGA':
        # Getting density evaluated on the AO basis and on the points
        ao_dens = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=1)
        # Getting rho components for the AA DFT terms
        # rho, dx_rho, dy_rho, dz_rho = eval_rho(APOSTBase._mol, ao_dens, dm, xctype='GGA')
        rho = eval_rho(APOSTBase._mol, ao_dens, dm_a, xctype=xc_type)
        # ao_dens returns array 4,N_AO with 4 including rho, dx, dy, dz.
        eval_ao_grid1 = ao_dens[0, :,:]
        eval_ao_grad_grid1 = ao_dens[1:, :, :]
        eval_mo_grad_grid1 = np.einsum('ni, opn->oip ', occ_coeff_a, eval_ao_grad_grid1)
    else:
        raise TypeError(f"Unknown XC type: {xc_type}")

    eval_ao_grid2 = eval_ao(APOSTBase._mol, grid_coords2, deriv=0)
    # Note: DEV happening with coeff a because restricted closed shell. Todo:  extend to beta
    eval_mo_grid1 = np.ascontiguousarray(np.einsum('pa,ai->ip ', eval_ao_grid1, occ_coeff_a))
    eval_mo_grid2 = np.ascontiguousarray(np.einsum('pa,ai->ip ', eval_ao_grid2, occ_coeff_a))
    dens_grid_1 = np.zeros((eval_mo_grid1.shape[1]))
    dens_grid_2 = np.zeros((eval_mo_grid2.shape[1]))
    for i in range(eval_mo_grid1.shape[0]):
        dens_grid_1 += eval_mo_grid1[i, :] * eval_mo_grid1[i, :] * occ_a[i]
        dens_grid_2 += eval_mo_grid2[i, :] * eval_mo_grid2[i, :] * occ_a[i]

    total_elect_int_1 = np.sum(dens_grid_1 * mol_w * qw)
    total_elect_int_2 = np.sum(dens_grid_2 * mol_w_2 * qw)
    print('CHECK TOTAL ELECT INT 1', total_elect_int_1)
    print('CHECK TOTAL ELECT INT 1', total_elect_int_2)


    if f_corr and f_exch:
        # Todo: Same problem as with iqa_atomic, we need to specify x and c separately. Note: see when we do a custom functional
        # Get the AA DFT terms
        iqa_r_a['t_dft_c'], iqa_r_a['at_dft_c'], coeff_mix = ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, mol_w, atgrid_idx)
        iqa_r_a['t_dft_x'], iqa_r_a['at_dft_x'], coeff_mix = ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, mol_w, atgrid_idx)
        # Get the AB DFT terms
        iqa_r_a['at_dft_c'], iqa_r_a['bod'] = ee_dft_energy_interatomic(at_nums, xc_code, xc_type, eval_mo_grid1, eval_mo_grad_grid1, qw, mol_w, atgrid_idx, APOSTBase.part.aim_w_allp)
        iqa_r_a['at_dft_c'], iqa_r_a['bod'] = ee_dft_energy_interatomic(at_nums, xc_code, xc_type, eval_mo_grid1, eval_mo_grad_grid1, qw, mol_w, atgrid_idx, APOSTBase.part.aim_w_allp)
        # Note: the way the code works Coulomb and HF Exchange are computed simultaneously, thus, even if we may not need HF exchange it is computed
        iqa_r_a['at_coul'], iqa_r_a['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                                      grid_coords1,
                                                                      grid_coords2,
                                                                      eval_mo_grid1,
                                                                      eval_mo_grid2,
                                                                      qw,
                                                                      mol_w,
                                                                      mol_w_2,
                                                                      atgrid_idx,
                                                                      rij_v=rij_v)
        iqa_r_a['t_coul'], iqa_r_a['t_exch'] = np.sum(iqa_r_a['at_coul']), np.sum(iqa_r_a['at_exch'])
        if coeff_mix:
            iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * coeff_mix
        else:
            iqa_r_a.pop('t_exch')
            iqa_r_a.pop('at_exch')


    elif f_exch:
        # Get the AA DFT terms
        iqa_r_a['t_dft_xc'], iqa_r_a['at_dft_xc'], coeff_mix = ee_dft_energy(at_nums, xc_code, xc_type,
                                                                       rho, ao_dens, qw, mol_w,
                                                                       atgrid_idx)
        # Get the AB DFT terms
        iqa_r_a['at_dft_xc'], iqa_r_a['bod'] = ee_dft_energy_interatomic(at_nums, xc_code, xc_type,
                                                                    eval_mo_grid1,
                                                                    eval_mo_grad_grid1, qw, mol_w,
                                                                    atgrid_idx,
                                                                    APOSTBase.part.aim_w_allp)
        # Note: the way the code works Coulomb and HF Exchange are computed simultaneously, thus, even if we may not need HF exchange it is computed
        iqa_r_a['at_coul'], iqa_r_a['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                                      grid_coords1,
                                                                      grid_coords2,
                                                                      eval_mo_grid1,
                                                                      eval_mo_grid2,
                                                                      qw,
                                                                      mol_w,
                                                                      mol_w_2,
                                                                      atgrid_idx,
                                                                      rij_v=rij_v)
        iqa_r_a['t_coul'], iqa_r_a['t_exch'] = np.sum(iqa_r_a['at_coul']), np.sum(iqa_r_a['at_exch'])
        if coeff_mix:
            iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * coeff_mix
        else:
            iqa_r_a.pop('t_exch')
            iqa_r_a.pop('at_exch')

    elif not APOSTBase.dft:
        iqa_r_a['at_coul'], iqa_r_a['at_exch'] = ee_hf_energy_interatomic(at_nums,
                                                                      grid_coords1,
                                                                      grid_coords2,
                                                                      eval_mo_grid1,
                                                                      eval_mo_grid2,
                                                                      qw,
                                                                      mol_w,
                                                                      mol_w_2,
                                                                      atgrid_idx,
                                                                      rij_v=rij_v)
        iqa_r_a['t_coul'], iqa_r_a['t_exch'] = np.sum(iqa_r_a['at_coul']), np.sum(iqa_r_a['at_exch'])


    #  Print Summary
    logging.info('Summary IQA')
    print(f"Nucleus-Nucleus repulsion energy: {iqa_r_a['t_nn']:.5f}")
    # print('Nucleus-Nucleus repulsion energy (PySCF/Libcint): ', nn_int)
    # diff = abs(iqa_r_a['nn_total'] - nn_int)
    # print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)

    print('------------------------------------')
    print(f"Electron-Nucleus attraction energy: {iqa_r_a['t_en']:.5f}")
    print(f'Electron-Nucleus attraction energy (PySCF/Libcint): {iqa_r_a["en_analyt"]: .5f}')
    diff = abs(iqa_r_a['t_en'] - iqa_r_a['en_analyt'])
    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
    print()
    print("Atomic Electron-Nucleus attraction energy (Condensed):")
    for idx, at in enumerate(at_nums):
        print(f"{at}   {iqa_r_a['at_en_cond'][idx]: .5f}")
    print('------------------------------------')
    print(f"Kinetic energy:  {iqa_r_a['t_kin']:.5f}")
    print(f'Kinetic energy(PySCF/Libcint):  {iqa_r_a["kin_analyt"]: .5f}')
    diff = abs(iqa_r_a['t_kin'] - iqa_r_a['kin_analyt'])
    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
    print()
    print('Atomic Kinetic energy:')
    for idx, at in enumerate(at_nums):
        print(f"{at}   {iqa_r_a['at_kin'][idx]: .5f}")

    print('------------------------------------')
    print('Interatomic Electron-Electron interaction energy')
    print()
    print(f"Coulomb energy: {iqa_r_a['t_coul']: .5f}", )
    print(f'Coulomb energy(PySCF/Libcint):  {iqa_r_a["coul_analyt"]: .5f}')
    diff = abs(iqa_r_a['t_coul'] - iqa_r_a["coul_analyt"])
    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
    print()
    print('Full Coulomb energy decomposition:')



    for idx, at in enumerate(at_nums):
        print(f"{at}   {iqa_r_a['at_coul'][idx]: .5f}")
    print()
    if f_exch and f_corr:
        print(f"DTF Exchange energy: {iqa_r_a['t_dft_x']: .5f}")
        print(f"DTF Correlation energy: {iqa_r_a['t_dft_c']: .5f}")
        print()
        print('Atomic DFT Exchange energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_r_a['at_dft_x'][idx]: .5f}")
        print('Atomic DFT Correlation energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_r_a['at_dft_c'][idx]: .5f}")

    if f_exch and not f_corr:
        print(f"DTF Exchange-Correlation energy: {iqa_r_a['t_dft_xc']: .5f}")
        print()
        print('Atomic DFT Exchange-Correlation energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_r_a['at_dft_xc'][idx]: .5f}")

    if 'x_hf_total' in iqa_r_a.keys():
        print(f"HF Exchange: {iqa_r_a['t_exch']: .5f}")
        print(f'HF Exchange energy(PySCF/Libcint):  {exch_analyt: .5f}')
        diff = abs(iqa_r_a['t_exch'] - exch_analyt)
        print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print('COEFF MIX :', coeff_mix)
        print()
        print('Atomic HF Exchange energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_r_a['at_exch'][idx]: .5f}")


    total_sum_energy = 0
    for k in iqa_r_a.keys():
        if k.startswith('t_'):
            total_sum_energy += iqa_r_a[k]

    print('TOTAL ENERGY (SUM OF COMPONENTS):', total_sum_energy)
    print('TOTAL ENERGY (PySCF):', APOSTBase.energy)
    print('DIFF (AU) :', total_sum_energy - APOSTBase.energy)
    print('DIFF (Kcal) :', (total_sum_energy - APOSTBase.energy) / 0.0015936014376406278)
    if not ee_energy:
        print('WARNING: EE energy not calculated')

    assert 5 ==6


    return iqa_r_a



# def iqa_atomic(APOSTBase, f_corr=None, f_exch=None, ee_hf=True):
# Note: To add flexibility in the future and for testing purposes we could add an argument to specify if we only want the alpha/beta electrons computed
# independently of the calculation
def iqa_atomic(APOSTBase, ee_hf=True):

    """WIP: Function to obtain one electron dependent energy quantities"""

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
    qw = APOSTBase.part._grid.quadrature_weights
    mol_w = APOSTBase.part.aim_w
    atgrid_idx = APOSTBase.part.grid_to_atom_idx
    # mg = APOSTBase.part.grid_weights

    #Orbitals and density
    # Todo: extend to post hartree fock methods
    # For now doing SD
    thresh = 1e-8
    occ_coeff_a = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
    occ_coeff_b = APOSTBase.mo_coeff_b[:, APOSTBase.occ_b > thresh]
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
            pop = np.sum(rho_a[0][atgrid_idx == ia] * qw[atgrid_idx == ia] * mol_w[atgrid_idx == ia])
        else:
            pop = np.sum(rho_a[atgrid_idx == ia] * qw[atgrid_idx == ia] * mol_w[atgrid_idx == ia])
        pop_all.append(pop)
        print(f'POPULATION AT {ia +1} = {pop: .5f}')
    print(f'CHECK SUM POPULATION = {np.sum(pop_all):.5f}')

    iqa_r_a = {}
    iqa_r_b = {}
    two_eint = APOSTBase._mol.intor('int2e')
    if APOSTBase.unrestr:
        iqa_r_b['kin_analyt'] = np.trace(np.dot(dm_b + dm_a, APOSTBase._mol.intor_symmetric('int1e_kin')))
        iqa_r_b['en_analyt'] = np.trace(np.dot(dm_b +dm_a, APOSTBase._mol.intor_symmetric('int1e_nuc')))
        # 2elec coulomb and HF exchange
        coul_1_b = np.einsum("ab,abcd,cd->ad", dm_a + dm_b, two_eint, dm_a + dm_b)
        exch_1_b = np.einsum("ac,abcd,bd->ad", dm_b, two_eint, dm_b)
        exch_1_a = np.einsum("ac,abcd,bd->ad", dm_a, two_eint, dm_a)
        iqa_r_b['coul_analyt'] = 0.5 * np.sum(np.einsum("ad->a", coul_1_b))
        iqa_r_b['exch_analyt'] = -0.5 * np.sum(np.einsum("ad->a", exch_1_b))
        iqa_r_a['exch_analyt'] = -0.5 * np.sum(np.einsum("ad->a", exch_1_a))
    else:
        # EN
        iqa_r_a['kin_analyt'] = np.trace(np.dot(dm_a, APOSTBase._mol.intor_symmetric('int1e_kin')))
        iqa_r_a['en_analyt'] = np.trace(np.dot(dm_a, APOSTBase._mol.intor_symmetric('int1e_nuc')))
        # 2elec coulomb and HF exchange
        coul_1_a = np.einsum("ab,abcd,cd->ad", dm_a, two_eint, dm_a)
        exch_1_a = np.einsum("ac,abcd,bd->ad", dm_a, two_eint, dm_a)
        iqa_r_a['coul_analyt'] = 0.5 * np.sum(np.einsum("ad->a", coul_1_a))
        iqa_r_a['exch_analyt'] = -0.25 * np.sum(np.einsum("ad->a", exch_1_a))


    iqa_r_a['t_nn'] = nn_energy(at_nums, at_coords)

    if APOSTBase.unrestr:
        if xc_type == 'GGA':
            iqa_r_b['t_en'], iqa_r_b['at_en'], iqa_r_b['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, rho_a[0] + rho_b[0], qw, mol_w, atgrid_idx)
        else:
            iqa_r_b['t_en'], iqa_r_b['at_en'], iqa_r_b['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, rho_a + rho_b, qw, mol_w, atgrid_idx)
        iqa_r_b['t_kin'], iqa_r_b['at_kin'] =  kin_energy(at_nums, at_coords, APOSTBase._mol, dm_a + dm_b, grid_coords, qw, mol_w, atgrid_idx)
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
                                                                     qw,
                                                                     mol_w,
                                                                     atgrid_idx,
                                                                     coul=True,
                                                                     x=False)
            if xc_type != 'LDA':
                # HF exchange for alpha and beta separately
                _, _, iqa_r_a['t_exch'], iqa_r_a['at_exch'] = ee_hf_energy(at_nums,
                                                                          at_coords,
                                                                          APOSTBase._mol,
                                                                          dm_a,
                                                                          grid_coords,
                                                                          APOSTBase.part.ao_eval_grid,
                                                                          occ_coeff_a,
                                                                          occ_a,
                                                                          qw,
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
                                                                            qw,
                                                                            mol_w,
                                                                            atgrid_idx,
                                                                            coul=False,
                                                                            x=True)


    else:
        iqa_r_b['t_nn'] = nn_energy(at_nums, at_coords)
        if xc_type == 'GGA':
            iqa_r_a['t_en'], iqa_r_a['at_en'], iqa_r_a['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, rho_a[0], qw, mol_w, atgrid_idx)
        else:
            iqa_r_a['t_en'], iqa_r_a['at_en'], iqa_r_a['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, rho_a, qw, mol_w, atgrid_idx)
        iqa_r_a['t_kin'], iqa_r_a['at_kin'] = kin_energy(at_nums, at_coords, APOSTBase._mol, dm_a, grid_coords, qw, mol_w, atgrid_idx)
        if ee_hf:
            iqa_r_a['t_coul'],  iqa_r_a['at_coul'], _, _ = ee_hf_energy(at_nums,
                                                                        at_coords,
                                                                        APOSTBase._mol,
                                                                        dm_a,
                                                                        grid_coords,
                                                                        APOSTBase.part.ao_eval_grid,
                                                                        occ_coeff_a,
                                                                        occ_a,
                                                                        qw,
                                                                        mol_w,
                                                                        atgrid_idx,
                                                                        coul=True,
                                                                        x=False)
            if xc_type != 'LDA':
                _, _, iqa_r_a['t_exch'], iqa_r_a['at_exch'] = ee_hf_energy(at_nums,
                                                                           at_coords,
                                                                           APOSTBase._mol,
                                                                           dm_a,
                                                                           grid_coords,
                                                                           APOSTBase.part.ao_eval_grid,
                                                                           occ_coeff_a,
                                                                           occ_a,
                                                                           qw,
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
            iqa_r_b['t_dft_xc'], iqa_r_b['at_dft_xc'], iqa_r_b['coeff_mix'] = ee_dft_energy(at_nums, xc_code, xc_type,(rho_a, rho_b), ao_dens, qw, mol_w, atgrid_idx, spin=1)
            # The calculated energy density is on obtained with alpha and beta densities. Only populating results for beta dict and assigning 0 to alpha results dict
            # iqa_r_a['t_dft_xc'], iqa_r_a['at_dft_xc'], iqa_r_a['coeff_mix'] = 0, 0, 0
            if ee_hf:
                if iqa_r_b['coeff_mix']:
                    # Scaling exchange
                    # iqa_r_b['t_dft_xc'] = iqa_r_b['t_dft_xc'] * (1 - iqa_r_b['coeff_mix'])
                    # iqa_r_b['at_dft_xc'] = iqa_r_b['at_dft_xc'] * (1 - iqa_r_b['coeff_mix'])
                    iqa_r_b['t_exch'] = iqa_r_b['t_exch'] * iqa_r_b['coeff_mix']
                    iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * iqa_r_b['coeff_mix']
                elif xc_type != 'LDA':
                    iqa_r_b.pop('t_exch')
                    iqa_r_b.pop('at_exch')
                    iqa_r_a.pop('t_exch')
                    iqa_r_a.pop('at_exch')


        else:
            iqa_r_a['t_dft_xc'], iqa_r_a['at_dft_xc'], iqa_r_a['coeff_mix'] = ee_dft_energy(at_nums, xc_code, xc_type,rho_a, ao_dens, qw, mol_w, atgrid_idx, spin=0)
            if ee_hf:
                if iqa_r_a['coeff_mix']:
                    # Scaling exchange
                    # iqa_r_a['t_dft_xc'] = iqa_r_a['t_dft_xc'] * (1 - iqa_r_a['coeff_mix'])
                    # iqa_r_a['at_dft_xc'] = iqa_r_a['at_dft_xc'] * (1 - iqa_r_a['coeff_mix'])
                    iqa_r_a['t_exch'] = iqa_r_a['t_exch'] * iqa_r_a['coeff_mix']
                elif xc_type != 'LDA':
                    iqa_r_a.pop('t_exch')
                    iqa_r_a.pop('at_exch')

    # Note: we could add printing as optional since we return the dict at the end
    if APOSTBase.unrestr:
        out_summary([iqa_r_a, iqa_r_b], at_nums, APOSTBase.energy)
    else:
        out_summary([iqa_r_a], at_nums, APOSTBase.energy)

    return (iqa_r_a, iqa_r_b)

