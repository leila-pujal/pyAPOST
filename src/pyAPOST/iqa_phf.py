# IQA module Post Hartree-Fork
import copy as copy
import time as time

import multiprocessing as mp
from multiprocessing import set_start_method, get_start_method
import numba as numba
from numba import njit

import numpy as np
import logging

from pyscf.dft.numint import eval_ao, eval_rho

# from pyAPOST.bond_order_wip1 import compute_bod
from pyAPOST.iqa_sd import _coul_integral,_exchange_integral,  iqa_atomic, ee_hf_energy_interatomic
from pyAPOST.utils import out_summary_phf
from pyAPOST.bond_order import  improved_bod, compute_bod

#Note:
# Some of the functions, specially the ones dealing with ee integrals have significant amount of arguments
# PySCF and Horton I think they use some type of internal cache of properties that the can access
# This in an easy way would be to pass the APOSTBase object to these functions (we pass it already to the main functions)

@njit(nopython=True)
# def _two_elect_integral_phf(id_p1, p1, dm1, mo_ovlp_alt1, mo_ovlp_alt2, a_grid_coords, grid_coords2, qw, aim_w1, aim_w2, rij_v=1e-5):
def _two_elect_integral_phf(id_p1, p1, mo_ovlp_alt1, mo_ovlp_alt2, atgrid_idx_b, a_grid_coords,b_grid_coords, wq, aim_w1, aim_w2, occ, rij_v=1e-5):

    """ WIP: Auxiliary function for parallel calculation of 2e integrals from monadic diagonalized cumulant"""

    rij = a_grid_coords[id_p1, :] - b_grid_coords
    rij = np.sqrt(np.sum(np.square(rij), axis=-1))

    ee = np.zeros(mo_ovlp_alt2[0, atgrid_idx_b].shape[0])
    # With the cumulant the term we get is determined before
    for i in range(mo_ovlp_alt1.shape[0]):
        ee += occ[i]* mo_ovlp_alt1[i, p1] * mo_ovlp_alt2[i, atgrid_idx_b ]
    ee = ee / rij
    ee[rij < 1e-9] = 0
    ee_t1 = np.sum(ee * wq[atgrid_idx_b] * aim_w2[atgrid_idx_b])
    ee_t = ee_t1 * wq[p1] * aim_w1[p1]

    return ee_t, rij[rij<rij_v].shape[0]


def ee_terms_phf(at_nums, norb, dm1, dm2, grid_coords1, grid_coords2, mo_grid1, mo_grid2, wq, aim_w1, aim_w2, atgrid_idx, cumul='xc', rij_v=1e-5, ncpus=1, u12=None, allw=None):
    """WIP: This is a dev function to test decomposing electron-electron repulsion energy from post-hartree fock method into
    coulomb, exchange and correlation(cumulant).
    """

    print('DEV ee_terms_phf')
    start = time.time()
    cumulant_mat = np.zeros((len(at_nums), len(at_nums)))

    if u12 is not None:
        print('u12')
        print(type(u12))
        print(u12.shape)
        u12[u12 == -0.0] = 0
        u12[(u12 < -1e-45) & (u12 > -1e-5)] = 0
        print(u12)
        # assert 5 == 6
        ovlp_a = np.sum(mo_grid1[None, :norb] * allw * mo_grid1[ :norb, None, :] * wq * aim_w1, axis=-1)
        print(np.trace(u12))
        print('Check u', np.trace(np.dot(u12, ovlp_a)))

    # Now we compute the cumulant. for this we build spdm1 and psdm1
    # norb = total orbitals CAS
    # For now APOST names
    sfdm1 = np.zeros((norb, norb)) #spinfree density
    psdm1 = np.zeros((norb, norb)) # spin density
    for i in range(norb):
        for j in range(norb):
            # print('ij',i,j,i*2, j*2,  i*2+1, j*2+1)
            sfdm1[i,j] = dm1[i*2, j*2] + dm1[i*2+1, j*2+1]
            psdm1[i,j] = dm1[i*2, j*2] - dm1[i*2+1, j*2+1]
            # print(dm1[i*2, j*2], dm1[i*2+1, j*2+1], sfdm1[i,j])
    # assert 5 == 6
    #COMPUTE ALL CUMULANTS
    xc_cumul = np.zeros((norb, norb, norb, norb))
    x_cumul = np.zeros((norb, norb, norb, norb))
    gx_cumul = np.zeros((norb, norb, norb, norb))
    c_cumul = np.zeros((norb, norb, norb, norb))
    c_gx_cumul = np.zeros((norb, norb, norb, norb))
    print(dm2.shape)
    for i in range(norb):
        for j in range(norb):
            for k in range(norb):
                for l in range(norb):
                    # Todo: check if below code can be achieved without the 4 index loop
                    # xc cumulant--> without coulomb
                    coult =sfdm1[i,j] * sfdm1[k,l]
                    x1 = (sfdm1[i,k] * sfdm1[j,l])/2
                    x2 = (psdm1[i,k] * psdm1[j,l])/2
                    # x_unp = (u12[i,k] * u12[j,l])/2
                    # print('dev u12', u12[i,k], u12[j,l], u12[i,k] * u12[j,l])
                    x_unp = (np.sqrt(u12[i,k]) * np.sqrt(u12[j,l]))/2
                    xc_cumul[i,j,k,l] = dm2[i,j,k,l] - coult
                    c_cumul[i,j,k,l] = dm2[i,j,k,l] - coult + x1 + x2
                    x_cumul[i,j,k,l] = - x1 - x2
                    gx_cumul[i,j,k,l] = -x1 - x_unp
                    c_gx_cumul[i, j, k, l] = dm2[i, j, k, l] - coult + x1 + x_unp
                    # x_cumul[i,j,k,l] = - x2
                    # print('ijkl xc coult', i, j, k, l, xc_cumul[i,j,k,l], dm2[i,j,k,l], coult)
                    # print(i,j,k,l, coult, sfdm1[i,j], sfdm1[k,l])

    if cumul == 'xc':
        eigval_cumul, eigvec_cumul = _monadic_diag(xc_cumul)
    if cumul == 'c':
        eigval_cumul, eigvec_cumul = _monadic_diag(c_cumul)
    if cumul == 'c_gx':
        print('c_gx')
        eigval_cumul, eigvec_cumul = _monadic_diag(c_gx_cumul)
    if cumul == 'x':
        eigval_cumul, eigvec_cumul = _monadic_diag(x_cumul)
    if cumul == 'gx':
        print('gx')
        eigval_cumul, eigvec_cumul = _monadic_diag(gx_cumul)
    if cumul == 'dm2':
        eigval_cumul, eigvec_cumul = _monadic_diag(dm2)

    # After diagonalizing the condensed cumulant we obtained the transformation from indices of MO pairs to Natural pairs
    # Now we need to build up the density corresponding to the MO pairs and then use the eigenvectors to transform from
    # the MO pairs to the natural pairs

    # First we will generate the fij here because we have all the data exposed already but probably
    # we will move it to inside _monadic_diag
    # Initializing storage matrices
    p_mos = int(norb * (norb + 1) / 2)
    pairs_mo1 = np.zeros((p_mos, mo_grid1.shape[-1]))
    pairs_mo2 = np.zeros((p_mos, mo_grid2.shape[-1]))

    i2 = 0
    for i in range(norb):
        for j in range(i, norb):
            pairs_mo1[i2] = mo_grid1[i, :] * mo_grid1[j, :]
            pairs_mo2[i2] = mo_grid2[i, :] * mo_grid2[j, :]
            i2 += 1

    nat_pairs_mo1 = np.dot(eigvec_cumul.T, pairs_mo1)
    nat_pairs_mo2 = np.einsum('nm,mp->np ', eigvec_cumul.T, pairs_mo2)

    # Decomposition into atomic/interatomic terms
    for ia in range(len(at_nums)):
        # Get atomic grid for at A
        a_grid_coords = grid_coords1[atgrid_idx == ia, :]
        atgrid_idx_a = np.where(atgrid_idx == ia)[0]
        for ib in range(ia + 1):
            b_grid_coords = grid_coords2[atgrid_idx == ib, :]
            atgrid_idx_b = np.where(atgrid_idx == ib)[0]
            # id_p1 = index for the atomic grid a
            # p1 = index of the atomic grid a within the molecular grid
            # rij_small = 0
            pool = mp.Pool(processes=ncpus)
            segments = []
            for id_p1, p1 in enumerate(atgrid_idx_a): # atgrid_idx_a index inside molecular grid from a
                segments.append((id_p1, p1, nat_pairs_mo1, nat_pairs_mo2, atgrid_idx_b, a_grid_coords, b_grid_coords, wq, aim_w1, aim_w2, eigval_cumul, 1e-5))


            results = pool.starmap_async(_two_elect_integral_phf, segments).get()
            pool.close()
            pool.join()

            integral_cumulant = 0
            rij_small = 0
            for out in results:
                integral_cumulant += out[0]
                rij_small += out[-1]

            if ia != ib:
                cumulant_mat[ia, ib] += integral_cumulant
                cumulant_mat[ib, ia] = cumulant_mat[ia,ib]
            else:
                cumulant_mat[ia, ib] += integral_cumulant/2

    print(f'DEBUG: TOTAL NUMBER OF POINTS WITH Rij < {rij_v} : {rij_small}')
    print('SUMMARY TWO ELECTRON NUMERICAL INTEGRATION')
    print(f'*** CUMULANT {cumul.upper()} PART ***')
    print(cumulant_mat)
    print()
    end = time.time()
    print("computation time = ", end - start)

    return cumulant_mat

def coul_phf(at_nums, grid_coords1, grid_coords2, rho1, rho2, wq, aim_w1, aim_w2, atgrid_idx, rij_v, ncpus):
    """WIP: Interface function to prepare data to compute coulomb intra/inter atomic terms in parallel"""

    print('DEV Coulomb interatomic from RHO (Obtained from Natural orbitals) for CASSCF/CASCI')

    start = time.time()
    coul_mat = np.zeros((len(at_nums), len(at_nums)))



    print('CHECK integration')
    print('RHO1', np.sum(rho1*wq*aim_w1))
    print('RHO2', np.sum(rho2*wq*aim_w2))

    # ## ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
    if get_start_method(allow_none=True) is None:
        set_start_method('fork')

    logging.info(f"USING {ncpus} CPUS")
    for ia in range(len(at_nums)):
        # Get atomic grid for at A
        a_grid_coords = grid_coords1[atgrid_idx == ia, :]
        atgrid_idx_a = np.where(atgrid_idx == ia)[0]
        for ib in range(ia+1):
            b_grid_coords = grid_coords2[atgrid_idx == ib, :]
            atgrid_idx_b = np.where(atgrid_idx == ib)[0]
            # id_p1 = index for the atomic grid a
            # p1 = index of the atomic grid a within the molecular grid
            # rij_small = 0
            pool = mp.Pool(processes=ncpus)
            segments = []
            for id_p1, p1 in enumerate(atgrid_idx_a):
                segments.append((id_p1, p1, rho1, rho2, atgrid_idx_b, a_grid_coords, b_grid_coords, wq, aim_w1, aim_w2, rij_v))
            results = pool.starmap_async(_coul_integral, segments).get()
            pool.close()
            pool.join()

            # Add integral for r1
            integral_coul = 0
            rij_small = 0
            for out in results:
                integral_coul += out[0]
                rij_small += out[1]

            if ia != ib:
                coul_mat[ia, ib] += integral_coul
                coul_mat[ib, ia] = integral_coul
            else:
                coul_mat[ia, ib] += integral_coul/2

    print(f'DEBUG: TOTAL NUMBER OF POINTS WITH Rij < {rij_v} : {rij_small}')
    print('SUMMARY TWO ELECTRON NUMERICAL INTEGRATION')
    print(f'*** COULOMB PART ***')
    print(coul_mat)
    print()
    end = time.time()
    print("computation time = ", end - start)

    return coul_mat


def _monadic_diag(cumul):
    """WIP: Auxiliary function for monadic diag of cumulants"""

    # We need to group i,j and k,l indices into ij,kl
    # In APOST we loop over the 4 indices and save equivalent values of i,j,k,l into equivalent ij,kl
    # We start doing that to make it work but all the for loops we should check if they can be substituted by slicing
    norb = cumul.shape[0]
    # Pairs of MOs = n(n+1)/2 (norb2 in APOST)
    p_mos = int(norb*(norb+1)/2)
    print('TOTAL MO PAIRS: ', p_mos)
    cumul_cond =  np.zeros((p_mos, p_mos))
    i2 = 0
    for i in range(cumul.shape[0]):
        for j in range(i, cumul.shape[1]):
            k2 = 0
            for k in range(cumul.shape[2]):
                for l in range(k, cumul.shape[3]):
                    cumul_cond[i2, k2] = cumul[i, j, k, l]
                    if i != j:
                        cumul_cond[i2,k2] += cumul[j,i,k,l]
                    if k != l:
                        cumul_cond[i2,k2] += cumul[i,j,l,k]
                    if i != j and k != l:
                        cumul_cond[i2,k2] += cumul[j,i,l,k]
                    k2 += 1
            i2 += 1
    print('check final i2 and k2', i2, k2)

    # Diagonalize condensed cumulant
    # e_values, e_vect = scipy.linalg.eigh(cumul_cond)
    e_values, e_vect = np.linalg.eigh(cumul_cond)

    e_values = np.flip(e_values, axis=-1)
    e_vect = np.flip(e_vect, axis=-1)

    return e_values, e_vect

#Note: For now only restricted CASSCF because the code is using the spinless dm2 but should be straightforward to extend to unrestricted


def iqa_phf(APOSTBase, ph2=0.162, rij_v=1e-5, correlation='classic', ncpus=1):
    """ WIP: Call function to obtain full decomposition into atomic/interatomic terms from a CAS calc"""

    # Gather information from molecule and integration grid
    at_coords = APOSTBase.at_coords
    at_nums = np.array(APOSTBase.at_nums)
    # Grid info
    wq = APOSTBase.part.quadrature_weights
    mol_w = APOSTBase.part.aim_w
    atgrid_idx = APOSTBase.part.grid_to_atom_idx

    # Density info
    dm1 = APOSTBase.dm1
    dm2 = APOSTBase.dm2

    # We store the dma/dmb because rotate the grid we need the to regenerate the aim_w, and because we are by default using TFVC
    # these need the density
    dm_a = APOSTBase.dm_a
    dm_b = APOSTBase.dm_b

    # Molecular orbital coefficients form the CAS. If CASSCF these are optimized and if CASCI they should be the same as the underlying SCF calc
    mo_coeff = APOSTBase._cas_mo_coeff
    no_coeff = APOSTBase.no_coeff
    no_coeff_ao_basis = np.einsum('bk,ki ->bi', mo_coeff, no_coeff)
    occ_no = APOSTBase._occ_no
    norb = APOSTBase._cas_data['ncas'] + APOSTBase._cas_data['ncore']

    # To perform the two electron integrals we need a rotated second grid
    # for each point in grid_1 (Atom A) integrate all points in grid_2 (Atom B).
    # We need to 1.Generate a second grid and 2.Atomic weights (quadrature) are the same for both grids as they depend only on the distance
    # 3. Regenerate AIM weights
    # Rotate specific angle in rads
    # 1. Get points and compute the polar angles theta and phi
    grid_coords1 = np.ascontiguousarray(copy.copy(APOSTBase.part.grid_coords))
    print(f'TOTAL NUMBER OF GRID1 POINTS: {grid_coords1.shape}')

    # First phb values
    # Calling method to rotate the grid coords
    # ph =  [(0,0.162), (0,0.18)]
    pha = 0
    # phb = 0.162
    phb = ph2
    # Note for the code:
    # APOST3D code: APOST3D generates the rotated grid not by rotating grid1 but by starting from scratch.
    # It generates distance and theta,fix for the same radial and angular points and then instead of passing 0 for pha and phb
    # it passes the specified value. This second call of the prenumint routine is only called inside enpart.f (and only if numerical 2e integration
    # is requested, not for semi analytical IQA). Once the theta and fix angles have been rotated the atomic coordinates are addedd
    # Here because we rely on the Grid pyscf machinery to get the points and the atomic(quadrature) weights before rotating the points
    # we substract the atomic coordinates, and then rotate them again. After that we generate a second set of AIM weights for the rotated grid but keep the same
    # atomi(quadrature) weights.
    # Note: Grids are computed for total density not spin separated
    grid_coords2, mol_w_2 = APOSTBase.part.rotate_grid(at_coords, at_nums, pha, phb, dm_a + dm_b, mol=APOSTBase._mol)
    print(f'DEV: ROTATED GRID AND AIM : {grid_coords2.shape}, {mol_w_2.shape}')

    eval_ao_grid1 = eval_ao(APOSTBase._mol,grid_coords1, deriv=0)
    eval_mo_grid1 = np.einsum('pa,ai->ip ', eval_ao_grid1, mo_coeff)
    eval_no_grid1 = np.einsum('pa,ai->ip ', eval_ao_grid1, no_coeff_ao_basis)

    print('CHECK TRANSFORMATION FROM MO TO NO IN AO BASIS')
    d2 = np.einsum('ai,ib,ab ->', no_coeff_ao_basis*occ_no, no_coeff_ao_basis.T, APOSTBase._ovlp)
    print('From NO in AO (P)', d2)

    eval_ao_grid2 = eval_ao(APOSTBase._mol, grid_coords2, deriv=0)
    eval_mo_grid2 = np.einsum('pa,ai->ip ', eval_ao_grid2, mo_coeff)
    eval_no_grid2 = np.einsum('pa,ai->ip ', eval_ao_grid2, no_coeff_ao_basis)

    # Computing rho1/rho2
    print(eval_mo_grid1.shape)
    rho1 = np.zeros(eval_ao_grid1.shape[0])
    rho2 = np.zeros(eval_ao_grid2.shape[0])
    # Note: We could also obtain rho with eval_rho function from pyscf
    for i in range(occ_no.shape[0]):
        rho1 += occ_no[i] * eval_no_grid1[i] * eval_no_grid1[i]
        rho2 += occ_no[i] * eval_no_grid2[i] * eval_no_grid2[i]
    print('CHECK RHO1', np.sum(rho1*wq*mol_w))
    print('CHECK RHO2', np.sum(rho2*wq*mol_w_2))

    # Initialize dictionary
    iqa_phf = {}
    # Compute one electron terms.
    iqa_a, iqa_b = iqa_atomic(APOSTBase, ee_hf=False, out=False)
    if APOSTBase.unrestr:
        # Because underlying SD calc is unrestricted or ROHF combining both alpha and beta results
        iqa_phf.update(iqa_a)
        iqa_phf.update(iqa_b)
        iqa_phf['exch_analyt']= iqa_a['exch_analyt'] + iqa_b['exch_analyt']
    else:
        iqa_phf.update(iqa_a)

    # Compute 2 electron terms
    iqa_phf['vee'] = APOSTBase.energy - iqa_phf['t_nn'] - iqa_phf['t_kin'] - iqa_phf['t_en']
    # Compute Coulomb from rho1/rho2.
    iqa_phf['at_coul']= coul_phf(at_nums, grid_coords1, grid_coords2, rho1 , rho2, wq, mol_w, mol_w_2, atgrid_idx, rij_v=rij_v, ncpus=ncpus)
    iqa_phf['t_coul'] = np.sum(np.triu(iqa_phf['at_coul']))

    if correlation == 'generalized':
        # iqa_phf['at_x'] = ee_terms_phf(at_nums, norb, dm1,  dm2, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, wq, mol_w, mol_w_2, atgrid_idx, cumul='x',  rij_v=rij_v, ncpus=ncpus, u12=u12, allw=APOSTBase.part.aim_w_allp[0])
        # iqa_phf['at_c'] = ee_terms_phf(at_nums, norb, dm1,  dm2, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, wq, mol_w, mol_w_2, atgrid_idx, cumul='c',  rij_v=rij_v, ncpus=ncpus, u12=u12, allw=APOSTBase.part.aim_w_allp[0])
        iqa_phf['at_x'] = ee_terms_phf(at_nums, norb, dm1, dm2, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, wq, mol_w, mol_w_2, atgrid_idx,cumul='gx', rij_v=rij_v, ncpus=ncpus, u12=u12, allw=APOSTBase.part.aim_w_allp[0])
        iqa_phf['at_c'] = ee_terms_phf(at_nums, norb, dm1, dm2, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, wq, mol_w, mol_w_2, atgrid_idx,cumul='c_gx', rij_v=rij_v, ncpus=ncpus, u12=u12, allw=APOSTBase.part.aim_w_allp[0])
        iqa_phf['t_x'] = np.sum(np.triu(iqa_phf['at_x']))
        iqa_phf['t_c'] = np.sum(np.triu(iqa_phf['at_c']))
    if correlation == 'classic':
        iqa_phf['at_x'] = ee_terms_phf(at_nums, norb, dm1,  dm2, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, wq, mol_w, mol_w_2, atgrid_idx, cumul='x',  rij_v=rij_v, ncpus=ncpus, u12=u12, allw=APOSTBase.part.aim_w_allp[0])
        iqa_phf['at_c'] = ee_terms_phf(at_nums, norb, dm1,  dm2, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, wq, mol_w, mol_w_2, atgrid_idx, cumul='c',  rij_v=rij_v, ncpus=ncpus, u12=u12, allw=APOSTBase.part.aim_w_allp[0])
        # iqa_phf['at_x'] = ee_terms_phf(at_nums, norb, dm1, dm2, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, wq, mol_w, mol_w_2, atgrid_idx,cumul='gx', rij_v=rij_v, ncpus=ncpus, u12=u12, allw=APOSTBase.part.aim_w_allp[0])
        # iqa_phf['at_c'] = ee_terms_phf(at_nums, norb, dm1, dm2, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, wq, mol_w, mol_w_2, atgrid_idx,cumul='c_gx', rij_v=rij_v, ncpus=ncpus, u12=u12, allw=APOSTBase.part.aim_w_allp[0])
        iqa_phf['t_x'] = np.sum(np.triu(iqa_phf['at_x']))
        iqa_phf['t_c'] = np.sum(np.triu(iqa_phf['at_c']))



    out_summary_phf([iqa_phf], at_nums, APOSTBase.energy, correlation=True)

    return iqa_phf

