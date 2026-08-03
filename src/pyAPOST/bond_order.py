# Bond order module
import numpy as np
import copy as copy

from pyscf.dft.numint import eval_ao
from pyscf import lo

from pyAPOST.base import APOSTBase
from pyAPOST.utils import out_bo


def bo_dens(at_nums, eval_mo_grid, wq, aim_w, atgrid_idx, aim_w_all, eval_mo_grad_grid=None, spin=0):
    """WIP: Returns the bond order density based on integration of A and B domains
    of r1 and r2 indices of the third order density matrix for SD"""


    print('DEV: bo dens based on the 3RDM')
    bo_d = np.zeros((len(at_nums), len(at_nums), eval_mo_grid.shape[1]))
    if isinstance(eval_mo_grad_grid, np.ndarray):
        grad_bo = np.zeros((len(at_nums), len(at_nums), 3, eval_mo_grad_grid.shape[2]))
    # Todo: Use smo
    for ia in range(len(at_nums)):
        for ib in range(len(at_nums)):
            sij_a = np.sum(eval_mo_grid[None, :] * aim_w_all[ia] * eval_mo_grid[:,None, :] * wq * aim_w, axis=-1)
            sij_b = np.sum(eval_mo_grid[None, :] * aim_w_all[ib] * eval_mo_grid[:,None, :] * wq * aim_w, axis=-1)
            dij = np.dot(sij_a, sij_b) + np.dot(sij_b, sij_a)
            for i in range(eval_mo_grid.shape[0]):
                for j in range(eval_mo_grid.shape[0]):
                    if spin == 1:
                        bo_d[ia, ib] += (eval_mo_grid[i] * dij[i,j] * eval_mo_grid[j])
                        if isinstance(eval_mo_grad_grid, np.ndarray):
                            grad_bo[ia, ib] += 2 * eval_mo_grad_grid[:, i, :] * dij[i, j] * eval_mo_grid[None, j, :]
                    else:
                        bo_d[ia, ib] += 2*(eval_mo_grid[i] * dij[i,j] * eval_mo_grid[j])
                        if isinstance(eval_mo_grad_grid, np.ndarray):
                            grad_bo[ia,ib] += 4*  eval_mo_grad_grid[:, i, :] * dij[i,j] * eval_mo_grid[None, j, :]

    if isinstance(eval_mo_grad_grid, np.ndarray):
        return bo_d, grad_bo
    else:
        return bo_d



def unpaired_elec(dij, at_nums, smo):
    """Function to compute matrix of unpaired electrons"""
    u = 2 * dij- np.dot(dij, dij)
    for idx1, at1 in enumerate(at_nums):
        if len(smo.shape) > 3:
            print('Warning: Unpaired for unrestricted post-HF methods not tested')
            smo_t = smo[0][idx1] + smo[1][idx1]
            print(f'Number of unpaired electrons atom {idx1+1} {at1}', np.trace(np.dot(u, smo_t)) )
        else:
            print(f'Number of unpaired electrons atom {idx1+1} {at1}', np.trace(np.dot(u, smo[idx1])))


    # Construct u^1/2
    u_eigval, u_eigvec = np.linalg.eigh(u)
    u12 = np.dot((u_eigvec * np.sqrt(u_eigval)), u_eigvec.T)
    return u, u12



def compute_bo(APOSTBase, bo_type=('all')):
    """WIP: General function to call bond orders"""

    # Convert to list
    if isinstance(bo_type, str):
        bo_type = (bo_type)

    # check populations
    print('DEV COMPUTE BO')

    # Deal with type of wavefunction
    if APOSTBase._cas or APOSTBase._fci:
        if APOSTBase._cas:
            norb = APOSTBase._cas_data['ncas'] + APOSTBase._cas_data['ncore']
        elif APOSTBase._fci:
            norb = APOSTBase._fci_data['norb']
        d = np.zeros((norb, norb))  # spinfree density
        ps = np.zeros((norb, norb))  # spin density
        d_alpha = APOSTBase._dm1_a
        d_beta = APOSTBase._dm1_b
        for i in range(norb):
            for j in range(norb):
                # print('ij',i,j,i*2, j*2,  i*2+1, j*2+1)
                # Todo: We should save this matrices because we use them for IQA-phf and bond orders
                d[i, j] = APOSTBase.dm1[i * 2, j * 2] + APOSTBase.dm1[i * 2 + 1, j * 2 + 1]
                ps[i, j] = APOSTBase.dm1[i * 2, j * 2] - APOSTBase.dm1[i * 2 + 1, j * 2 + 1]
    elif APOSTBase.unrestr:
        # UHF
        # Construct diagonal matrices with the occupations in the diagonal
        occ_a = APOSTBase.occ_a[APOSTBase.occ_a > 0]
        occ_b = np.zeros_like(APOSTBase.occ_a[APOSTBase.occ_a > 0])
        occ_b[:APOSTBase.occ_b[APOSTBase.occ_b > 0].shape[0]] = APOSTBase.occ_b[APOSTBase.occ_b > 0]
        d_alpha = np.diag(occ_a)
        d_beta = np.diag(occ_b)
        d = d_alpha + d_beta
        ps = d_alpha - d_beta
    else:
        occ_a = APOSTBase.occ_a[APOSTBase.occ_a > 0]
        d = np.diag(occ_a)

    # Compute Populations
    for idx1, at1 in enumerate(APOSTBase.at_nums):
        if APOSTBase.unrestr and not (APOSTBase.cas or APOSTBase._fci):
            pop_a = np.trace(np.dot(d_alpha, APOSTBase.smo[0, idx1]))
            pop_b = np.trace(np.dot(d_beta, APOSTBase.smo[1, idx1]))
            print(f"POP {at1}   {(pop_a+pop_b): .4f}/ {(pop_a): .4f},{(pop_b): .4f}")
        else:
            pop = np.trace(np.dot(d, APOSTBase.smo[idx1]))
            print(f"POP {at1}   {(pop): .4f}")

    # Compute  Bond orders
    bo_r = {}
    if 'mayer' in bo_type or 'all' in bo_type:
        # bo mayer
        bo_mayer = np.zeros((len(APOSTBase.at_nums), len(APOSTBase.at_nums)))
        for ia in range(len(APOSTBase.at_nums)):
            for ib in range(len(APOSTBase.at_nums)):
                if APOSTBase.unrestr and not (APOSTBase.cas or APOSTBase._fci):
                    ds_a_alpha = np.dot(d_alpha, APOSTBase.smo[0, ia])
                    ds_b_alpha = np.dot(d_alpha, APOSTBase.smo[0, ib])
                    ds_a_beta = np.dot(d_beta, APOSTBase.smo[1, ia])
                    ds_b_beta = np.dot(d_beta, APOSTBase.smo[1, ib])
                    if ia == ib:
                        bo_mayer[ia, ib] += (np.trace(np.dot(ds_a_alpha, ds_b_alpha)) + np.trace(np.dot(ds_a_beta, ds_b_beta)))
                    else:
                        bo_mayer[ia, ib] += 2*(np.trace(np.dot(ds_a_alpha, ds_b_alpha)) + np.trace(np.dot(ds_a_beta, ds_b_beta)))
                else:
                    ds_a = np.dot(d, APOSTBase.smo[ia])
                    ds_b = np.dot(d, APOSTBase.smo[ib])
                    bo = np.trace(np.dot(ds_a, ds_b))
                    if APOSTBase.cas or APOSTBase._fci:
                        ps_a = np.dot(ps, APOSTBase.smo[ia])
                        ps_b = np.dot(ps, APOSTBase.smo[ib])
                        bo =np.trace(np.dot(ds_a, ds_b)) +  np.trace(np.dot(ps_a, ps_b))

                    if ia == ib:
                        bo_mayer[ia, ib] += 0.5 * bo
                    else:
                        bo_mayer[ia, ib] += bo
        bo_r['mayer'] =  bo_mayer


    if 'improved' in bo_type or 'all' in bo_type:
        # Obtained unpaired electrons matrix
        # Todo: Decide what different 1-RDM we keep/have access to
        # Todo: For now we can only do the improved for Post-HF methods
        u, u12 = unpaired_elec(APOSTBase._dij, APOSTBase.at_nums, APOSTBase.smo)
        print(u12.shape)
        bo_improved = np.zeros((len(APOSTBase.at_nums), len(APOSTBase.at_nums)))
        for ia in range(len(APOSTBase.at_nums)):
            for ib in range(len(APOSTBase.at_nums)):
                ds_a = np.dot(d, APOSTBase.smo[ia])
                ds_b = np.dot(d, APOSTBase.smo[ib])
                u12_a = np.dot(u12, APOSTBase.smo[ia])
                u12_b = np.dot(u12, APOSTBase.smo[ib])
                bo = np.trace(np.dot(ds_a, ds_b)) + np.trace(np.dot(ds_a, ds_b)) + np.trace(np.dot(u12_a, u12_b)) + np.trace(np.dot(u12_a, u12_b))
                if ia == ib:
                    bo_improved[ia, ib] += 0.25*bo
                else:
                    bo_improved[ia, ib] += 0.5*bo

        bo_r['improved'] =  bo_improved
        # print(np.sum(np.triu(bo_improved)))



    if 'DI' in bo_type or 'all' in bo_type:
        print(APOSTBase.cas, APOSTBase._fci)
        if not APOSTBase.cas and not APOSTBase._fci:
            raise TypeError('LI/DI only available for CASSCF and FCI')

        bo_di = np.zeros((len(APOSTBase.at_nums), len(APOSTBase.at_nums)))
        # Obtain xc cumulant from DM2
        xc_cumul = np.zeros((norb, norb, norb, norb))
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l in range(norb):
                        coult = d[i, j] * d[k, l]
                        xc_cumul[i, j, k, l] = APOSTBase.dm2[i, j, k, l] - coult
        for ia in range(len(APOSTBase.at_nums)):
            for ib in range(len(APOSTBase.at_nums)):
                di = np.einsum('ijkl,ij,kl->', xc_cumul, APOSTBase.smo[ia], APOSTBase.smo[ib])
                if ia == ib:
                    bo_di[ia, ib] = -di
                else:
                    bo_di[ia, ib] = -2*di

        bo_r['DI'] =  bo_di

    out_bo(APOSTBase.at_nums, bo_r)


    return bo_r