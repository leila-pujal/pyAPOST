# IQA module Single Determinant Hilbert space

import numpy as np

import multiprocessing as mp
from multiprocessing import set_start_method, get_start_method

from pyAPOST.base import APOSTBase
from pyAPOST.energy_analyt import compute_energy_terms
from pyAPOST.iqa_sd import nn_energy


def _ee_int(coeff, eri, eta_m, occs, ind, part='K78', term='Coul'):
    """"WIP: Auxiliar function to parallelize EE coulomb and exchange energy"""

    ia, ib, ee_el = ind
    print('Element:', ee_el)
    print('Term:', term)
    print('DECOMPOSITION:', part)
    j_eta_at = 0
    k_eta_at = 0
    # eri_eta = np.einsum('ax,abcd, bz->xzcd', eta_m[ia], eri, eta_m[ib], optimize=True)
    print(term)
    print(bool(term == 'Coul' or term=='All'))
    if term == 'Coul' or term=='All':
        # print('COULOMB in AUX')
        if part == 'K78':
            eri_eta = np.einsum('ax,abcd, bz->xzcd', eta_m[ia], eri, eta_m[ib], optimize=True)
            j_eta_at1 = np.einsum('ia,bj,abcd ->ijcd', coeff.T, coeff, eri_eta, optimize=True)
            j_eta_at2 = np.einsum('kc,dl,ijcd ->ijkl', coeff.T, coeff, j_eta_at1, optimize=True)
            j_eta_at = 0.5 * np.einsum('ij,kl,ijkl ->', np.diag(occs), np.diag(occs), j_eta_at2, optimize=True)
        elif part in ['IY99', 'M03', 'NK05']:
            eri_eta = np.einsum('ax,abcd, cz->xbzd', eta_m[ia], eri, eta_m[ib], optimize=True)
            j_eta_at1 = np.einsum('ia,bj,abcd ->ijcd', coeff.T, coeff, eri_eta, optimize=True)
            j_eta_at2 = np.einsum('kc,dl,ijcd ->ijkl', coeff.T, coeff, j_eta_at1, optimize=True)
            j_eta_at = 0.5 * np.einsum('ij,kl,ijkl ->', np.diag(occs), np.diag(occs), j_eta_at2, optimize=True)
    if term=='Exch' or term=='All':
        # print('EXCH in AUX')
        if part in ['K78', 'IY99']:
            eri_eta = np.einsum('ax,adcb, bz->xdcz', eta_m[ia], eri, eta_m[ib], optimize=True)
            k_eta_at1 = np.einsum('ix,zj,xdcz ->idcj', coeff.T, coeff, eri_eta, optimize=True)
            k_eta_at2 = np.einsum('kc,dl,idcj ->ilkj', coeff.T, coeff, k_eta_at1, optimize=True)
            k_eta_at = 0.25 * np.einsum('ij,kl,ilkj ->', np.diag(occs), np.diag(occs), k_eta_at2, optimize=True)
        elif part in ['M03', 'NK05']:
            eri_eta = np.einsum('ax,adcb, cz->xdzb', eta_m[ia], eri, eta_m[ib], optimize=True)
            k_eta_at1 = np.einsum('ix,bj,xdzb ->idzj', coeff.T, coeff, eri_eta, optimize=True)
            k_eta_at2 = np.einsum('kz,dl,idzj ->ilkj', coeff.T, coeff, k_eta_at1, optimize=True)
            k_eta_at = 0.25 * np.einsum('ij,kl,ilkj ->', np.diag(occs), np.diag(occs), k_eta_at2, optimize=True)

    return j_eta_at,k_eta_at, ind


def kin_energy(at_nums, mol, coeff, occs, eta_m, transf_m=None):
    r"""WIP:Compute IQA's kinetic energy for Hilbert space partitions.

            Returns
            -------
            t_kin: np.array()
                Total value for kinetic energy (from sum of atomic values)
            at_kin: np.array(natoms)
                Atomic kinetic energies.

            """

    # get the integral from libcint through the mol calling function intor
    print('DEV KINETIC')
    kin_ao = mol.intor("int1e_kin")
    print(kin_ao.shape)
    # in AO basis
    k=0
    print('Shape occ coeff', coeff.shape)
    # Below slow implementation
    # for mu in range(coeff.shape[0]):
    #     for nu in range(coeff.shape[0]):
    #         k += dm[mu][nu] * kin_ao[nu][mu]
    # k = np.trace(np.dot(dm, kin_ao))
    # print('total kin', k)
    # Transform AO --> MO with eta matrices. Partition already included.
    # _AO_frags = [np.linalg.multi_dot((eta_slices, APOSTBase.ovlp, eta_slices)) for eta_slices
    #              in APOSTBase.part.eta_frags]
    print('ETA shape', eta_m[0].shape)

    # DEV: Below for orthonormal AO basis
    # Todo: See if  we can combine mulliken and AO orthogonal basis
    # Transform T into orthonormal AO basis
    if transf_m:
        t = transf_m[0]
        t_inv = transf_m[1]
        # check pop
        # todo: change variable name to the one used for mulliken and return
        orth_coeff = np.einsum('la,ai -> li', t, coeff)
        # pop_a = np.trace(np.linalg.multi_dot((orth_coeff.T, eta_m[0], orth_coeff))*occs)
        print(eta_m[0].shape)
        eta_orth = np.linalg.multi_dot((t.T, eta_m[0], t))
        print(eta_orth.shape)
        # pop_a = np.trace(np.linalg.multi_dot((orth_coeff.T, eta_orth, orth_coeff))*occs)
        pop_a = np.trace(np.linalg.multi_dot((coeff.T, eta_orth, coeff))*occs)
        print('check pop a', pop_a)


        # Transform kinetic operator matrix
        # for lowdin t_inv == t_inv.T but not for meta_lowdin or nao
        t_kin_ao_orth = np.linalg.multi_dot((t_inv.T, kin_ao, t_inv))
        # Transform MO-LCAO coefficients into orthogonal AO basis
        orth_coeff = np.einsum('la,ai -> li', t, coeff)
        # Generate atomic contributions
        t_kin_ao_orth_a = [np.dot(t_kin_ao_orth, eta_slice) for eta_slice in eta_m]
        t_kin_a = [np.trace(np.linalg.multi_dot((orth_coeff.T, kin_orth_a, orth_coeff))*occs) for kin_orth_a in t_kin_ao_orth_a]
        print('KINETIC ATOMIC MATRIX', t_kin_a)
        print('TOTAL FROM ATOMIC', np.sum(t_kin_a))

    else:
        # DEV: Below is for mulliken
        kin_mat = np.zeros((len(at_nums), len(at_nums)))
        for ia in range(len(at_nums)):
            for ib in range(len(at_nums)):
                kin_ao_s = np.linalg.multi_dot((eta_m[ia], kin_ao, eta_m[ib]))
                # kin_ao_s = np.dot(kin_ao, eta_m[ia])
                kin_at = np.trace(np.linalg.multi_dot((coeff.T, kin_ao_s, coeff)) * occs)
                if ia!=ib:
                    kin_mat[ia, ib] = 2*kin_at
                else:
                    kin_mat[ia, ib] = kin_at

        print('KINETIC ATOMIC MATRIX', kin_mat)
        print('TOTAL FROM ATOMIC', np.sum(kin_mat))
    return np.sum(np.triu(kin_mat)), kin_mat


def en_energy(at_nums, mol, coeff, occs, eta_m, transf_m=None, part='K78'):
    r"""Compute IQA's electron-nuclear attraction energy for Hilbert space partitionings.
    """
    print('DEV EN for Hilbert space')
    # Same as with the electrostatic potential we need to specify we want to use the
    # atomic coordinates of atoms. If not it uses the origin.
    v_int_ao = np.zeros((mol.natm, coeff.shape[0], coeff.shape[0]))
    for iz,z in enumerate(at_nums):
        print(f'Check at coords at {iz +1}', mol.atom_coords()[iz, :])
        with mol.with_rinv_at_nucleus(iz):
            v_int_ao[iz] = -mol.atom_charge(iz)* mol.intor('int1e_rinv')

    if transf_m:
        en_at_m = np.zeros((len(at_nums), len(at_nums), len(at_nums)))
        print('AO orth EN')
        t = transf_m[0]
        t_inv = transf_m[1]
        for ia in range(len(at_nums)):
            for ib in range(len(at_nums)):
                # Transform kinetic operator matrix
                v_int_ao_orth = np.linalg.multi_dot((t_inv.T, v_int_ao[ia], t_inv))
                # Transform MO-LCAO coefficients into orthogonal AO basis
                orth_coeff = np.einsum('la,ai -> li', t, coeff)
                # Generate atomic contributions
                en_at_1 = np.dot(v_int_ao_orth, eta_m[ib])
                en_at = np.trace(np.linalg.multi_dot((orth_coeff.T, en_at_1, orth_coeff)) * occs)
                en_at_m[ia, ib] = en_at
    else:
        print('Mulliken EN')
        if part == 'K78':
            print('Kollmar 1978')
            en_at_m = np.zeros((len(at_nums), len(at_nums), len(at_nums)))
            for c in range(len(at_nums)):
                for ia in range(len(at_nums)):
                    for ib in range(len(at_nums)):
                        en_at_1 = np.linalg.multi_dot((eta_m[ia], v_int_ao[c], eta_m[ib]))
                        en_at = np.trace(np.linalg.multi_dot((coeff.T, en_at_1, coeff)) * occs)
                        if ia != ib:
                            en_at_m[c, ia, ib] =  2*en_at
                        else:
                            en_at_m[c, ia, ib] =  en_at
            print('COMPLETE ATOMIC EN MATRIX')
            print(en_at_m)
            # print(en_ao_m.shape)
            # print(np.triu(en_ao_m))
            print('ATOMIC EN ', np.sum(np.triu(en_at_m), axis=2))
            print('EFFECTIVE ATOMIC EN', np.sum(np.triu(en_at_m), axis=(1, 2)))
            print('TOTAL ATOMIC EN', np.sum(np.triu(en_at_m)))
            print('END OF EN ENERGY')
            return en_at_m, np.sum(np.triu(en_at_m), axis=(1, 2)), np.sum(np.triu(en_at_m))

        elif part in ['IY99', 'M03', 'SS06', 'NK05']:
            print('Ichikawa and Yoshida 1999. Also used for Mayer-03')
            en_at_m = np.zeros((len(at_nums), len(at_nums)))
            for ia in range(len(at_nums)):
                for ib in range(len(at_nums)):
                    # en_at_1 = np.linalg.multi_dot((eta_m[ia], v_int_ao[c], eta_m[ib]))
                    # en_at = np.trace(np.linalg.multi_dot((coeff.T, en_at_1, coeff)) * occs)
                    if ia != ib:
                        en_at_1 = np.linalg.multi_dot((eta_m[ia], v_int_ao[ib]))
                        en_at_ab1 = np.trace(np.linalg.multi_dot((coeff.T, en_at_1, coeff)) * occs)

                        en_at_2 = np.linalg.multi_dot((eta_m[ib], v_int_ao[ia]))
                        en_at_ab2 = np.trace(np.linalg.multi_dot((coeff.T, en_at_2, coeff)) * occs)

                        en_at_m[ia, ib] =   en_at_ab2 + en_at_ab1
                        # en_at_m[ia, ib] =   en_at_ab2
                    else:
                        en_at_1 = np.linalg.multi_dot((eta_m[ia], v_int_ao[ia]))
                        en_at = np.trace(np.linalg.multi_dot((coeff.T, en_at_1, coeff)) * occs)
                        en_at_m[ia, ib] =  en_at

            print('COMPLETE ATOMIC EN MATRIX')
            print(en_at_m)
            # print(en_ao_m.shape)
            # print(np.triu(en_ao_m))
            print('EFFECTIVE ATOMIC EN', np.sum(np.triu(en_at_m), axis=1))
            # print('EFFECTIVE ATOMIC EN', np.sum(en_at_m, axis=1))
            print('TOTAL FROM EFFECTUVE', np.sum(np.triu(en_at_m), axis=(0,1)))
            # print('TOTAL FROM EFFECTUVE', np.sum(en_at_m, axis=(0,1)))
            print('TOTAL ATOMIC EN', np.sum(np.triu(en_at_m)))
            # print('TOTAL ATOMIC EN', np.sum(en_at_m))
            print('END OF EN ENERGY')
            return en_at_m, np.sum(np.triu(en_at_m), axis=1), np.sum(np.triu(en_at_m))

def ee_hf_energy(at_nums, mol, coeff, occs, eta_m, transf_m=None, basis='MO', part='K78', coul=True, exch=True):
    """WIP: Electron-electron repulsion energy (Coulomb/Exchange) for Hilbert
    space partitions
    This should only give the effective atomic values?

    """

    if transf_m:
        raise NotImplementedError("EE for orthogonal AO basis partitioning not yet implemented")
    else:
        # ERI in 1122(J) and 1221(K) notation
        print('MULLIKEN EE COULOMB AND HF EXCHANGE')
        eri = mol.intor('int2e')
        # Initialize matrices
        j_mat = np.zeros((len(at_nums), len(at_nums)))
        k_mat = np.zeros((len(at_nums), len(at_nums)))
        if basis=='AO':
            # Construct P
            dm = np.dot((coeff[:, :occs.shape[0]] * occs), coeff[:, :occs.shape[0]].T)
            # Below totals
            # j = 0.5*np.einsum('ij, ijkl, kl ->', dm, eri,dm)
            # k = -0.25*np.einsum('ik, ijkl, jl ->', dm, eri,dm)
            # For now hard coded with indices  1 and 2  of the 4-index  eri for 1122 electrons
            for ia in range(len(at_nums)):
                for ib in range(len(at_nums)):
                    if coul:
                        j_eta = np.einsum('ax,abcd, bz->xzcd', eta_m[ia], eri, eta_m[ib], optimize=True)
                        j_eta_at = 0.5 * np.einsum('xz, xzcd, cd ->', dm, j_eta, dm, optimize=True)
                        if ia != ib:
                            j_mat[ia, ib] = 2 * j_eta_at
                        else:
                            j_mat[ia, ib] = j_eta_at
                    if exch:
                        k_eta = np.einsum('ax,adcb, bz->xdcz', eta_m[ia], eri, eta_m[ib], optimize=True)
                        k_eta_at = 0.25 * np.einsum('xz, xdcz, cd->', dm, k_eta, dm, optimize=True)
                        if ia != ib:
                            k_mat[ia, ib] = 2*k_eta_at
                        else:
                            k_mat[ia, ib] = k_eta_at
            print('CHECK AO')
            print(j_mat)
            print(k_mat)
            print(np.sum(np.triu(j_mat)))
            print(np.sum(np.triu(k_mat)))
            assert 5 == 6

        elif basis=='MO':
            # Total value
            # j_mo1 = np.einsum('ia,bj,abcd ->ijcd', coeff.T, coeff, eri)
            # j_mo = np.einsum('kc,dl,ijcd ->ijkl', coeff.T, coeff, j_mo1)
            # print('TOTAL J FROM Tr of MOs', 0.5 * np.einsum('ij,kl,ijkl ->', np.diag(occs), np.diag(occs), j_mo))
            # print('TOTAL K FROM Tr of MOs', 0.25 * np.einsum('il,kj,ijkl ->', np.diag(occs), np.diag(occs), j_mo))
            print('TOTAL NUMBER OF MATRIX ELEMENTS', (len(at_nums)*(len(at_nums)+1))/2)
            ee_el = 1
            if coul and not exch:
                pool_coul = mp.Pool(processes=4)
            if exch and not coul:
                pool_x = mp.Pool(processes=4)
            else:
                pool_all = mp.Pool(processes=4)
            segments_coul = []
            segments_x = []
            segments_all = []
            for ia in range(len(at_nums)):
                for ib in range(ia+1):
                    if coul and not exch:
                        segments_coul.append((coeff, eri, eta_m, occs, (ia, ib, ee_el), part, 'Coul'))
                    if exch and not coul:
                        segments_x.append((coeff, eri, eta_m, occs, (ia, ib, ee_el), part, 'Exch'))
                    else:
                        segments_all.append((coeff, eri, eta_m, occs, (ia, ib, ee_el), part, 'All'))
                    ee_el += 1

            if coul and not exch:
                print('CALL FOR COULOMB')
                results_coul = pool_coul.starmap_async(_ee_int, segments_coul).get()
                pool_coul.close()
                pool_coul.join()
                for out in results_coul:
                    j_eta_at = out[0]
                    ia, ib, ee_el = out[-1]
                    if ia != ib:
                        j_mat[ia, ib] = 2 * j_eta_at
                        j_mat[ib, ia] = j_mat[ia, ib]
                    else:
                        j_mat[ia, ib] = j_eta_at

            if exch and not coul:
                print('CALL FOR EXCHANGE')
                results_x = pool_x.starmap_async(_ee_int, segments_x).get()
                pool_x.close()
                pool_x.join()
                for out in results_x:
                    k_eta_at = out[1]
                    ia, ib, ee_el = out[-1]
                    if ia != ib:
                        k_mat[ia, ib] = 2 * -k_eta_at
                        k_mat[ib, ia] = k_mat[ia, ib]
                    else:
                        k_mat[ia, ib] = -k_eta_at
            else:
                print('CALL FOR COULOMB AND EXCHANGE')
                results_all = pool_all.starmap_async(_ee_int, segments_all).get()
                pool_all.close()
                pool_all.join()
                for out in results_all:
                    j_eta_at = out[0]
                    k_eta_at = out[1]
                    ia, ib, ee_el = out[-1]
                    if ia != ib:
                        j_mat[ia, ib] = 2 * j_eta_at
                        j_mat[ib, ia] = j_mat[ia, ib]
                        k_mat[ia, ib] = 2 * -k_eta_at
                        k_mat[ib, ia] = k_mat[ia, ib]
                    else:
                        k_mat[ia, ib] = -k_eta_at
                        j_mat[ia, ib] = j_eta_at


            print('COULOMB MATRIX')
            print(j_mat)
            print('EXCHANGE MATRIX')
            print(k_mat)
            print('TOTAL COULOMB FROM ATOMIC MATRIX')
            print(np.sum(np.triu(j_mat)))
            print('TOTAL EXCHANGE FROM ATOMIC MATRIX')
            print(np.sum(np.triu(k_mat)))
            print('END OF EE HILBERT')

    return np.sum(np.triu(j_mat)), j_mat, np.sum(np.triu(k_mat)), k_mat



def iqa_sd_hilbert(APOSTBase, part='K78'):
    """WIP: Interacting quantum atoms partition energy quantities for SD wavefunctions with
    Hilbert space partitiong"""


    print('DEV IQA SD HILBERT SPACE')
    print('PART', part)
    # Molecule info
    at_coords = APOSTBase.at_coords
    at_nums = np.array(APOSTBase.at_nums)

    thresh = 1e-8
    occ_coeff_a = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
    occ_coeff_b = APOSTBase.mo_coeff_b[:, APOSTBase.occ_b > thresh]
    occ_a = APOSTBase.occ_a[APOSTBase.occ_a > thresh]
    occ_b = APOSTBase.occ_b[APOSTBase.occ_b > thresh]
    dm_a = APOSTBase.dm_a
    dm_b = APOSTBase.dm_b
    # Diagonal matrices for sd
    occ_mat_a = np.diag(occ_a)
    occ_mat_b = np.diag(occ_b)

    print('occs a', occ_a)
    print('occs b', occ_b)

    # Initialize dicts
    iqa_h_a = {}
    iqa_h_b = {}
    # Compute analytic terms
    ee_terms = compute_energy_terms(APOSTBase, out=False)
    iqa_h_a['t_nn'] = ee_terms['nn']

    if APOSTBase.unrestr:
        iqa_h_b['kin_analyt'] = ee_terms['kin']
        iqa_h_b['en_analyt'] = ee_terms['en']
        iqa_h_b['coul_analyt'] = ee_terms['coul']
        iqa_h_b['exch_analyt'] = ee_terms['x_b']
        iqa_h_a['exch_analyt'] = ee_terms['x_a']
    else:
        iqa_h_a['kin_analyt'] = ee_terms['kin']
        iqa_h_a['en_analyt'] = ee_terms['en']
        iqa_h_a['coul_analyt'] = ee_terms['coul']
        iqa_h_a['exch_analyt'] = ee_terms['x_a']
        iqa_h_a['ee_analyt'] = ee_terms['ee']

    # DEV: For now only HF calculation
    if APOSTBase.dft:
        raise NotImplementedError("DFT calculation not implemented yet")
    else:
        if APOSTBase.unrestr:
            raise NotImplementedError("UHF calculation  not implemented yet")
        else:
            if APOSTBase.part.part_name == 'mulliken':
                # kinetic
                print('Mulliken for RHF calculation')
                # iqa_h_b['t_kin'], iqa_h_b['at_kin']= kin_energy(at_nums, APOSTBase._mol, occ_coeff_b, occ_b, APOSTBase.part.eta_frags)
                iqa_h_a['t_kin'], iqa_h_a['at_kin'] = kin_energy(at_nums, APOSTBase._mol, occ_coeff_a, occ_a, APOSTBase.part.eta_frags)
                # EN
                # iqa_h_b['at_en'], iqa_h_b['eff_at_kin'], iqa_h_b['t_en'] = en_energy(at_nums, APOSTBase._mol, occ_coeff_b, occ_b,  APOSTBase.part.eta_frags, part=part)
                iqa_h_a['at_en'], iqa_h_a['eff_at_kin'], iqa_h_a['t_en'] = en_energy(at_nums, APOSTBase._mol, occ_coeff_a, occ_a, APOSTBase.part.eta_frags, part=part)
                # EE
                if part == 'SS06':
                    iqa_h_a['t_coul'], iqa_h_a['at_coul'],t_x_iy99, at_x_iy99 = ee_hf_energy(at_nums, APOSTBase._mol, occ_coeff_a, occ_a, APOSTBase.part.eta_frags, part='IY99')
                    _, _, t_x_mo3,at_x_mo3 = ee_hf_energy(at_nums, APOSTBase._mol, occ_coeff_a, occ_a, APOSTBase.part.eta_frags, part='M03', coul=False, exch=True)
                    iqa_h_a['t_exch'] = 0.5*(t_x_iy99 + t_x_mo3)
                    iqa_h_a['at_exch'] = 0.5*(at_x_iy99 + at_x_mo3)
                else:
                    iqa_h_a['t_coul'], iqa_h_a['at_coul'],iqa_h_a['t_exch'], iqa_h_a['at_exch'] = ee_hf_energy(at_nums, APOSTBase._mol, occ_coeff_a, occ_a, APOSTBase.part.eta_frags, part=part)


                # Compute NN
                iqa_h_a['nn_t'], iqa_h_a['nn_at'] = nn_energy(at_nums, at_coords, all_terms=True)
                # Kollmar 78
                # Intra-atomic terms
                # Total from terms
                print('kinetic analyt/from pieces', ee_terms['kin'],iqa_h_a['t_kin'] )
                print('En analyt', ee_terms['en'], iqa_h_a['t_en'])
                print('Coulomb analyt', ee_terms['coul'], iqa_h_a['t_coul'])
                print('Exchange analyt', ee_terms['x_a'], iqa_h_a['t_exch'])
                # print('NN AT terms', iqa_h_a['nn_at'])
                print('NN ', ee_terms['nn'], iqa_h_a['nn_t'])

                print('TOTAL FROM TERMS', iqa_h_a['t_kin'] +  iqa_h_a['t_en'] + iqa_h_a['t_coul'] + iqa_h_a['t_exch'] + iqa_h_a['nn_t'])
                print('APOSTBase.energy', APOSTBase.energy)
                # INTRA ATOMIC terms
                intra_atomic_terms = np.zeros((len(at_nums)))
                inter_atomic_terms = np.zeros((len(at_nums),len(at_nums)))


                for ia in range(len(at_nums)):
                    if part == 'K78':
                        aa = 0.5*np.sum(iqa_h_a['nn_at'], axis=1)[ia]+ iqa_h_a['at_kin'][ia][ia] + np.sum(iqa_h_b['at_en'], axis=0)[ia][ia] + iqa_h_a['at_coul'][ia][ia] + iqa_h_a['at_exch'][ia][ia]
                    elif part in ['IY99', 'M03', 'SS06', 'NK05']:
                        # aa = iqa_h_a['nn_at'][ia]+ iqa_h_a['at_kin'][ia][ia] + iqa_h_b['at_en'][ia][ia] + iqa_h_a['at_coul'][ia][ia] + iqa_h_a['at_exch'][ia][ia]
                        aa = iqa_h_a['at_kin'][ia][ia] + iqa_h_a['at_en'][ia][ia] + iqa_h_a['at_coul'][ia][ia] + iqa_h_a['at_exch'][ia][ia]
                    intra_atomic_terms[ia] = aa
                for ia in range(len(at_nums)):
                    for ib in range(len(at_nums)):
                        if ia != ib:
                            if part=='K78':
                                ab = np.sum(iqa_h_a['at_en'], axis=0)[ia][ib] + iqa_h_a['at_kin'][ia][ib] + iqa_h_a['at_coul'][ia][ib] + iqa_h_a['at_exch'][ia][ib]
                            elif part in ['IY99', 'M03', 'SS06', 'NK05']:
                                ab = iqa_h_a['nn_at'][ia][ib] + iqa_h_a['at_en'][ia][ib] + iqa_h_a['at_kin'][ia][ib] + iqa_h_a['at_coul'][ia][ib] + iqa_h_a['at_exch'][ia][ib]
                            inter_atomic_terms[ia][ib] = ab

                if part == 'NK05':
                    extra_aa = np.zeros((len(at_nums)))
                    extra_ab = np.zeros((len(at_nums), len(at_nums)))
                    for ia in range(len(at_nums)):
                        for ib in range(len(at_nums)):
                            if ia != ib:
                                alpha_ab = 0.5
                                extra_aa[ia] += alpha_ab * iqa_h_a['at_kin'][ia][ib] +  iqa_h_a['at_exch'][ia][ib]
                                extra_ab[ia][ib] = alpha_ab * iqa_h_a['at_kin'][ia][ib] +  iqa_h_a['at_exch'][ia][ib]

                    for ia in range(len(at_nums)):
                        intra_atomic_terms[ia] += 0.5*extra_aa[ia]
                        for ib in range(len(at_nums)):
                            inter_atomic_terms[ia][ib] -= extra_ab[ia][ib]






                print('INTRA ATOMIC TERMS')
                print(intra_atomic_terms)
                print('INTER ATOMIC TERMS')
                print('Hartree', inter_atomic_terms)
                print('Kcal', inter_atomic_terms*23.06*27.212)
                # Check obtained effective and total
                # effective e_a = aa + 0.5 ab
                eff_at = np.zeros((len(at_nums)))
                for ia in range(len(at_nums)):
                    e_a = intra_atomic_terms[ia] + 0.5*np.sum(inter_atomic_terms[ia, :])
                    eff_at[ia] = e_a
                print('EFF ATOMIC TERMS')
                print(eff_at)
                print('TOTAL from effective')
                print(np.sum(eff_at))

            else:
                t_kin_b, at_kin_b = kin_energy(at_nums, APOSTBase._mol, occ_coeff_b, occ_b, APOSTBase.part.eta_frags, transf_m=[APOSTBase.part.T, APOSTBase.part.T_inv])
                t_en_b, at_en_b = en_energy(at_nums, APOSTBase._mol, occ_coeff_b, occ_b, APOSTBase.part.eta_frags, transf_m=[APOSTBase.part.T, APOSTBase.part.T_inv])
                assert 5 == 6

    return (iqa_h_a, iqa_h_b)


