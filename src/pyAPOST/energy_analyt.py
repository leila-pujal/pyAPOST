# Energy integrals
import numpy as np

from pyscf import ao2mo

from pyAPOST.bond_order import  unpaired_elec, compute_bo
from pyAPOST.utils import  out_ee_terms_analyt



def compute_energy_terms(APOSTBase, term='all', out=True):
    """Function to compute energy terms from AO integrals."""

    # Convert to list
    if isinstance(term, str):
        term = [term]

    if APOSTBase.unrestr:
        print('Compute global energy terms analytically: UNRESTRICTED CALCULATION')
        # P in atomic orbitals for alpha and beta
        dm_a = APOSTBase.dm_a
        dm_b = APOSTBase.dm_b
        dm = APOSTBase.dm_a + APOSTBase.dm_b
    else:
        print(APOSTBase.dm_a)
        dm = APOSTBase.dm_a

    # Check. What Mo coefficients do I need to pass to transform to ao2mo for unrestricted?
    # because for restricted there's only one set of Mo coeffs but here we have two and also
    # coulomb has to be done for all the density while exchange is only for a and be separately
    if ('Coul' in term) or ('Exch' in term) or ('ee' in term) or ('all' in term) or ('u12_ps' in term):
        if APOSTBase.sd:
            print('Single determinant')
            if APOSTBase.unrestr:
                occ_coeff_b = APOSTBase.mo_coeff_b[:, : APOSTBase.occ_b[APOSTBase.occ_b > 0].shape[0]]
                occ_coeff_a = APOSTBase.mo_coeff_a[:, : APOSTBase.occ_a[APOSTBase.occ_a > 0].shape[0]]
                occ_a = APOSTBase.occ_a[APOSTBase.occ_a > 0]
                occ_b = APOSTBase.occ_b[APOSTBase.occ_b > 0]
                occ_coeff = np.zeros((occ_coeff_b.shape[0], occ_coeff_a.shape[1] + occ_coeff_b.shape[1]))
                occ_coeff[:, : occ_coeff_a.shape[1]] = occ_coeff_a
                occ_coeff[:, occ_coeff_a.shape[1]:] = occ_coeff_b
                occs = np.zeros((occ_a.shape[0] + occ_b.shape[0]))
                occs[: occ_a.shape[0]] = occ_a
                occs[occ_a.shape[0]:] = occ_b

                norb = occs.shape[0]
                norb_a = occ_a.shape[0]
                norb_b = occ_b.shape[0]
                dij = np.diag(occs)
                dm1_a = np.diag(occ_a)
                dm1_b = np.zeros_like(occ_a)
                dm1_b[:occ_b.shape[0]] = occ_b
                dm1_b = np.diag(dm1_b)
                # For coulomb
                eri4 = APOSTBase._mol.ao2mo(occ_coeff)
                eri_4fold = ao2mo.restore(1, eri4, occs.shape[0])
                print('Shape eri4 fold all', eri_4fold.shape)
                # For exchange a
                eri4_a = APOSTBase._mol.ao2mo(occ_coeff_a)
                eri_4fold_a = ao2mo.restore(1, eri4_a, occ_a.shape[0])
                print('Shape eri4 fold alpha', eri_4fold_a.shape)
                # For exchange b
                eri4_b = APOSTBase._mol.ao2mo(occ_coeff_b)
                eri_4fold_b = ao2mo.restore(1, eri4_b, occ_b.shape[0])
                print('Shape eri4 fold beta', eri_4fold_b.shape)

            else:
                norb = APOSTBase.occ_a[APOSTBase.occ_a > 0].shape[0]
                dij = np.diag(APOSTBase.occ_a[APOSTBase.occ_a > 0])
                dm1_a = np.diag(APOSTBase.occ_a[APOSTBase.occ_a > 0])
                occ_b = np.zeros_like(APOSTBase.occ_a[APOSTBase.occ_a > 0])
                occ_b[:APOSTBase.occ_b[APOSTBase.occ_b > 0].shape[0]] = APOSTBase.occ_b[APOSTBase.occ_b > 0]
                dm1_b = np.diag(occ_b)
                eri4 = APOSTBase._mol.ao2mo(APOSTBase.mo_coeff_a[:, : APOSTBase.occ_a[APOSTBase.occ_a > 0].shape[0]])
                print(APOSTBase.occ_a[APOSTBase.occ_a > 0].shape[0])
                eri_4fold = ao2mo.restore(1, eri4, APOSTBase.occ_a[APOSTBase.occ_a > 0].shape[0])
                print('Shape eri4 fold', eri_4fold.shape)

        elif APOSTBase.cas:
            print('CASSCF')
            # Todo: This is for restricted. we need to check for unrestricted
            dij = APOSTBase._dij
            dm1_a = APOSTBase._dm1_a
            dm1_b = APOSTBase._dm1_b
            norb = APOSTBase._cas_data['ncas'] + APOSTBase._cas_data['ncore']

            eri4 = APOSTBase._mol.ao2mo(APOSTBase._cas_mo_coeff[:, :norb]) # cas_mo_coeff from SCF -> 1 set for RHF/ROHF
            eri_4fold = ao2mo.restore(1, eri4, norb)
            print('Shape eri4 fold', eri_4fold.shape)
        elif APOSTBase._fci:
            print('FCI')
            dij = APOSTBase._dij
            dm1_a = APOSTBase._dm1_a
            dm1_b = APOSTBase._dm1_b
            # Todo: This is for restricted. we need to check for unrestricted
            norb = APOSTBase._fci_data['norb']
            eri4 = APOSTBase._mol.ao2mo(APOSTBase._fci_mo_coeff)
            eri_4fold = ao2mo.restore(1, eri4, norb)
            print('Shape eri4 fold', eri_4fold.shape)


    # Initialize results dict
    e_terms = {}
    if 'kinetic' in term or 'all' in term :
        e_terms['kin'] = np.trace(np.dot(dm, APOSTBase._mol.intor_symmetric('int1e_kin')))
    if 'en' in term or 'all' in term :
        e_terms['en'] = np.trace(np.dot(dm, APOSTBase._mol.intor_symmetric('int1e_nuc')))
    if 'Coul' in term or 'all' in term :
        e_terms['coul'] = 0.5 * np.einsum("ab,abcd,cd->", dij, eri_4fold, dij)
    if 'Exch' in term or 'all' in term :
        if APOSTBase.unrestr and not APOSTBase.cas and not APOSTBase._fci:
            e_terms['x_a'] =  -0.5 * np.einsum("abba->", eri_4fold_a)
            e_terms['x_b'] =  - 0.5 *np.einsum("abba->", eri_4fold_b)
        else:
            if APOSTBase.cas or APOSTBase._fci:
                # Todo: Fix the mismatch for CASSCF/ROHF and FCI. We save the ROHF as an unrestricted
                # calculation but the internal scf.ROHF only saves 1 set of mo_coeff and no occ_a/occ_b. Because of
                # this we compute a/b separately and only apply 1/2 of the value instead of 1.4
                # that would be closed shell because we use a DM1_a/DM1_b with occs<=1. From natural orbitals?
                e_terms['x_a'] = -0.5 * np.einsum("ac,abcd,bd->", dm1_a, eri_4fold, dm1_a)
                e_terms['x_b'] = -0.5 * np.einsum("ac,abcd,bd->", dm1_b, eri_4fold, dm1_b)
            else:
                e_terms['x_a'] = - np.einsum("abba->", eri_4fold)
                # e_terms['x_b'] = -0.25 * np.einsum("ac,abcd,bd->", dm1_b[:norb, :norb], eri_4fold, dm1_b[:norb, :norb])


    if 'ee' in term or 'all' in term :
        if APOSTBase._cas or APOSTBase._fci:
            e_terms['ee'] = 0.5 * np.einsum("abcd,abcd->", APOSTBase.dm2, eri_4fold)
            e_terms['total_terms'] = e_terms['ee'] + APOSTBase._mol.energy_nuc() + e_terms['kin'] + e_terms['en']
            e_terms['total_pyscf'] = APOSTBase.energy
        else:
            if APOSTBase.unrestr:
                e_terms['ee'] = e_terms['x_a'] + e_terms['x_b'] + e_terms['coul']
                e_terms['total_terms'] = e_terms['x_a'] + e_terms['x_b'] + e_terms['coul'] + APOSTBase._mol.energy_nuc() + e_terms['kin'] +e_terms['en']
                e_terms['total_pyscf'] = APOSTBase.energy
            else:
                e_terms['ee'] = e_terms['x_a'] +  e_terms['coul']
                e_terms['total_terms'] = e_terms['x_a']  + e_terms['coul'] + APOSTBase._mol.energy_nuc() + e_terms['kin'] + e_terms['en']
                e_terms['total_pyscf'] = APOSTBase.energy


    if ('u12_ps' in term or 'all' in term) and (APOSTBase._cas or APOSTBase._fci):
        u, u12 = unpaired_elec(APOSTBase._dij, APOSTBase.at_nums, APOSTBase.smo)
        e_terms['u12'] = -0.25 * np.einsum("ac,abcd,bd->", u12, eri_4fold, u12)

        # DEV: For new exchange.

        print('DEV 1 with u12', e_terms['x_a'] + e_terms['x_b'] +  e_terms['coul'] + e_terms['u12'])
        print('DEV 2', e_terms['x_a'] + e_terms['x_b']+  e_terms['coul'])


        e_terms['old_x']=  e_terms['x_a']+ e_terms['x_b']
        e_terms['new_x']= e_terms['x_a'] + e_terms['x_b'] + e_terms['u12']
        e_terms['old_c'] = e_terms['ee'] -  e_terms['coul']- e_terms['old_x']
        e_terms['new_c'] = e_terms['ee'] -  e_terms['coul'] - e_terms['new_x']
        e_terms['xc'] = e_terms['ee'] - e_terms['coul']

    if out:
        out_ee_terms_analyt(e_terms, APOSTBase.energy, APOSTBase._mol.energy_nuc())
    e_terms['nn'] = APOSTBase._mol.energy_nuc()


    return e_terms






