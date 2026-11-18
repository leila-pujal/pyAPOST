#Local Spin analysis module

import numpy  as np

from pyscf.dft.numint import eval_ao, eval_rho

from pyAPOST.bond_order import unpaired_elec
from pyAPOST.utils import out_bo

#Implementation of equation 12 from paper dx.doi.org/10.1021/ct300050c. For this we need:
# - u(r): call to  unpaired_elec() in bond_order.py
# - cumulant($\Gamma$): Currently this is calculated(or at least all the machinery is) in iqa_phf.py. Maybe we will need
# to take the part where we generate the different terms from the 2-RDM to be a standalone function.
# - $\rho_{s}$(spin density): Easy to obtain from the alpha and beta 1-RDM.


def generate_two_electron_density(norb, dm1, dm2, cumul_type='xc', u12=None):
    """Function to generate different types of two electron density from the dm2
     """

    # Check if we ask for generalized exchange(gx) or generalized correlation(c_gx) we need to provide the u12
    if cumul_type == 'gx' and u12 is None:
        raise ValueError('u12 must be specified if cumul is set to gx')

    if cumul_type == 'c_gx' and u12 is None:
        raise ValueError('u12 must be specified if cumul is set to c_gx')

    # Now we compute the cumulant. for this we build spdm1 and psdm1
    # norb = total orbitals CAS
    # For now APOST names
    sfdm1 = np.zeros((norb, norb))  # spinfree density
    psdm1 = np.zeros((norb, norb))  # spin density
    for i in range(norb):
        for j in range(norb):
            sfdm1[i, j] = dm1[i * 2, j * 2] + dm1[i * 2 + 1, j * 2 + 1]
            psdm1[i, j] = dm1[i * 2, j * 2] - dm1[i * 2 + 1, j * 2 + 1]

    # COMPUTE ALL CUMULANTS
    cumul = np.zeros((norb, norb, norb, norb))
    # x_cumul = np.zeros((norb, norb, norb, norb))
    # gx_cumul = np.zeros((norb, norb, norb, norb))
    # c_cumul = np.zeros((norb, norb, norb, norb))
    # c_gx_cumul = np.zeros((norb, norb, norb, norb))
    print('STARTING LOOP')
    print(dm2.shape)
    for i in range(norb):
        for j in range(norb):
            for k in range(norb):
                for l in range(norb):
                    # Todo: check if below code can be achieved without the 4 index loop
                    # xc cumulant--> without coulomb
                    coult = sfdm1[i, j] * sfdm1[k, l]
                    x1 = (sfdm1[i, k] * sfdm1[j, l]) / 2
                    x1_s = (sfdm1[i, l] * sfdm1[k, j]) / 2
                    # x1_a = (sfdm1[l, i] * sfdm1[j, k]) / 2
                    x2 = (psdm1[i, k] * psdm1[j, l]) / 2
                    x2_s = (psdm1[i, l] * psdm1[k, j]) / 2
                    if cumul_type == 'xc':
                        # xc_cumul[i, j, k, l] = dm2[i, j, k, l] - coult
                        cumul[i, j, k, l] = dm2[i, j, k, l] - coult
                    elif cumul_type == 'x':
                        # x_cumul[i, j, k, l] = - x1 - x2
                        cumul[i, j, k, l] = - x1_s - x2_s
                    elif cumul_type == 'c':
                        # c_cumul[i, j, k, l] = dm2[i, j, k, l] - coult + x1 + x2
                        cumul[i, j, k, l] = dm2[i, j, k, l] - coult + x1_s+ x2_s
                    elif cumul_type == 'c_ps':
                        # print('DEV CHECK IN C_PS')
                        cumul[i, j, k, l] = dm2[i, j, k, l] - coult + x1_s
                        # cumul[i, j, k, l] = dm2[i, j, k, l] - coult
                    elif cumul_type == 'c_gx':
                        if u12 is not None:
                            x_unp = (u12[i, k] * u12[j, l]) / 2
                            # gx_cumul[i, j, k, l] = -x1 - x_unp
                            # c_gx_cumul[i, j, k, l] = dm2[i, j, k, l] - coult + x1 + x_unp
                            cumul[i, j, k, l] = dm2[i, j, k, l] - coult + x1_s + x_unp
                        else:
                            raise ValueError('Need u12 to compute c_gx')
                    elif cumul_type == 'gx':
                        if u12 is not None:
                            x_unp = (u12[i, k] * u12[j, l]) / 2
                            # gx_cumul[i, j, k, l] = -x1 - x_unp
                            cumul[i, j, k, l] = -x1_s - x_unp
                            # c_gx_cumul[i, j, k, l] = dm2[i, j, k, l] - coult + x1 + x_unp
                        else:
                            raise ValueError('Need u12 to compute gx')
                    else:
                        raise TypeError(f'Could not identify the type of two electrons density. Got{cumul_type}')


    return cumul

def local_spin(APOSTBase, a=0.75):

    # TODO: The code below is used here and at least in two other places. Think about moving it to have
    # its standalone function to avoid not introducing bugs every time.
    # Molecular orbital coefficients form the CAS. If CASSCF these are optimized and if CASCI they should be the same as the underlying SCF calc
    if APOSTBase._cas:
        mo_coeff = APOSTBase._cas_mo_coeff
        norb = APOSTBase._cas_data['ncas'] + APOSTBase._cas_data['ncore']
    elif APOSTBase._fci:
        mo_coeff = APOSTBase._fci_mo_coeff
        norb = APOSTBase._fci_data['norb']
    elif APOSTBase._ccsd:
        raise NotImplementedError
    else:
        raise TypeError('Not a valid post-hartree fock calculation.')

    # Evaluate molecular orbitals on the grid
    occ_no = APOSTBase._occ_no
    if APOSTBase._cas:
        norb = APOSTBase._cas_data['ncas'] + APOSTBase._cas_data['ncore']
    elif APOSTBase._fci:
        norb = APOSTBase._fci_data['norb']
    # Added here just in case we need it but should be removed after dev
    no_coeff = APOSTBase.no_coeff
    no_coeff_ao_basis = np.einsum('bk,ki ->bi', mo_coeff[:, : norb], no_coeff)


    eval_ao_grid1 = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=0)
    eval_mo_grid1 = np.einsum('pa,ai->ip ', eval_ao_grid1, mo_coeff)
    # eval_no_grid1 = np.einsum('pa,ai->ip ', eval_ao_grid1, no_coeff_ao_basis)

    # Get u matrix in terms of MOs
    u, u12 = unpaired_elec(APOSTBase._dij, APOSTBase.at_nums)

    # Contribution from cumulant and spin density:
    # prefactor
    pf = (1-2*a)
    print(f'PREFACTOR VALUE WITH a={a}', pf)
    # Cumulant(two-electron density) containing correlation and spin density
    cumul_c_ps = generate_two_electron_density(norb, APOSTBase.dm1, APOSTBase.dm2, cumul_type='c_ps')

    s234 = np.zeros((len(APOSTBase.at_nums), len(APOSTBase.at_nums)))
    # Get one and two center contributions
    # Arrays to save aa/ab terms for local spin analysis terms
    u_ats = np.zeros((len(APOSTBase.at_nums)))  # u contribution is only atomic
    t1 = np.zeros((len(APOSTBase.at_nums), len(APOSTBase.at_nums)))
    t2 = np.zeros((len(APOSTBase.at_nums), len(APOSTBase.at_nums)))
    for ia in range(len(APOSTBase.at_nums)):
        u_at = a *np.trace(np.dot(u, APOSTBase.smo[ia]))
        u_ats[ia] = u_at
        print('DEV CHECK: u_at', u_at)
        for ib in range(len(APOSTBase.at_nums)):
            dm2_t1 = np.einsum('ijkl, ij,kl ->', pf * cumul_c_ps, APOSTBase.smo[ia], APOSTBase.smo[ib])
            dm2_t2 = np.einsum('ijkl, il,kj ->', 0.5 * cumul_c_ps, APOSTBase.smo[ia], APOSTBase.smo[ib])
            t1[ia, ib] = dm2_t1
            t2[ia, ib] = dm2_t2

            if ia == ib:
                s234[ia, ib] = u_at - dm2_t1 -dm2_t2
            else:
                s234[ia, ib] = -dm2_t1 -dm2_t2

    print('DEBUG u_at', u_ats)
    lsa = dict(Local_spin=s234, a=a, u_at=u_ats, t1=t1, t2=t2)
    out_bo(APOSTBase.at_nums, lsa, index_type='local_spin')

    return lsa



