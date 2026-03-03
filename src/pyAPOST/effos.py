import numpy as np

import logging
import warnings as warnings

from pyAPOST.base import Hpart, Rpart

# Draft: The abstract can potentially vary between more OO or more function centered
# option1: Define an object to store the results of an effos/eos calculation
# option2: Define functions that return values but if you need them stored you should define variables
# to contain them within the script that calls them.

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

def smo_hilb(APOSTBase, gen=False):
    """"Function to return effos and occupations"""

    # Generate S_AO_frags
    if APOSTBase.part.part_name == 'mulliken':
        S_AO_frags = [np.linalg.multi_dot((eta_slices, APOSTBase.ovlp, eta_slices)) for eta_slices
                      in APOSTBase.part.eta_frags]
    else:
        S_AO_frags = [np.linalg.multi_dot((APOSTBase.part.T.T, eta_slice, APOSTBase.part.T)) for
                      eta_slice in APOSTBase.part.eta_frags]

    if gen: #Generalized effos obtained with Paired and Unpaired densities
        # Todo: Test checking all this sorting works
        # Because in APOSTBase we always save alpha/beta molecular coefficients and generalized is based on paired/unpaired density we sum them here
        # for all methods except restricted SD
        # For generalized, unrestricted calculations need to compute the coefficients as well
        thresh = 1.0e-8
        if APOSTBase._sd and APOSTBase._unrestr:
            dm = APOSTBase.dm_a + APOSTBase.dm_b
            occ, occ_coeff = APOSTBase.generate_mo_coeff(dm, overlap=APOSTBase.ovlp)

            occ_coeff = occ_coeff[:, occ > thresh]
            occ = occ[occ > thresh]

        # Post-Hartree Fock
        elif APOSTBase._casscf or APOSTBase._ccsd:
            # In APOSTbase we always generate dm_a and dm_b. For generalized effos original code APOSTpy works with paired Dm by calling
            # mcscf.addons.make_rdm1() instead of mcscf.addons.make_rdm1s(). Internally PySCF generates Dm returned by mcscf.addons.make_rdm1()
            # by adding dma and dmb in pyscf/fci/direct_spin1.py def make_rdm1. Here we are adding coefficents
            occ = APOSTBase.occ_a + APOSTBase.occ_b
            occ_coeff = APOSTBase.mo_coeff_a + APOSTBase.occ_b

            occ_coeff = occ_coeff[:, occ > thresh]
            occ = occ[occ > thresh]

        # Last, restricted single determinant. Assign the common variable to alpha coeff. The code below would work the same using beta attributes
        else:
            occ_coeff = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
            occ = APOSTBase.occ_a[APOSTBase.occ_a > thresh]

        # Smo_u could be copied from Smo_p
        Smo_p = np.array([np.dot(np.dot(occ_coeff.T, S_AO_frag), occ_coeff) for S_AO_frag in S_AO_frags])
        Smo_u = np.array([np.dot(np.dot(occ_coeff.T, S_AO_frag), occ_coeff) for S_AO_frag in S_AO_frags])

        # Normalize Smo_p and Smo_u for post-hartree fock methods and unrestricted SD
        if APOSTBase._casscf or APOSTBase._ccsd or APOSTBase._unrestr:
            # todo: comment on the outer product we are doing below
            norm_occ_p = np.sqrt(abs((occ - 1) * occ))[:, None] * np.sqrt(abs((occ - 1) * occ))[None, :]
            norm_occ_u = np.sqrt(abs((2 - occ) * occ))[:, None] * np.sqrt(abs((2 - occ) * occ))[None, :]

            Smo_p = Smo_p * norm_occ_p[None, :, :]
            Smo_u = Smo_u * norm_occ_u[None, :, :]

        return Smo_p, Smo_u

    else:
        # Standard effos obtained with Alpha and Beta densities
        # TODO: Make "thresh" availble to specify
        thresh = 1.0e-8
        occ_coeff_a = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
        occ_coeff_b = APOSTBase.mo_coeff_b[:, APOSTBase.occ_b > thresh]

        # Q: Smo_a changes for every calculation?
        Smo_a = np.array([np.dot(np.dot(occ_coeff_a.T, S_AO_frag), occ_coeff_a)  for S_AO_frag in S_AO_frags])
        Smo_b = np.array([np.linalg.multi_dot((occ_coeff_b.T, S_AO_frag, occ_coeff_b))  for S_AO_frag in S_AO_frags])

        # Normalize Smo_a and Smo_b for post-hartree fock methods
        if APOSTBase._casscf or APOSTBase._ccsd:
            norm_occ_a = np.sqrt(APOSTBase.occ_a[APOSTBase.occ_a > thresh])[:, None] * np.sqrt(APOSTBase.occ_a[APOSTBase.occ_a > thresh])[None, :]
            norm_occ_b = np.sqrt(APOSTBase.occ_b[APOSTBase.occ_b > thresh])[:, None] * np.sqrt(APOSTBase.occ_b[APOSTBase.occ_b > thresh])[None, :]

            Smo_a = Smo_a * norm_occ_a[None, :, :]
            Smo_b = Smo_b * norm_occ_b[None, :, :]

        return Smo_a, Smo_b

def qij_real(APOSTBase, gen=False):
    """Function to construct Q_ij matrix to obtain effos defined in a real space partition
    …math::
     Q_{ij} = \int w^{2}_{A}(r)\varphi^{*}_{i}(r)\varphi_{j}(r)dr

    For natural orbitals the occupations are included as:
    Q_{ij} = \sqrt{n_i} \sqrt{n_j} \int w^{2}_{A}(r)\varphi^{*}_{i}(r)\varphi_{j}(r)dr

    Where:
        \varphi_{i}(r): Molecular or Natural orbital
        w^{2}_{A}(r): AIM weight for atom A
        n_{i}: occupation of the i-th natural orbital

    Parameters:
    ----------
    APOSTBase: Object of class APOSTBase
    gen: Bool
        Flag to use generalized or standard effos formulation

    Return:
    ------
    qij_a, qij_b/ qij_u, qij_p: np.array(Frags, MOs, MOs)

    """

    if gen:
        # Assign occ and occ_coeff
        # Todo: change this when both Natural and SCF molecular coefficients are updated
        thresh = 1.0e-8
        if APOSTBase._sd and APOSTBase._unrestr:
            dm = APOSTBase.dm_a + APOSTBase.dm_b
            occ, occ_coeff = APOSTBase.generate_mo_coeff(dm, overlap=APOSTBase.ovlp)

            occ_coeff = occ_coeff[:, occ > thresh]
            occ = occ[occ > thresh]

        # Post-Hartree Fock
        elif APOSTBase._casscf or APOSTBase._ccsd:
            # In APOSTbase we always generate dm_a and dm_b. For generalized effos original code APOSTpy works with paired Dm by calling
            # mcscf.addons.make_rdm1() instead of mcscf.addons.make_rdm1s(). Internally PySCF generates Dm returned by mcscf.addons.make_rdm1()
            # by adding dma and dmb in pyscf/fci/direct_spin1.py def make_rdm1. Here we are adding coefficents
            occ = APOSTBase.occ_a + APOSTBase.occ_b
            occ_coeff = APOSTBase.mo_coeff_a + APOSTBase.occ_b

            occ_coeff = occ_coeff[:, occ > thresh]
            occ = occ[occ > thresh]

        # Last, restricted single determinant. Assign the common variable to alpha coeff. The code below would work the same using beta attributes
        else:
            occ_coeff = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
            occ = APOSTBase.occ_a[APOSTBase.occ_a > thresh]

        # Compute molecular orbital overlap matrix. For now using molecular grid
        mo_grid = np.einsum('pa,ai->ip ', APOSTBase.part.ao_eval_grid, occ_coeff)

        # Check 1: total integrated electrons
        dens_grid = np.zeros((mo_grid.shape[1]))
        for i in range(mo_grid.shape[0]):
            dens_grid += mo_grid[i, :] * mo_grid[i, :] * occ[i]

        total_elect_int = np.sum(dens_grid * APOSTBase.part.grid_weights)

        logger.info(f'  Check integrated elect: {total_elect_int: .5f}\n')

        qij_p = np.zeros((len(APOSTBase.part.frags), occ.shape[0], occ.shape[0]))
        qij_u = np.zeros((len(APOSTBase.part.frags), occ.shape[0], occ.shape[0]))
        pop_all = []
        for ifrag, frag in enumerate(APOSTBase.part.frags):
            tmp_w = np.zeros((APOSTBase.part.grid_weights.shape[0]))
            for ia in frag:
                # Adding weights for all atoms in the fragment
                ia = ia-1 # Reverting to python index
                tmp_w[APOSTBase.part.grid_to_atom_idx == ia] += APOSTBase.part.aim_w[APOSTBase.part.grid_to_atom_idx == ia]
            pop = np.sum(dens_grid * APOSTBase.part._grid.quadrature_weights * tmp_w)
            pop_all.append(pop)
            logger.debug(f' TOTAL POPULATION Frag {ifrag + 1} : {pop :.4f}\n')

            for i in range(occ.shape[0]):
                for j in range(APOSTBase.occ_a[APOSTBase.occ_a > thresh].shape[0]):
                    ij_p = mo_grid[i, :] * mo_grid[j, :] * (tmp_w**2) * APOSTBase.part._grid.quadrature_weights
                    qij_p[ifrag, i, j] = np.sum(ij_p)
            for i in range(occ.shape[0]):
                for j in range(occ.shape[0]):
                    ij_u =  mo_grid[i, :] * mo_grid[j, :] * (tmp_w**2) * APOSTBase.part._grid.quadrature_weights
                    qij_u[ifrag, i, j] = np.sum(ij_u)

        # Each qij has the same dimensions as smo
        logger.debug(f'  SUM POP A = {sum(pop_all): .6f}')

        # Normalize Qij_p and Qij_u for post-hartree fock methods and unrestricted SD
        if APOSTBase._casscf or APOSTBase._ccsd or APOSTBase._unrestr:
            # outer product
            norm_occ_p = np.sqrt(abs((occ - 1) * occ))[:, None] * np.sqrt(abs((occ - 1) * occ))[None, :]
            norm_occ_u = np.sqrt(abs((2 - occ) * occ))[:, None] * np.sqrt(abs((2 - occ) * occ))[None, :]

            qij_p = qij_p * norm_occ_p[None, :, :]
            qij_u = qij_u * norm_occ_u[None, :, :]

        return qij_p, qij_u

    else:
        # Compute molecular orbital overlap matrix. For now using molecular grid
        mo_grid_a = np.einsum('pa,ai->ip ', APOSTBase.part.ao_eval_grid, APOSTBase.mo_coeff_a)
        mo_grid_b = np.einsum('pa,ai->ip ', APOSTBase.part.ao_eval_grid, APOSTBase.mo_coeff_b)
        # Note: Storing molecular overlap matrices is too memory expensive to allocate
        # S_mo_a = np.zeros((mo_grid_a.shape[0], mo_grid_a.shape[0], mo_grid_a.shape[1]))
        # total_elect_int_a = np.sum(np.trace(S_mo_a * APOSTBase.occ_a[:, None, None]) * APOSTBase.part.grid_weights)

        # Check 1: total integrated electrons
        dens_grid_a = np.zeros((mo_grid_a.shape[1]))
        dens_grid_b = np.zeros((mo_grid_b.shape[1]))
        for i in range(mo_grid_a.shape[0]):
            dens_grid_a += mo_grid_a[i, :] * mo_grid_a[i, :] * APOSTBase.occ_a[i]
            dens_grid_b += mo_grid_b[i, :] * mo_grid_b[i, :] * APOSTBase.occ_b[i]

        total_elect_int_a = np.sum(dens_grid_a * APOSTBase.part.grid_weights)
        total_elect_int_b = np.sum(dens_grid_b * APOSTBase.part.grid_weights)

        if APOSTBase.sd and not APOSTBase.unrestr:
            logger.info(f'  Check integrated elect: {total_elect_int_a: .5f}\n')
        else:
            logger.info(f'  Check integrated elect alpha: {total_elect_int_a: .5f}')
            logger.info(f'  Check integrated elect beta: {total_elect_int_b: .5f}\n')

        thresh = 1.0e-8
        qij_a = np.zeros((len(APOSTBase.part.frags), APOSTBase.occ_a[APOSTBase.occ_a > thresh].shape[0],APOSTBase.occ_a[APOSTBase.occ_a > thresh].shape[0]))
        qij_b = np.zeros((len(APOSTBase.part.frags), APOSTBase.occ_b[APOSTBase.occ_b > thresh].shape[0], APOSTBase.occ_b[APOSTBase.occ_b > thresh].shape[0]))
        pop_a_all = []
        pop_b_all = []
        # todo: loop over lower triangle only
        # todo: Code below same for generalized. Try to only have one
        for ifrag, frag in enumerate(APOSTBase.part.frags):
            tmp_w = np.zeros((APOSTBase.part.grid_weights.shape[0]))
            for ia in frag:
                # Adding weights for all atoms in the fragment
                ia = ia-1 # Reverting to python index
                tmp_w[APOSTBase.part.grid_to_atom_idx == ia] += APOSTBase.part.aim_w[APOSTBase.part.grid_to_atom_idx == ia]
            pop_a = np.sum(dens_grid_a * APOSTBase.part._grid.quadrature_weights * tmp_w)
            pop_b = np.sum(dens_grid_b * APOSTBase.part._grid.quadrature_weights * tmp_w)
            pop_a_all.append(pop_a)
            pop_b_all.append(pop_b)

            if APOSTBase._sd and not APOSTBase.unrestr:
                logger.debug(f'  TOTAL POPULATION Frag {ifrag + 1} : {pop_a :.4f}\n')
            else:
                logger.debug(f'  ALPHA POPULATION Frag {ifrag+1} : {pop_a :.4f}')
                logger.debug(f'  BETA POPULATION Frag {ifrag + 1} : {pop_b :.4f}')
                logger.debug(f'  TOTAL POPULATION Frag {ifrag + 1} : {pop_a + pop_b :.4f}\n')
            for i in range(APOSTBase.occ_a[APOSTBase.occ_a > thresh].shape[0]):
                for j in range(APOSTBase.occ_a[APOSTBase.occ_a > thresh].shape[0]):
                    ij_a = mo_grid_a[i, :] * mo_grid_a[j, :] * (tmp_w**2) * APOSTBase.part._grid.quadrature_weights
                    qij_a[ifrag, i, j] = np.sum(ij_a)
            for i in range(APOSTBase.occ_b[APOSTBase.occ_b > thresh].shape[0]):
                for j in range(APOSTBase.occ_b[APOSTBase.occ_b > thresh].shape[0]):
                    ij_b = mo_grid_b[i, :] * mo_grid_b[j, :] * (tmp_w**2) * APOSTBase.part._grid.quadrature_weights
                    qij_b[ifrag, i, j] = np.sum(ij_b)

            if APOSTBase._casscf or APOSTBase._ccsd:
                norm_occ_a = np.sqrt(APOSTBase.occ_a[APOSTBase.occ_a > thresh])[:, None] * np.sqrt(APOSTBase.occ_a[APOSTBase.occ_a > thresh])[None, :]
                norm_occ_b = np.sqrt(APOSTBase.occ_b[APOSTBase.occ_b > thresh])[:, None] * np.sqrt(APOSTBase.occ_b[APOSTBase.occ_b > thresh])[None, :]

                qij_a = qij_a * norm_occ_a[None, :, :]
                qij_b = qij_b * norm_occ_b[None, :, :]


        # todo: return always alpha/beta (paired/unpaired) qij.
        # Each qij has the same dimensions as smo
        logger.debug(f'  SUM POP A = {sum(pop_a_all): .6f}')
        logger.debug(f'  SUM POP B = {sum(pop_b_all) : .6f}')

        return qij_a, qij_b


def effos_occ(smo1, nelect1, smo2, nelect2, frags=None, gen=False):
    """WIP: function to generate effos occupations.
    option1: Independent of the APOSTBase object and add all the necessary information as arguments. Passing the information should be dealt at the script level
    or define another funciton that performs the EOS calculation on a single run
    option2: Make it depend on the APOSTBAse object and the SMO. This means only APOSTBAse is needed and inside the function the information needed is retreived.

    """

    def _compute_rmsd(eig_1, eig_2, n1, n2, gen=False):
        """Auxiliary function to compute the Root Mean Squared Error of effos occupations.

        Parameters:
        -----------
            eig_1: List
                Ordered list of effos eigenvalues for alpha or paired
            eig_2: List
                Ordered list of effos eigenvalues for beta or unpaired
            n1: Int
                alpha/paired electrons
            n2: Int
                beta/unpaired electrons
            gen: Bool
                Flag to use Generalized or standard effos formalism

        Returns:
        --------
            RMSD: Float
                Root Mean Squared Error between eigenvalues and theoretical occupations
        """

        effos_occ = np.zeros((len(eig_1) + len(eig_2)))
        if gen:
            effos_occ[:n1] = 2
            effos_occ[len(eig_1): len(eig_1) + n2] = 1
            complete_effos = np.concatenate((eig_1, eig_2))
        else:
            effos_occ[:n1] = 1
            effos_occ[len(eig_1): len(eig_1) + n2] = 1
            complete_effos = np.concatenate((eig_1, eig_2))

        rmsd_1 = (complete_effos - effos_occ) ** 2
        rmsd = np.sqrt(np.mean(rmsd_1))
        return rmsd

    # for alpha/beta effos occupations we only need to get the usual ocupations
    # for generalized we need to iterate and check the RMSD for different number of occupations

    # Check nelect1,nelect2 type
    if not (isinstance(nelect1, int) and isinstance(nelect2, int)):
        raise TypeError("Either nelect1 or nelect2 must be an integer")

    if gen:
        logger.info("  STARTING EFFOS OCCUPATIONS FOR PAIRED/UNPAIRED SMO/Qij")
        logger.info(f"  DOING FOR {nelect1} PAIRED AND {nelect2} UNPAIRED \n")
    else:
        logger.info("  STARTING EFFOS OCCUPATIONS FOR ALPHA/BETA SMO/Qij")
        logger.info(f"  DOING FOR {nelect1} ALPHA AND {nelect2} BETA\n")

    # Check smo_1/smo_2: type numpy array, np.array: N_frags, MOs, MOs
    if not (isinstance(smo1, np.ndarray) and smo1.ndim == 3 and smo1.dtype == float):
        raise TypeError("S_MO_1 must be a 2D-array float `dtype`")

    if not (isinstance(smo2, np.ndarray) and smo2.ndim == 3 and smo2.dtype == float):
        raise TypeError("S_MO_2 must be a 2D-array float `dtype`")

    # Define dic for storage
    effos_data = {'dens1': {}, 'dens2': {}}

    # Compute eigenvec/eigenval
    s1_egval, s1_egvec = np.linalg.eigh(smo1)
    s2_egval, s2_egvec = np.linalg.eigh(smo2)
    s1_egval, s1_egvec = s1_egval[:, ::-1], s1_egvec[:, :, ::-1]
    s2_egval, s2_egvec = s2_egval[:, ::-1], s2_egvec[:, :, ::-1]

    # Avoid division by 0
    s1_egvec[s1_egvec == 0] = 1e-10
    s2_egvec[s2_egvec == 0] = 1e-10

    #Normalize eigenvectors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1_egvec = s1_egvec / np.sqrt(s1_egval[:, None, :])
        s2_egvec = s2_egvec / np.sqrt(s2_egval[:, None, :])
    # print('shape eigval', s1_egval.shape, s2_egval.shape)

    effos_data['dens1']['egvec'] = s1_egvec
    effos_data['dens2']['egvec'] = s2_egvec

    effos_data['dens1']['egval'] = s1_egval
    effos_data['dens2']['egval'] = s2_egval

    # Generates sorted list of tuples (eigenvalue, frag)
    s1 = sorted([(s, i) for i, row in enumerate(s1_egval) for s in row], reverse=True)
    s2 = sorted([(s, i) for i, row in enumerate(s2_egval) for s in row], reverse=True)

    effos_data['dens1']['s1'] = s1
    effos_data['dens2']['s2'] = s2

    # Unzip the tuple items into two list to have ordered fragment indices. Equivalent to occs_1 = [eigval[1] for eigval in s1]
    eig_1, occs_1 = zip(*s1)
    eig_2, occs_2 = zip(*s2)

    # RMSD assignation:
    logger.info('  RMSD ASSIGNATION CHECK \n')
    n1 = nelect1
    n2 = nelect2
    if gen:
        logger.info(f"  INITIAL ASSIGNATION:  {n1} PAIRED AND {n2} UNPAIRED")
    else:
        logger.info(f"  INITIAL ASSIGNATION:  {n1} ALPHA AND {n2} BETA")
    rmsd = _compute_rmsd(eig_1, eig_2, n1, n2, gen=gen)
    logger.info(f'  RMSD: {rmsd: .4f} \n')
    # Now we check RMSD splitting 2 Paired electrons into unpaired ones
    it = 0
    if n1 == nelect1 and n2 == nelect2 and gen:
        while True:
            it += 1
            n1 -= 1
            n2 += 2
            logger.info(f"{it} ITERATION: {n1} PAIRED AND {n2} UNPAIRED ")
            rmsd_alt = _compute_rmsd(eig_1, eig_2, n1, n2, gen=gen)
            logger.info(f'  RMSD: {rmsd_alt: .4f} \n')
            if rmsd_alt > rmsd:
                break
            else:
                rmsd = rmsd_alt
                nelect1 = n1
                nelect2 = n2

    logger.info(f"  FINAL RMSD ASSIGNATION : {nelect1} PAIRED AND {nelect2} UNPAIRED and RMSD = {rmsd: .4f} \n")

    frag_occs_1 = [occs_1[:nelect1].count(index) for index in range(len(frags))]
    frag_occs_2 = [occs_2[:nelect2].count(index) for index in range(len(frags))]

    frag_effo_occ_1 = s1[:nelect1]
    frag_effo_occ_2 = s2[:nelect2]

    frag_effo_unocc_1 = s1[nelect1:]
    frag_effo_unocc_2 = s2[nelect2:]

    lo_1 = nelect1 - 1  # Python index
    fu_1 = nelect1  # Python index

    lo_2 = nelect2 - 1  # Python index
    fu_2 = nelect2  # Python index

    # Check if pseudodegenerate effos
    # Note rounding the values to use count
    eig_1_r = [round(egv, 5) for egv in eig_1]
    eig_2_r = [round(egv, 5) for egv in eig_2]
    # Checking if the last assignation corresponds to a degenerate atomic orbital
    # Todo: Check if we could condense the two if statements
    if eig_1_r.index(eig_1_r[nelect1]) > 1:
        logger.info('  EOS: Warning, pseudo-degeneracies detected for Alpha/Paired effos')
        deg_idx_1 = np.array([i for i, x in enumerate(eig_1_r) if x == eig_1_r[nelect1]])
        frac_1 = np.sum(deg_idx_1 <= nelect1-1) / deg_idx_1.size
        logger.info(f'  Distributing {np.sum(deg_idx_1 <= nelect1-1)} electrons over {deg_idx_1.size} pseudodegenerate atomic orbital')
        logger.debug(f'  Frac occupation: {frac_1: .3f} \n')
        # Update occupation for degenerate orbitals
        sub_e_1 = np.zeros(len(frags))
        add_e_1 = np.zeros(len(frags))
        for i in deg_idx_1:
            add_e_1[occs_1[i]] += frac_1
            if i < nelect1:
                sub_e_1[occs_1[i]] += 1
        frag_occs_1 -= sub_e_1
        frag_occs_1 += add_e_1

        # Update with new occupations
        diff_1 = deg_idx_1.size - np.sum(deg_idx_1 <= nelect1-1)

        frag_effo_occ_1 = s1[:nelect1 + diff_1]
        frag_effo_unocc_1 = s1[nelect1 + diff_1:]

        lo_1 = (nelect1 + diff_1) - 1  # Python index
        fu_1 = nelect1 + diff_1  # Python index

    if eig_2_r.index(eig_2_r[nelect2]) > 1:
        logger.info('  EOS: Warning, pseudo-degeneracies detected for Beta/Unpaired effos')
        deg_idx_2 = np.array([i for i, x in enumerate(eig_2_r) if x == eig_2_r[nelect2]])
        frac_2 = np.sum(deg_idx_2 <= nelect2 - 1) / deg_idx_2.size
        logger.info(f'  Distributing {np.sum(deg_idx_2 <= nelect2 - 1)} electrons over {deg_idx_2.size} pseudodegenerate atomic orbital')
        logger.debug(f'  Frac occupation: {frac_2: .4f} \n')
        # Update occupation for degenerate orbitals
        sub_e_2 = np.zeros(len(frags))
        add_e_2 = np.zeros(len(frags))
        for i in deg_idx_2:
            add_e_2[occs_2[i]] += frac_2
            if i < nelect2:
                sub_e_2[occs_2[i]] += 1
        frag_occs_2 -= sub_e_2
        frag_occs_2 += add_e_2

        # Update with new occupations
        diff_2 = deg_idx_2.size - np.sum(deg_idx_2 <= nelect2 - 1)

        frag_effo_occ_2 = s1[:nelect2 + diff_2]
        frag_effo_unocc_2 = s1[nelect2 + diff_2:]

        lo_2 = (nelect2 + diff_2) - 1  # Python index
        fu_2 =  (nelect2 + diff_2)  # Python index

    # Molecule LO/FU
    # lo_1 = nelect1 - 1  # Python index
    # lo_2 = nelect2 - 1  # Python index
    # fu_1 = nelect1  # Python index

    effos_data['dens1']['frags_effo_occ_1'] = frag_effo_occ_1
    effos_data['dens2']['frags_effo_occ_2'] = frag_effo_occ_2

    effos_data['dens1']['frags_effo_unocc_1'] = frag_effo_unocc_1
    effos_data['dens2']['frags_effo_unocc_2'] = frag_effo_unocc_2

    # Gather LO/FU information
    frag_lo_1 = []
    frag_fu_1 = []
    frag_lo_2 = []
    frag_fu_2 = []

    # loop over fragments:
    # Todo: check if this could be done more efficiently
    for f1 in range(len(frags)):
        # Deal with empty list
        if frag_effo_occ_1:
            for i_e, effo in enumerate(frag_effo_occ_1[::-1]):
                # Assuming frags_effo_occ_1 is ordered
                if f1 == effo[1]:
                    frag_lo_1.append((f1, effo[0], 'lo'))
                    break
                elif i_e+1 >= len(frag_effo_occ_1[::-1]):
                    frag_lo_1.append(None)
        else:
            frag_lo_1 = [None]*len(frags)
        # if f1 == s1[lo_1][1]:
        #     frag_lo_fu_1.append((f1, s1[lo_1][0], 'lo'))
        # else:
        if frag_effo_unocc_1:
            for i_e, effo in enumerate(frag_effo_unocc_1):
                if f1 == effo[1]:
                    frag_fu_1.append((f1, effo[0], 'fu'))
                    break
                elif i_e+1 >= len(frag_effo_unocc_1):
                    frag_fu_1.append(None)
        else:
            frag_fu_1 = [None]*len(frags)
    # for f1 in range(len(frags)):
    #     if f1 == s1[lo_1][1]:
    #         frag_lo_fu_1.append((f1, s1[lo_1][0], 'lo'))
    #     else:
    #         for effo in frags_effo_unocc_1:
    #             if f1 == effo[1]:
    #                 frag_lo_fu_1.append((f1, effo[0], 'fu'))
    #                 break

    for f2 in range(len(frags)):
        # Deal with empty list
        if frag_effo_occ_2:
            for i_e, effo in enumerate(frag_effo_occ_2[::-1]):
                # Assuming frags_effo_occ_1 is ordered
                if f2 == effo[1]:
                    frag_lo_2.append((f2, effo[0], 'lo'))
                    break
                elif i_e + 1 >= len(frag_effo_occ_2[::-1]):
                    frag_lo_2.append(None)
        else:
            frag_lo_2 = [None]*len(frags)
        # if f1 == s1[lo_1][1]:
        #     frag_lo_fu_1.append((f1, s1[lo_1][0], 'lo'))
        # else:
        if frag_effo_unocc_2:
            for i_e, effo in enumerate(frag_effo_unocc_2):
                if f2 == effo[1]:
                    frag_fu_2.append((f2, effo[0], 'fu'))
                    break
                elif i_e + 1 >= len(frag_effo_unocc_2):
                    frag_fu_2.append(None)
        else:
            frags_effo_occ_2 = [None]*len(frags)



    r1 = 100 * min(1, s1[lo_1][0] - s1[fu_1][0] + 0.5)
    r2 = 100 * min(1, s2[lo_2][0] - s2[fu_2][0] + 0.5)

    if gen:
        logger.info(f'  Reliability index for Paired : {r1: .4f}')
        logger.info(f'  Reliability index for Unpaired : {r2: .4f}')
    else:
        logger.info(f'  Reliability index for Alpha : {r1: .4f}')
        logger.info(f'  Reliability index for Beta : {r2: .4f}')



    effos_data['dens1']['frag_occs_1'] = frag_occs_1
    effos_data['dens2']['frag_occs_2'] = frag_occs_2

    effos_data['dens1']['frag_lo_1'] = frag_lo_1
    effos_data['dens2']['frag_lo_2'] = frag_lo_2

    effos_data['dens1']['frag_fu_1'] = frag_fu_1
    effos_data['dens2']['frag_fu_2'] = frag_fu_2

    return effos_data



def dump_molden(APOSTBase, filename, eigval, eigvec, eta_frags, mol, thresh=0.01, spin='alpha', ovlp=None, transform_m=None, file_molden=None):
    """FIRST DRAFT: Function to dump molden file for effos visualization
    Adapted from APOSTpy functions local_dump_scf and local orbital coeff. Those are directly taken
    from pyscf.tools.molden
    """
    from pyscf.tools.molden import remove_high_l, order_ao_index
    from pyscf import __version__, lib

    # todo: Check that mol is a pyscf

    # Pass eigenvec, eigenvalue, and obtain the coeff_Efo that should substitute mo_coeff below
    # We could pass the part_name and mol instead of the APOSTBase object
    thresh_coeff = 1e-8

    # Todo: Change below when we save both molecular coefficients from the calc and the natural orbital basis
    if APOSTBase._sd and APOSTBase._unrestr:
        dm = APOSTBase.dm_a + APOSTBase.dm_b
        occ, occ_coeff = APOSTBase.generate_mo_coeff(dm, overlap=APOSTBase.ovlp)

        occ_coeff = occ_coeff[:, occ > thresh_coeff]
        occ = occ[occ > thresh_coeff]
    else:
        occ_coeff = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh_coeff]
        occ = APOSTBase.occ_a[APOSTBase.occ_a > thresh_coeff]


    if APOSTBase.part.part_name == 'mulliken':

        # we could do further checks
        if not isinstance(ovlp, np.ndarray):
            raise TypeError('For mulliken effos the overlap matrix is needed')

        eta_coeff = np.einsum("fbc,ci->fbi", eta_frags, occ_coeff)
        t2 = np.einsum("ab,fbi->fai", ovlp, eta_coeff)
        t3 = np.einsum("ja,fai->fji", occ_coeff.T, t2)
        # t3 should be in the base of MOs
        assert t3.shape[1] == occ_coeff.shape[1]
        assert t3.shape[2] == occ_coeff.shape[1]
        logger.debug('POP FRAGMENT: ', np.trace(t3, axis1=1, axis2=2))

        # DEV: Transform effao coefficients (eigvec) from MOs basis to AOs basis for the Mulliken
        # eigvec: array(F,MO,EFFO)
        # eta_coeff: array(F, AO, MO)
        effo_coeff_ao = np.einsum('fbk,fki ->fbi', eta_coeff, eigvec)

        # effo_coeff_ao should have the same dimensions as eta coeff
        assert effo_coeff_ao.shape[1] == occ_coeff.shape[0]
        assert effo_coeff_ao.shape[2] == occ_coeff.shape[1]

    else:
        # ORTHONORMAL BASIS
        if transform_m:
            t, t_inv = transform_m
            if not t.shape == t_inv.shape:
                raise ValueError('Transformation matrix must have the same shape as the inverse')
        else:
            raise ValueError('Effos in orthogonal basis needs a transformation matrix')

        # Equation to obtain the Effaos for Orthogonal Hilbert partitons (e.g Lowdin) expressed in the original non-orthogonal AO base
        # \chi_k = \frac{1}{\sqrt{\lambda_k}} \sum_{l}\sum_{\nu\sigma\mu i} S^{-\frac{1}{2}}_{l \nu} \eta_{\nu \sigma} S^{\frac{1}{2}}_{\sigma \mu} C_{\mu i} U_{ik} \phi_{l}(r)
        # Transform MOs from LCAO(AO non-orthogonal) to Orthogonal AO basis: S^{\frac{1}{2}}_{\sigma \mu} C_{\mu i}
        orth_coeff = np.einsum('la,ai -> li', t, occ_coeff)
        if spin == 'paired':
            print('Check Total Nelect:', 2* np.trace(np.dot(orth_coeff.T, orth_coeff)))  # total electrons
        elif spin == 'unpaired':
            print('Check Total Nelect:',  np.trace(np.dot(orth_coeff.T, orth_coeff)))  # total electrons
        else:
            print('Check Total Nelect:', np.trace(np.dot(orth_coeff.T, orth_coeff)))  # total electrons
        # Filter for only fragment basis functions: \eta_{\nu \sigma} S^{\frac{1}{2}}_{\sigma \mu} C_{\mu i}
        eta_orth_coeff = np.einsum('fla, ai -> fli', eta_frags, orth_coeff)
        pop = np.trace(np.einsum('jl, fli -> fji', orth_coeff.T, eta_orth_coeff), axis1=1, axis2=2)
        # print('pop', pop)
        logger.debug('Population fragment: ' +' '.join(str(round(v,4)) for v in pop))

        # DEV: Transform effao coefficients (eigvec) from MOs basis to AOs basis for the Orthogonal basis
        # eigvec: array(F,MO,MO)
        # eta_coeff: array(F,AO, MO)
        # Expressing Effao in the orthogonal base for fragment: \eta_{\nu \sigma} S^{\frac{1}{2}}_{\sigma \mu} C_{\mu i} U_{ik}
        eta_orth_coeff_effao = np.einsum('fbi, fik -> fbk', eta_orth_coeff, eigvec)
        # Getting back to AOs: S^{-\frac{1}{2}}_{l \nu} \eta_{\nu \sigma} S^{\frac{1}{2}}_{\sigma \mu} C_{\mu i} U_{ik}
        eta_orth_coeff_effao_ao = np.einsum('lb, fbk -> flk', t_inv, eta_orth_coeff_effao)
        effo_coeff_ao = eta_orth_coeff_effao_ao

    # open a file
    # Always removing high angular basis functions
    mol_r, mo_coeff_r = remove_high_l(mol, occ_coeff)
    # Reorder coefficients for Molden files
    aoidx = order_ao_index(mol_r)
    if file_molden:
        with open(file_molden, 'a') as fout:
            icount = 0
            for f in range(eigval.shape[0]):
                for imo in range(eigval.shape[1]):
                    if (eigval[f, imo] > thresh):
                        icount += 1
                        fout.write(' Sym= %s\n' % 'A')
                        fout.write(' Ene= %10.5f\n' % eigval[f, imo])
                        fout.write(' Spin= %s\n' % spin)
                        fout.write(' Occup= %10.5f\n' % eigval[f, imo])
                        for i, j in enumerate(aoidx):
                            fout.write(' %3d    %18.14g\n' % (i + 1, effo_coeff_ao[f, j, imo]))
            print("Writting", icount, spin, "EFOs on molden file. Threshold: ", thresh)

    else:
        with open(filename, 'w') as fout:
            if mol.cart:
                # pyscf Cartesian GTOs are not normalized. This may not be consistent
                # with the requirements of molden format. Normalize Cartesian GTOs here
                norm = mol.intor('int1e_ovlp').diagonal() ** .5
                occ_coeff = np.einsum('i,ij->ij', norm, occ_coeff)

            # Write header
            fout.write('[Molden Format]\n')
            fout.write('made by pyscf v[%s]\n' % __version__)
            fout.write('[Atoms] (AU)\n')
            for ia in range(mol_r.natm):
                symb = mol_r.atom_pure_symbol(ia)
                chg = mol_r.atom_nelec_core(ia) + mol_r.atom_charge(ia)
                fout.write('%s   %d   %d   ' % (symb, ia + 1, chg))
                coord = mol_r.atom_coord(ia)
                fout.write('%18.14f   %18.14f   %18.14f\n' % tuple(coord))

            fout.write('[GTO]\n')
            for ia, (sh0, sh1, p0, p1) in enumerate(mol_r.offset_nr_by_atom()):
                fout.write('%d 0\n' % (ia + 1))
                for ib in range(sh0, sh1):
                    l = mol_r.bas_angular(ib)
                    nprim = mol_r.bas_nprim(ib)
                    nctr = mol_r.bas_nctr(ib)
                    es = mol_r.bas_exp(ib)
                    cs = mol_r.bas_ctr_coeff(ib)
                    for ic in range(nctr):
                        fout.write(' %s   %2d 1.00\n' % (lib.param.ANGULAR[l], nprim))
                        for ip in range(nprim):
                            fout.write('    %18.14g  %18.14g\n' % (es[ip], cs[ip, ic]))
                fout.write('\n')

            if mol_r.cart:
                fout.write('[6d]\n[10f]\n[15g]\n')
            else:
                fout.write('[5d]\n[7f]\n[9g]\n')

            if mol_r.has_ecp():  # See https://github.com/zorkzou/Molden2AIM
                fout.write('[core]\n')
                for ia in range(mol_r.natm):
                    nelec_ecp_core = mol_r.atom_nelec_core(ia)
                    if nelec_ecp_core != 0:
                        fout.write('%s : %d\n' % (ia + 1, nelec_ecp_core))
            fout.write('\n')

            # if spin == 'alpha' or spin == 'unpaired':
            #     # Avoid duplicated [MO] session when dumping beta /paired orbitals
            #     fout.write('[MO]\n')
            fout.write('[MO]\n')

            # Add to function arguments
            icount = 0
            for f in range(eigval.shape[0]):
                for imo in range(eigval.shape[1]):
                    if eigval[f, imo] > thresh:
                        icount += 1
                        fout.write(' Sym= %s\n' % 'A')
                        fout.write(' Ene= %10.5f\n' % eigval[f, imo])
                        fout.write(' Spin= %s\n' % spin)
                        fout.write(' Occup= %10.5f\n' % eigval[f, imo])
                        for i, j in enumerate(aoidx):
                            fout.write(' %3d    %18.14g\n' % (i + 1, effo_coeff_ao[f,j,imo]))
            print("Writting", icount, spin, "EFOs on molden file. Threshold: ", thresh)


    # print('WIP: dump molden file')
    #1. get effos in the AO base
    #2. write header
    #3. reorder coefficients for Molden ordering using order_ao_index(). order_ao_index breaks for l=5. For now we could
    # rely on this to get a heads up that we are not applying the header?
    #4 print effos + coefficients above threshold. For cart we need to normalize with the sqrt of the overlap matrix
    # header(mol, mol.stdout)
    # print(order_ao_index(mol))
    # orbital_coeff(mol, mol.stdout, m.mo_coeff)



# def dump_fchk():
#     """WIP: function to dump fchk file for effos visualization
#     """



def EOS_calc(APOSTBase, gen=False, vis=None, log='info'):
    """WIP:Perform EOS calculation
    Prints results of Effective oxidation calculation
    Q: Do we return values? if we don't define an object and don't return values then we only get the printing
    """

    if log == 'debug':
        logger.setLevel(level=logging.DEBUG)

    # Obtain frags information from APOSTBase.part object
    frags = APOSTBase.part.frags

    # Todo: printing 1: EOS information
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info('  Effective oxidation calculation  ')
    logger.info('  Information ')

    if len(frags)  < APOSTBase.natm:
        logger.info(f'  Fragments: {len(APOSTBase.part.frags)} ')
        for idx, f1 in enumerate(APOSTBase.part.frags):
            logger.info(f'  Fragment {idx}: Atoms {f1}, Total charge {APOSTBase.part.zfrags[idx]} ')
    else:
        logger.info(f'  Atomic EOS ')

    logger.info("")
    logger.info(f'  AIM Definition: {APOSTBase.part.part_name.upper()}')

    logger.info(f"  Doing generalized EOS: {gen}")

    logger.info("")
    logger.info("")
    logger.info("")

    if isinstance(APOSTBase.part, Hpart):
        # Generate SMOs for fragments, partition and type of EOS

        logger.info("  Generating SMOs")
        logger.info("")
        logger.info("")

        smo_1, smo_2 = smo_hilb(APOSTBase, gen=gen)

    elif isinstance(APOSTBase.part, Rpart):
        logger.info("  Generating Qij matrices")
        logger.info("")
        logger.info("")

        smo_1, smo_2 = qij_real(APOSTBase, gen=gen)

    else:
        raise ValueError("Unknown type of part object")

    # Get EOS
    if gen:
        nel2 = abs(int(APOSTBase.alpha_elec - APOSTBase.beta_elec)) # Unpaired electrons
        nel1 = int((APOSTBase.alpha_elec + APOSTBase.beta_elec - nel2) / 2) # Pairs of electrons
    else:
        nel1 = APOSTBase.alpha_elec
        nel2 = APOSTBase.beta_elec

    effos_data = effos_occ(smo_1, nel1, smo_2, nel2, frags=frags, gen=gen)
    # Assign values to variables
    s1_egval = effos_data['dens1']['egval']
    s2_egval = effos_data['dens2']['egval']

    s1_egvec = effos_data['dens1']['egvec']
    s2_egvec = effos_data['dens2']['egvec']

    frag_occs_1 = effos_data['dens1']['frag_occs_1']
    frag_occs_2 = effos_data['dens2']['frag_occs_2']

    frag_lo_1 = effos_data['dens1']['frag_lo_1']
    frag_lo_2 = effos_data['dens2']['frag_lo_2']

    frag_fu_1 = effos_data['dens1']['frag_fu_1']
    frag_fu_2 = effos_data['dens2']['frag_fu_2']

    # Compute oxidation state
    if gen:
        os = np.array(APOSTBase.part.zfrags) - 2*np.array(frag_occs_1) - np.array(frag_occs_2)
    else:
        os = np.array(APOSTBase.part.zfrags) - np.array(frag_occs_1) - np.array(frag_occs_2)

    logger.debug("  PRINTING OCCUPATIONS")
    logger.debug("  Alpha part\n")
    # For now lopping over the list of effos. todo: see if we can improve this code
    for i_f in range(len(frags)):
        t = []
        logger.debug(f"  **FRAGMENT  {i_f+1} **")
        logger.debug("  Net occupation using thresh>=0.01")
        for effo in effos_data['dens1']['s1']:
            if effo[1] == i_f and effo[0] > 0.01:
                t.append(effo[0])
        logger.debug('  OCC  ' + ' '.join(str(round(v, 4)) for v in t))
        logger.debug('\n')

    logger.debug("  Beta part\n")
    # For now lopping over the list of effos. todo: see if we can improve this code
    for i_f in range(len(frags)):
        t = []
        logger.debug(f"  **FRAGMENT  {i_f + 1} **")
        logger.debug("  Net occupation using thresh>=0.01")
        for effo in effos_data['dens2']['s2']:
            if effo[1] == i_f and effo[0] > 0.01:
                t.append(effo[0])
        logger.debug('  OCC  ' + ' '.join(str(round(v, 4)) for v in t))
        logger.debug('\n')


    logger.info("")
    logger.info("")
    logger.info("  EOS ANALYSIS")
    if gen:
        logger.info(f"  {'Frag': <4}  {'Paired':<7} {'Last occ':<8} {'First unocc':<11}   {'Unpaired':<9} {'Last occ':<8} {'First unocc':<11} {'OS':<3}")
    else:
        logger.info(f"  {'Frag': <4}  {'Alpha':<5} {'Last occ':<8} {'First unocc':<11}   {' Beta  ':<9} {'Last occ':<8} {'First unocc':<11} {'OS':<3}")
    logger.info("  "+"-" * 74)
    for f in range(len(frags)):
        # Todo: use a printing function to get rid of the repetition. Below code works for the brown ring 43paired/3unpaired def2svp b3lyp
        if frag_lo_1[f] and frag_lo_1[f][-1] == 'lo':
            if frag_fu_1[f] and frag_fu_1[f][-1] == 'fu':
                if frag_lo_2[f] and frag_lo_2[f][-1] == 'lo':
                    if frag_fu_2[f] and frag_fu_2[f][-1] == 'fu':
                        logger.info(f"  Frag{f:<4}  {round(frag_occs_1[f],1):<3}  {round(frag_lo_1[f][1], 3):<8} {round(frag_fu_1[f][1], 3):<11}   {round(frag_occs_2[f],0):<9} {round(frag_lo_2[f][1],3):<8} {round(frag_fu_2[f][1],3):<11} {os[f]:<3}")
                    else:
                        logger.info(f"  Frag{f:<4}  {round(frag_occs_1[f],1):<3}  {round(frag_lo_1[f][1], 3):<8} {round(frag_fu_1[f][1], 3):<11}   {round(frag_occs_2[f],0):<9} {round(frag_lo_2[f][1],3):<8} {'-':<11} {os[f]:<3}")
                else:
                    if frag_fu_2[f] and frag_fu_2[f][-1] == 'fu':
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {round(frag_lo_1[f][1], 3):<8} {round(frag_fu_1[f][1], 3):<11}   {round(frag_occs_2[f], 0):<9} {'-':<8} {round(frag_fu_2[f][1], 3):<11} {os[f]:<3}")
                    else:
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {round(frag_lo_1[f][1], 3):<8} {round(frag_fu_1[f][1], 3):<11}   {round(frag_occs_2[f], 0):<9} {'-':<8} {'-':<11} {os[f]:<3}")
            else:
                if frag_lo_2[f] and frag_lo_2[f][-1] == 'lo':
                    if frag_fu_2[f] and frag_fu_2[f][-1] == 'fu':
                        logger.info(f"  Frag{f:<4}  {round(frag_occs_1[f],1):<3}  {round(frag_lo_1[f][1], 3):<8} {'-':<11}   {round(frag_occs_2[f],0):<9} {round(frag_lo_2[f][1],3):<8} {round(frag_fu_2[f][1],3):<11} {os[f]:<3}")
                    else:
                        logger.info(f"  Frag{f:<4}  {round(frag_occs_1[f],1):<3}  {round(frag_lo_1[f][1], 3):<8} {'-':<11}   {round(frag_occs_2[f],0):<9} {round(frag_lo_2[f][1],3):<8} {'-':<11} {os[f]:<3}")
                else:
                    if frag_fu_2[f] and frag_fu_2[f][-1] == 'fu':
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {round(frag_lo_1[f][1], 3):<8} {'-':<11}   {round(frag_occs_2[f], 0):<9} {'-':<8} {round(frag_fu_2[f][1], 3):<11} {os[f]:<3}")
                    else:
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {round(frag_lo_1[f][1], 3):<8} {'-':<11}   {round(frag_occs_2[f], 0):<9} {'-':<8} {'-':<11} {os[f]:<3}")
        else:
            if frag_fu_1[f] and frag_fu_1[f][-1] == 'fu':
                if frag_lo_2[f] and frag_lo_2[f][-1] == 'lo':
                    if frag_fu_2[f] and frag_fu_2[f][-1] == 'fu':
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {'-':<8} {round(frag_fu_1[f][1], 3):<11}   {round(frag_occs_2[f], 0):<9} {round(frag_lo_2[f][1], 3):<8} {round(frag_fu_2[f][1], 3):<11} {os[f]:<3}")
                    else:
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {'-':<8} {round(frag_fu_1[f][1], 3):<11}   {round(frag_occs_2[f], 0):<9} {round(frag_lo_2[f][1], 3):<8} {'-':<11} {os[f]:<3}")
                else:
                    if frag_fu_2[f] and frag_fu_2[f][-1] == 'fu':
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {'-':<8} {round(frag_fu_1[f][1], 3):<11}   {round(frag_occs_2[f], 0):<9} {'-':<8} {round(frag_fu_2[f][1], 3):<11} {os[f]:<3}")
                    else:
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {'-':<8} {round(frag_fu_1[f][1], 3):<11}   {round(frag_occs_2[f], 0):<9} {'-':<8} {'-':<11} {os[f]:<3}")
            else:
                if frag_lo_2[f] and frag_lo_2[f][-1] == 'lo':
                    if frag_fu_2[f] and frag_fu_2[f][-1] == 'fu':
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {'-':<8} {'-':<11}   {round(frag_occs_2[f], 0):<9} {round(frag_lo_2[f][1], 3):<8} {round(frag_fu_2[f][1], 3):<11} {os[f]:<3}")
                    else:
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {'-':<8} {'-':<11}   {round(frag_occs_2[f], 0):<9} {round(frag_lo_2[f][1], 3):<8} {'-':<11} {os[f]:<3}")
                else:
                    if frag_fu_2[f] and frag_fu_2[f][-1] == 'fu':
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {'-':<8} {'-':<11}   {round(frag_occs_2[f], 0):<9} {'-':<8} {round(frag_fu_2[f][1], 3):<11} {os[f]:<3}")
                    else:
                        logger.info(
                            f"  Frag{f:<4}  {round(frag_occs_1[f], 1):<3}  {'-':<8} {'-':<11}   {round(frag_occs_2[f], 0):<9} {'-':<8} {'-':<11} {os[f]:<3}")



    # we need to sort out the calculation type to pass the correct mo_coeff
    if vis and isinstance(APOSTBase.part, Hpart):
        if vis == 'Molden':
            # APOSTBase, filename, mo_coeff, fra_list, eig_list, spin = 'unpaired', thresh = 0.01, ignore_h = True
            eta_frags = APOSTBase.part.eta_frags
            ovlp = APOSTBase.ovlp
            dump_molden(APOSTBase, 'test_molden', s1_egval, s1_egvec, eta_frags, APOSTBase._mol, spin='paired', ovlp=ovlp, transform_m=[APOSTBase.part.T, APOSTBase.part.T_inv])
            dump_molden(APOSTBase, 'test_molden', s2_egval, s2_egvec, eta_frags, APOSTBase._mol, spin='unpaired', ovlp=ovlp, transform_m=[APOSTBase.part.T, APOSTBase.part.T_inv], file_molden='test_molden')
        elif vis == 'fchk':
            raise NotImplementedError
        else:
            raise TypeError('Dump file format not identified. Please choose from: Molden, fchk')
    else:
        raise NotImplementedError()

