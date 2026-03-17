# IQA module in real space
import copy as copy
import numpy as np
import logging

from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.libxc import eval_xc, is_hybrid_xc, hybrid_coeff


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


def ee_hf_energy(at_nums, at_coords, mol, dm, grid_coords, eval_ao_grid, mo_coeff, occ, qw, aim_w, atgrid_idx):
    r"""Compute Hartree Fock electron-electron interaction energy."""

    # Follow libcint approach, loop for each charge
    # Compute molecular orbitals numerically
    # Note: mo_coeff filtered by occ
    mo_grid = np.einsum('pa,ai->ip ', eval_ao_grid, mo_coeff)
    dens_grid = np.zeros((mo_grid.shape[1]))
    for i in range(mo_grid.shape[0]):
        dens_grid += mo_grid[i, :] * mo_grid[i, :] * occ[i]

    total_elect_int = np.sum(dens_grid * qw *aim_w)

    print(f'  Check integrated elect: {total_elect_int: .5f}\n')

    # For now explicit loop but if it works look into applying a funct to a list of points
    chunk_size = 250000000 // (mol.nao_nr() ** 2)
    print('chunk_size', chunk_size)
    istart = 0
    t_exch_num = np.zeros((grid_coords.shape[0]))
    t_coul_num = np.zeros((grid_coords.shape[0]))
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
        pc_rho =  np.trace(np.tensordot(dm, (chunk_pc * chunk_dens), axes=(1, 0)))
        coul_chunk = -0.5*pc_rho
        # occupied_mo[molecule.mo.occsa > 0] = 1
        t1 = np.einsum("abn,ai->ibn", chunk_pc, mo_coeff)
        t1 = np.einsum("ibn,bj->ijn", t1, mo_coeff)
        exch_chunk = np.einsum("ijn,in,jn->n", t1, mo_grid, mo_grid)

        t_coul_num[istart:iend] = coul_chunk
        t_exch_num[istart:iend] = exch_chunk
        istart = iend

        del t1

    t_coul = np.sum(t_coul_num*qw*aim_w)
    t_exch = np.sum(t_exch_num*qw*aim_w)

    print(f'TOTAL COULOMB {t_coul: .5f}')
    print(f'TOTAL EXCHANGE {t_exch: .5f}')

    # Compute atomic values
    print(at_nums.size)
    at_coul = np.zeros((at_nums.size))
    at_exch = np.zeros((at_nums.size))
    for at in range(len(at_nums)):

        # print(np.sum(dens_grid[atgrid_idx == at] * qw[atgrid_idx == at] * aim_w[atgrid_idx == at]))
        at_c = np.sum(t_coul_num[atgrid_idx == at] * qw[atgrid_idx == at] * aim_w[atgrid_idx == at])
        at_x = np.sum(t_exch_num[atgrid_idx == at] * qw[atgrid_idx == at] * aim_w[atgrid_idx == at])
        at_coul[at] = at_c
        at_exch[at] = at_x

    print('DEV ATOMIC VALUES for semianalitic ee terms')
    print(at_coul)
    print(at_exch)
    print('CHECK: SUM OF ATOMIC VALUES')
    print(np.sum(at_coul))
    print(np.sum(at_exch))

    return t_coul, t_exch, at_coul, at_exch


def ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, aim_w, atgrid_idx, spin=0):

    print('DEV dft exchange-correlation')

    # 1. get the functional string
    # 2. call eval_xc to get the energy density
    # 3. Check for an example of each type if we compute the correct total energy density integrating
    # correctly

    print('XC CODE AND TYPE :', xc_code, xc_type)
    # prepare rho to pass to eval_xc. rho contains necessary components e.g. dens, grad
    # todo: we need to sort the spin too
    # todo: we need to sort xc, x and c individually
    # rho includes necessary density quantities for each type of functional
    # LDA: density
    # GGA: density, grad_x, grad_y, grad_z
    # There are several functions whose goal is to evaluate the DFT exchange-correlation energy density but ultimately
    # all of them end up on the libxc module interface calling eval_xc which calls the protected method _eval_xc (calls the C++ routine)
    # eval_xc returns ex(energy density), vxc(first derivatives of the xc energy with respect density, gradient, lap and kin, fxc(second derivs), kxc(third derivs)
    # shape of ex, vxc, fxc and kxc different when spin=0 or 1
    r = eval_xc(xc_code, rho, spin=spin)
    xc_e_dens = r[0]
    # rho[0] = density
    xc_e = np.sum(xc_e_dens * rho[0] *  qw * aim_w)
    print('TOTAL DFT EXCHANGE-CORRELATION ENERGY : ', xc_e)
    # get HF exchange contribution and alpha, beta and omega parameters if necessary
    # Todo: for now only dealing with hybrid but not range separated
    coeff_hf = None
    if is_hybrid_xc(xc_code):
        coeff_hf = hybrid_coeff(xc_code, spin=spin)

    # Atomic contributions
    at_dft = np.zeros((at_nums.size))
    for at in range(len(at_nums)):
        at_dft[at] = np.sum(xc_e_dens[atgrid_idx == at] *rho[0][atgrid_idx == at]* qw[atgrid_idx == at] * aim_w[atgrid_idx == at])

    print(f'DEV ATOMIC VALUES for DFT functional {xc_code}')
    print(at_dft)
    print('CHECK: SUM OF ATOMIC VALUES')
    print(np.sum(at_dft))

    return xc_e, at_dft, coeff_hf

def ee_hf_energy_interatomic(at_nums, grid_coords1, grid_coords2, mo_grid1, mo_grid2, qw, aim_w1, aim_w2, atgrid_idx):
    """Function to compute Coulomb and HF exchange numerically"""
    print('DEV ee hf energy interatomic')

    t_coul = np.zeros((len(at_nums), len(at_nums)))
    t_exch = np.zeros((len(at_nums), len(at_nums)))


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
            rij_small = 0
            for id_p1, p1 in enumerate(atgrid_idx_a):
                # Loop only for occupied orbitals
                coul_dens = 0
                exch_dens = 0
                for i in range(mo_grid1.shape[0]):
                    for j in range(mo_grid2.shape[0]):
                        coul_dens += 2 * mo_grid1[i, p1] * mo_grid1[i, p1] * mo_grid2[j, atgrid_idx_b ] * mo_grid2[j, atgrid_idx_b]
                        exch_dens += mo_grid1[i, p1] * mo_grid1[j, p1] * mo_grid2[j, atgrid_idx_b] * mo_grid2[i, atgrid_idx_b]
                # Compute rij between i point index (p1) and atom b atgrid points
                rij = np.linalg.norm(a_grid_coords[id_p1, :] - b_grid_coords, axis=-1)

                coul_dens = coul_dens / rij
                exch_dens = exch_dens / rij
                # filter values of the points wit rij < 1e-3. If not filtered they add spurious big values
                coul_dens[rij<1e-5] = 0
                exch_dens[rij<1e-5] = 0
                rij_small += rij[rij<1e-5].shape[0]
                # Now integration starts. Get aim weights for p1 atgridA and all points in atgridB
                coul_b = np.sum(coul_dens* qw[atgrid_idx_b] * aim_w2[atgrid_idx_b]) # integrated for atB
                exch_b = np.sum(exch_dens* qw[atgrid_idx_b] * aim_w2[atgrid_idx_b]) # integrated for atB
                coul = coul_b * qw[p1] * aim_w1[p1]
                exch = exch_b * qw[p1] * aim_w1[p1]
                if ia != ib:
                    t_coul[ia, ib] += 2*coul
                    t_exch[ia, ib] += 2*exch
                    t_coul[ib, ia] = t_coul[ia, ib]
                    t_exch[ib, ia] = t_exch[ia, ib]
                else:
                    t_coul[ia, ib] += coul
                    t_exch[ia, ib] += exch

            print(f'DEBUG: TOTAL NUMBER OF POINTS WITH Rij < 1e3 : {rij_small}')
    print('SUMMARY TWO ELECTRON NUMERICAL INTEGRATION')
    print(f'*** COULOMB PART ***')
    print(t_coul)

    # Q: If pure functional theoretically we don't need to compute it. However the bootleneck right now should be
    # the `for` loop so for the same price we can integrate both
    print(f'*** HF EXCHANGE PART ***')
    print(-t_exch)
    print()
    return t_coul, -t_exch








def iqa_interatomic(APOSTBase, f_corr=None, f_exch=None):
    """WIP: Function to obtain two electron dependent energy quantities and
    interatomic components for DFT xc energy based on the BOD method"""

    at_coords = APOSTBase.at_coords
    at_nums = np.array(APOSTBase.at_nums)

    # Grid info
    grid_coords = APOSTBase.part.grid_coords
    qw = APOSTBase.part._grid.quadrature_weights
    mol_w = APOSTBase.part.aim_w
    atgrid_idx = APOSTBase.part.grid_to_atom_idx
    # mg = APOSTBase.part.grid_weights

    # Density info. For now only SD
    thresh = 1e-8
    occ_coeff_a = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
    occ_coeff_b = APOSTBase.mo_coeff_b[:, APOSTBase.occ_a > thresh]
    occ_a = APOSTBase.occ_a[APOSTBase.occ_a > thresh]
    occ_b = APOSTBase.occ_b[APOSTBase.occ_b > thresh]
    dm = APOSTBase.dm_a

    # Compute interatomic terms for HF exchange/coulomb and DFT
    # for each point in grid_1 (Atom A) integrate all points in grid_2 (Atom B).
    # We need to 1.Generate a second grid and 2.Atomic weights (quadrature) are the same for both grids as they depend only on the distance
    # 3. Regenerate AIM weights
    # Rotate specific angle in rads
    # 1. Get points and compute the polar angles theta and phi
    grid_coords1 = copy.copy(APOSTBase.part.grid_coords)
    print('DEV HF 2 electron numerical integration')
    print(f'TOTAL NUMBER OF GRID1 POINTS: {grid_coords1.shape}')

    # Calling method to rotate the grid coords
    pha = 0
    phb = 0.162
    # Note for the code:
    # APOST3D code: APOST3D generates the rotated grid not by rotating grid1 but by starting from scratch.
    # It generates distance and theta,fix for the same radial and angular points and then instead of passing 0 for pha and phb
    # it passes the specified value. This second call of the prenumint rutine is only called inside enpart.f (and only if numerical 2e integration
    # is requested, not for semi analytical IQA). Once the theta and fix angles have been rotated the atomic coordinates are addedd
    # Here because we rely on the Grid pyscf machinery to get the points and the atomic(quadrature) weights before rotating the points
    # we substract the atomic coordinates, and then rotate them again. After that we generate a second set of AIM weights for the rotated grid but keep the same
    # atomi(quadrature) weights.

    grid_coords2, aim_n = APOSTBase.part.rotate_grid(at_coords, at_nums, pha, phb, dm)
    print(f'DEV: ROTATED GRID AND AIM: {grid_coords2.shape}, {aim_n.shape}')

    # Evaluating molecular orbitals on Grid1 and Grid2

    eval_ao_grid1 = eval_ao(APOSTBase._mol, grid_coords1, deriv=0)
    eval_ao_grid2 = eval_ao(APOSTBase._mol, grid_coords2, deriv=0)
    # Note: DEV happening with coeff a because restricted closed shell. Todo:  extend to beta
    eval_mo_grid1 = np.einsum('pa,ai->ip ', eval_ao_grid1, occ_coeff_a)
    eval_mo_grid2 = np.einsum('pa,ai->ip ', eval_ao_grid2, occ_coeff_a)
    dens_grid_1 = np.zeros((eval_mo_grid1.shape[1]))
    dens_grid_2 = np.zeros((eval_mo_grid2.shape[1]))
    for i in range(eval_mo_grid1.shape[0]):
        dens_grid_1 += eval_mo_grid1[i, :] * eval_mo_grid1[i, :] * occ_a[i]
        dens_grid_2 += eval_mo_grid2[i, :] * eval_mo_grid2[i, :] * occ_a[i]

    total_elect_int_1 = np.sum(dens_grid_1 * mol_w * qw)
    total_elect_int_2 = np.sum(dens_grid_2 * qw * aim_n)
    print('CHECK TOTAL ELECT INT 1', total_elect_int_1)
    print('CHECK TOTAL ELECT INT 1', total_elect_int_2)


    coul_n, exch_n = ee_hf_energy_interatomic(at_nums, grid_coords1, grid_coords2, eval_mo_grid1, eval_mo_grid2, qw, mol_w,aim_n, atgrid_idx)

    #analytic value for 2elec integrals coulomb and hf exchange
    two_eint = APOSTBase._mol.intor('int2e')
    coul_1 = np.einsum("ab,abcd,cd->ad", dm, two_eint, dm)
    exch_1 = np.einsum("ac,abcd,bd->ad", dm, two_eint, dm)
    coul_analyt = 0.5 * np.sum(np.einsum("ad->a", coul_1))
    exch_analyt = -0.25 * np.sum(np.einsum("ad->a", exch_1))

    iqa_inter_r = {}
    iqa_inter_r['t_coul'] = np.sum(np.triu(coul_n))
    iqa_inter_r['t_exch'] = np.sum(np.triu(exch_n))

    print('TOTAL ENERGY SUM OF COMPONENTS ')
    print('COULOMB', f"{iqa_inter_r['t_coul']: .5f}")
    print(f'COULOMB (PySCF/Libcint): {coul_analyt: .5f}')
    diff_c = iqa_inter_r['t_coul'] - coul_analyt
    print('DIFF ', diff_c)
    print('DIFF (kcal/mol) ', diff_c / 0.0015936014376406278)
    print()
    print('HF EXCHANGE', f"{iqa_inter_r['t_exch']: .5f}")
    print(f'HF EXCHANGE (PySCF/Libcint): {exch_analyt: .5f}')
    diff_x = iqa_inter_r['t_exch'] - exch_analyt
    print('DIFF', diff_x )
    print('DIFF (kcal/mol) ', diff_x / 0.0015936014376406278)
    print('TOTAL DIFF', (diff_c + diff_x))
    print('TOTAL DIFF (kcal/mol)', (diff_c + diff_x)/ 0.0015936014376406278)

def iqa_atomic(APOSTBase, f_corr=None, f_exch=None):
    """WIP: Function to obtain one electron dependent energy quantities"""

    # NN energy: Coulomb formula of charge repulsion-Analytic (Done)
    # EN energy: Compute density (Done)
    # Kinetic energy: Obtain density and density gradients (Done)
    # Exchange and Coulomb HF (pseudo analytical with only one electron numerical): Use the point charge integral within pyscf (Done)
    # DFT exchange-correlation (Done)


    print('DEV" ATOMIC IQA CALC')
    # Molecule info
    at_coords = APOSTBase.at_coords
    at_nums = np.array(APOSTBase.at_nums)



    # Grid info
    grid_coords = APOSTBase.part.grid_coords
    qw = APOSTBase.part._grid.quadrature_weights
    mol_w = APOSTBase.part.aim_w
    atgrid_idx = APOSTBase.part.grid_to_atom_idx
    mg = APOSTBase.part.grid_weights

    #Orbitals and density
    # For now doing SD
    thresh = 1e-8
    occ_coeff = APOSTBase.mo_coeff_a[:, APOSTBase.occ_a > thresh]
    occ = APOSTBase.occ_a[APOSTBase.occ_a > thresh]
    dm = APOSTBase.dm_a

    # Dens info
    xc_code, xc_type = APOSTBase.xc_code
    dens_grid = APOSTBase.part.dens_grid
    # ao_dens = APOSTBase.part.ao_eval_grid
    if xc_type == 'LDA':
        ao_dens = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=0)
    elif xc_type == 'GGA':
        ao_dens = eval_ao(APOSTBase._mol, APOSTBase.part.grid_coords, deriv=2)
    else:
        raise TypeError(f"Unknown XC type: {xc_type}")
    # print(np.sum(dens_grid *  APOSTBase.part._grid.quadrature_weights * APOSTBase.part.aim_w))
    # getting density and derivatives for dft functional
    # rho, dx_rho, dy_rho, dz_rho = eval_rho(APOSTBase._mol, ao_dens, dm, xctype='LDA')
    rho = eval_rho(APOSTBase._mol, ao_dens, dm, xctype=xc_type)
    print(rho.shape)



    print('CHECK ATOMIC CHARGES')
    pop_all = []
    for ia in range(len(at_nums)):
        pop = np.sum(dens_grid[atgrid_idx == ia] * qw[atgrid_idx == ia] * mol_w[atgrid_idx == ia])
        pop_all.append(pop)
        print(f'POPULATION AT {ia +1} = {pop: .5f}')
    print(f'CHECK SUM POPULATION = {np.sum(pop_all):.5f}')



   # EN
    kin_analyt =  np.trace(np.dot(dm, APOSTBase._mol.intor_symmetric('int1e_kin')))
    en_analyt = np.trace(np.dot(dm, APOSTBase._mol.intor_symmetric('int1e_nuc')))
    # 2elec coulomb and HF exchange
    two_eint = APOSTBase._mol.intor('int2e')
    coul_1 = np.einsum("ab,abcd,cd->ad", dm, two_eint, dm)
    exch_1 = np.einsum("ac,abcd,bd->ad", dm, two_eint, dm)
    coul_analyt = 0.5 * np.sum(np.einsum("ad->a", coul_1))
    exch_analyt = -0.25 * np.sum(np.einsum("ad->a", exch_1))

    iqa_r = {}

    iqa_r['t_nn'] = nn_energy(at_nums, at_coords)
    iqa_r['t_en'], iqa_r['at_en'], iqa_r['at_en_cond'] = en_energy(at_nums, at_coords, grid_coords, dens_grid, qw, mol_w, atgrid_idx)
    iqa_r['t_kin'], iqa_r['at_kin'] =  kin_energy(at_nums, at_coords, APOSTBase._mol, dm, grid_coords, qw, mol_w, atgrid_idx)
    iqa_r['t_coul'], iqa_r['t_exch'], iqa_r['at_coul'], iqa_r['at_exch'] =  ee_hf_energy(at_nums, at_coords, APOSTBase._mol,dm, grid_coords, APOSTBase.part.ao_eval_grid, occ_coeff, occ, qw, mol_w, atgrid_idx)


    # iqa_r['t_xc'] = ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, mol_w, atgrid_idx)
    # assuming dft_corr and dft_exch specified together
    if f_corr and f_exch:
        iqa_r['t_coul'], iqa_r['t_exch'], iqa_r['at_coul'], iqa_r['at_exch'] = ee_hf_energy(at_nums, at_coords, APOSTBase._mol,dm, grid_coords, APOSTBase.part.ao_eval_grid, occ_coeff, occ, qw, mol_w, atgrid_idx)
        iqa_r['t_dft_c'], iqa_r['at_dft_c'], coeff_mix = ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, mol_w, atgrid_idx)
        iqa_r['t_dft_x'], iqa_r['at_dft_x'], coeff_mix = ee_dft_energy(at_nums, xc_code, xc_type, rho, ao_dens, qw, mol_w, atgrid_idx)
        if coeff_mix:
            iqa_r['t_exch'] = iqa_r['t_exch'] * coeff_mix
            iqa_r['at_coul'] = iqa_r['at_coul'] * coeff_mix
            # iqa_r['t_dft_x'] = iqa_r['t_dft_x'] * (1-coeff_mix)
            # iqa_r['at_dft_x'] = iqa_r['at_dft_x'] * (1-coeff_mix)
        else:
            iqa_r.pop('t_exch')
            iqa_r.pop('at_exch')

    elif f_exch:
        iqa_r['t_coul'], iqa_r['t_exch'], iqa_r['at_coul'], iqa_r['at_exch'] = ee_hf_energy(at_nums, at_coords, APOSTBase._mol,dm, grid_coords, APOSTBase.part.ao_eval_grid, occ_coeff, occ, qw, mol_w, atgrid_idx)
        iqa_r['t_dft_xc'], iqa_r['at_dft_xc'], coeff_mix = ee_dft_energy(at_nums, xc_code, xc_type,rho, ao_dens, qw, mol_w, atgrid_idx)
        if coeff_mix:
            # Scaling HF exchange
            iqa_r['t_exch'] = iqa_r['t_exch'] * coeff_mix
            iqa_r['at_coul'] = iqa_r['at_coul'] * coeff_mix
            # iqa_r['t_dft_xc'] = iqa_r['t_dft_xc'] * (1 - coeff_mix)
            # iqa_r['at_dft_xc'] = iqa_r['at_dft_xc'] * (1 - coeff_mix)
        else:
            iqa_r.pop('t_exch')
            iqa_r.pop('at_exch')
    elif not APOSTBase.dft:
        iqa_r['t_coul'], iqa_r['t_exch'], iqa_r['at_coul'], iqa_r['at_exch'] = ee_hf_energy(at_nums,
                                                                                            at_coords,
                                                                                            APOSTBase._mol,
                                                                                            dm,
                                                                                            grid_coords,
                                                                                            APOSTBase.part.ao_eval_grid,
                                                                                            occ_coeff,
                                                                                            occ, qw,
                                                                                            mol_w,
                                                                                            atgrid_idx)

    else:
        raise ValueError(f'Need to specify an exchange functional too. Got {f_exch}')


    logging.info('Summary IQA')
    print(f"Nucleus-Nucleus repulsion energy: {iqa_r['t_nn']:.5f}")
    # print('Nucleus-Nucleus repulsion energy (PySCF/Libcint): ', nn_int)
    # diff = abs(iqa_r['nn_total'] - nn_int)
    # print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)

    print('------------------------------------')
    print(f"Electron-Nucleus attraction energy: {iqa_r['t_en']:.5f}" )
    print(f'Electron-Nucleus attraction energy (PySCF/Libcint): {en_analyt: .5f}')
    diff = abs(iqa_r['t_en'] - en_analyt)
    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
    print()
    # assert 5 ==6
    print("Atomic Electron-Nucleus attraction energy (Condensed):")
    for idx, at in enumerate(at_nums):
        print(f"{at}   {iqa_r['at_en_cond'][idx]: .5f}")
    print('------------------------------------')
    print(f"Kinetic energy:  {iqa_r['t_kin']:.5f}" )
    print(f'Kinetic energy(PySCF/Libcint):  {kin_analyt: .5f}')
    diff = abs(iqa_r['t_kin'] - kin_analyt)
    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
    print()
    print('Atomic Kinetic energy:')
    for idx, at in enumerate(at_nums):
        print(f"{at}   {iqa_r['at_kin'][idx]: .5f}")
    print('------------------------------------')
    print('Electron-Electron interaction energy')
    print()
    print(f"Coulomb energy: {iqa_r['t_coul']: .5f}", )
    print(f'Coulomb energy(PySCF/Libcint):  {coul_analyt: .5f}')
    diff = abs(iqa_r['t_coul'] - coul_analyt)
    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
    print()
    print('Atomic Coulomb energy:')
    for idx, at in enumerate(at_nums):
        print(f"{at}   {iqa_r['at_coul'][idx]: .5f}")
    print()
    if f_exch and f_corr:
        print(f"DTF Exchange energy: {iqa_r['t_dft_x']: .5f}")
        print(f"DTF Correlation energy: {iqa_r['t_dft_c']: .5f}")
        print()
        print('Atomic DFT Exchange energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_r['at_dft_x'][idx]: .5f}")
        print('Atomic DFT Correlation energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_r['at_dft_c'][idx]: .5f}")

    if f_exch and not f_corr:
        print(f"DTF Exchange-Correlation energy: {iqa_r['t_dft_xc']: .5f}" )
        print()
        print('Atomic DFT Exchange-Correlation energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_r['at_dft_xc'][idx]: .5f}")

    if 'x_hf_total' in iqa_r.keys():
        print(f"HF Exchange: {iqa_r['t_exch']: .5f}" )
        print(f'HF Exchange energy(PySCF/Libcint):  {exch_analyt: .5f}')
        diff = abs(iqa_r['t_exch'] - exch_analyt)
        print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print('COEFF MIX :', coeff_mix)
        print()
        print('Atomic HF Exchange energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_r['at_exch'][idx]: .5f}")

    total_sum_energy = 0
    for k in iqa_r.keys():
        if k.startswith('t_'):
            total_sum_energy += iqa_r[k]

    print('TOTAL ENERGY (SUM OF COMPONENTS):', total_sum_energy)
    print('TOTAL ENERGY (PySCF):', APOSTBase.energy)
    print('DIFF (AU) :', total_sum_energy - APOSTBase.energy)
    print('DIFF (Kcal) :', (total_sum_energy - APOSTBase.energy) / 0.0015936014376406278)
    # print('CHARGE PART OBJECT', self.part.charges)
    assert 5 == 7
    return iqa_r
