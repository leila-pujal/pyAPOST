import numpy as np
import pytest as pytest
import copy as copy
from numpy.ma.testutils import assert_equal, assert_almost_equal, assert_allclose

from pyscf import gto,scf,dft,lib, mcscf

from pyAPOST.base import APOSTBase, Rpart, Hpart

def o2_pyscf(method, spin,basis, cas_method=None):
    """Class to compute o2 molecule with pyscf specifying method and spin to pass to CAS method"""
    with lib.with_omp_threads(1):
        mol = gto.M(atom=[
            [8, (0.5845381, 1.3618218, 0.0)],
            [8, (1.0054419, 2.3424182, 0.0)]], basis=basis)
        mol.cart = False
        mol.symmetry = False
        mol.ms = spin
        mol.build()
        if method == 'RHF':
            calc = scf.RHF(mol)
        elif method == 'UHF':
            calc = scf.UHF(mol)
        elif len(method) == 2 and method[0] == 'RKS':
            calc = dft.RKS(mol)
            calc.xc = method[1].upper()
        elif len(method) == 2 and method[0] == 'UKS':
            calc = dft.UKS(mol)
            calc.xc = method[1].upper()
        calc.kernel()

        # check that we pass the correct cas_method
        if cas_method and len(cas_method) == 3:
            if cas_method[0] == 'CASSCF':
                cas_calc = mcscf.CASSCF(calc,cas_method[1],cas_method[2]) # orbitals, electrons
            elif cas_method[0] == 'CASCI':
                cas_calc = mcscf.CASCI(calc,cas_method[1],cas_method[2]) # orbitals, electrons
            cas_calc.kernel()

            return mol, cas_calc
        else:
            return mol, calc

def h2o_pyscf(method, spin,basis, cas_method=None):
    """Class to compute h2o molecule with pyscf specifying method and spin to pass to CAS method"""
    with lib.with_omp_threads(1):
        mol = gto.M(atom=[
                    [8, (-2.3534315116, 1.7976043966, 0.0000000000)],
                    [1, (-1.3674018242, 1.8792998063, 0.000000000)],
                    [1, (-2.6055654975, 2.7543473799, 0.0000000000)]], basis=basis)
        mol.cart = False
        mol.symmetry = False
        mol.ms = spin
        mol.build()
        if method == 'RHF':
            calc = scf.RHF(mol)
        elif method == 'UHF':
            calc = scf.UHF(mol)
        elif len(method) == 2 and method[0] == 'RKS':
            calc = dft.RKS(mol)
            calc.xc = method[1].upper()
        elif len(method) == 2 and method[0] == 'UKS':
            calc = dft.UKS(mol)
            calc.xc = method[1].upper()
        calc.kernel()

        # check that we pass the correct cas_method
        if cas_method and len(cas_method) == 3:
            if cas_method[0] == 'CASSCF':
                cas_calc = mcscf.CASSCF(calc,cas_method[1],cas_method[2]) # orbitals, electrons
            elif cas_method[0] == 'CASCI':
                cas_calc = mcscf.CASCI(calc,cas_method[1],cas_method[2]) # orbitals, electrons
            cas_calc.kernel()

            return mol, cas_calc
        else:
            return mol, calc


def test_APOSTBase_raise_exceptions():
    atom_grid_specs = {}
    atom_grid_specs['atom_grid'] = {"O": (40, 146)}
    mol, calc = o2_pyscf('RHF', 0, basis='cc-pvtz')
    with pytest.raises(TypeError):
        APOSTBase(1, calc, part_name="tfvc", rpart_specs=atom_grid_specs)
    with pytest.raises(TypeError):
        APOSTBase(mol, 1, part_name="tfvc", rpart_specs=atom_grid_specs)
    with pytest.raises(ValueError):
        APOSTBase(mol, calc, part_name="Hirshfeld", rpart_specs=atom_grid_specs)
    with pytest.raises(TypeError):
        APOSTBase(mol, calc, part_name="tfvc", rpart_specs=[])
    mol, calc_sd = o2_pyscf('RHF', 0, basis='cc-pvtz')
    with pytest.raises(AttributeError):
        calc_sd.dm2

def test_dm1_dm2():
    atom_grid_specs = {}
    atom_grid_specs['atom_grid'] = {"O": (40, 146)}
    mol, calc = o2_pyscf('RHF', 1, basis='cc-pvtz', cas_method=['CASSCF', 6, 8])
    base_apost = APOSTBase(mol, calc, part_name="tfvc", rpart_specs=atom_grid_specs)
    t_core=0
    t_active=0
    t_full=0

    for i in range(4):
        for j in range(4):
            # for j in range(norb):
            t_core += base_apost.dm2[i,i,j,j]
    for i in range(4, 10):
        for j in range(4, 10):
            t_active += base_apost.dm2[i, i, j, j]
    for i in range(10):
        for j in range(10):
            t_full += base_apost.dm2[i, i, j, j]

    assert_almost_equal(t_core, 56.0)
    assert_almost_equal(t_active, 56.0)
    assert_almost_equal(t_full, 240.0)
    apost_occ_no = np.array([2.00000,   2.00000,   2.00000,   2.00000,   1.97843,   1.97780,   1.97780,   1.02182,
                            1.02182,   0.02233])
    assert_allclose(base_apost._occ_no[base_apost._occ_no > 1e-8], apost_occ_no, rtol=1.e-3, atol=1e-8)


def test_Rpart_raise_exceptions():
    atom_grid_specs = {}
    atom_grid_specs['atom_grid'] = {"O": (40, 146)}
    mol, calc = h2o_pyscf('RHF', 0, basis='def2-svp')
    base_apost = APOSTBase(mol, calc, part_name="tfvc", rpart_specs=atom_grid_specs)

    with pytest.raises(NotImplementedError):
        Rpart('Hirshfeld', mol, base_apost.dm_a, base_apost.at_nums, frags=[[1],[2],[3]], grid_specs=atom_grid_specs)
    with pytest.raises(TypeError):
        Rpart('tfvc', mol, np.array([0,1]), base_apost.at_nums, frags=[[1], [2], [3]], grid_specs=atom_grid_specs)
    with pytest.raises(TypeError):
        Rpart('tfvc', mol, base_apost.dm_a, base_apost.at_nums, frags=123, grid_specs=atom_grid_specs)
    with pytest.raises(ValueError):
        base_apost.part.rotate_grid(base_apost.at_coords, base_apost.at_nums, 0, 0.162, base_apost.dm_a + base_apost.dm_b)



# test grid: with different specifications check the expected number of points
# For now only check non pruned grids
def test_grid_h2o():
    atom_grid_specs = {}
    atom_grid_specs['atom_grid'] = {"H": (40, 146),"O": (40, 146) }
    mol, calc = h2o_pyscf('RHF', 0, basis='def2-svp')
    base_apost = APOSTBase(mol, calc, part_name="tfvc", rpart_specs=atom_grid_specs)
    assert base_apost.part.grid_coords.shape == (17520, 3)
    # check quadrature and atomic weights by integrating total density and atomic populations
    n_elect = np.sum(base_apost.part.dens_grid * base_apost.part.quadrature_weights * base_apost.part.aim_w)
    assert_allclose(n_elect, 10.0,  rtol=1.e-4, atol=1e-8)
    # Compare against APOST3D version 4 populations
    populations_ref = np.array([9.199806, 0.400097, 0.400097])
    populations_all = []
    for ifrag, frag in enumerate(base_apost.part.frags):
        tmp_w = np.zeros((base_apost.part.grid_weights.shape[0]))
        for ia in frag:
            # Adding weights for all atoms in the fragment
            ia = ia - 1  # Reverting to python index
            tmp_w[base_apost.part.grid_to_atom_idx == ia] += base_apost.part.aim_w[base_apost.part.grid_to_atom_idx == ia]
        pop = np.sum(base_apost.part.dens_grid * base_apost.part._grid.quadrature_weights * tmp_w)
        populations_all.append(pop)
    populations_all = np.array(populations_all)
    assert_allclose(populations_all, populations_ref, rtol=1.e-3, atol=1e-8)
    base_apost_f = APOSTBase(mol, calc, part_name="tfvc", frags=[[1,2], [3]], rpart_specs=atom_grid_specs)
    populations_ref = np.array([9.599903, 0.400097])
    populations_all = []
    for ifrag, frag in enumerate(base_apost_f.part.frags):
        tmp_w = np.zeros((base_apost_f.part.grid_weights.shape[0]))
        for ia in frag:
            # Adding weights for all atoms in the fragment
            ia = ia - 1  # Reverting to python index
            tmp_w[base_apost_f.part.grid_to_atom_idx == ia] += base_apost_f.part.aim_w[base_apost_f.part.grid_to_atom_idx == ia]
        pop = np.sum(base_apost_f.part.dens_grid * base_apost_f.part._grid.quadrature_weights * tmp_w)
        populations_all.append(pop)
    populations_all = np.array(populations_all)
    assert_allclose(populations_all, populations_ref, rtol=1.e-3, atol=1e-8)
    # Check rotation and that we get back correct aim for rotated grid
    rot_mat = np.array([
        [np.cos(0.162), np.sin(0.162), 0],
        [-np.sin(0.162), np.cos(0.162), 0],
        [0, 0, 1]
    ])
    print(APOSTBase.at_nums)
    grid_coords2_pre = [copy.copy(base_apost.part.grid_coords)[base_apost.part.grid_to_atom_idx == ia] - base_apost.at_coords[ia] for ia in range(len(base_apost.at_nums))]
    for ia in range(len(grid_coords2_pre)):
        grid_coords2_pre[ia] = np.dot(grid_coords2_pre[ia], rot_mat)
        grid_coords2_pre[ia] = grid_coords2_pre[ia] + base_apost.at_coords[ia]

    grid_coords2 = np.vstack(grid_coords2_pre)

    grid_coords2_ref, mol_w_2_ref = base_apost.part.rotate_grid(base_apost.at_coords, base_apost.at_nums, 0, 0.162, base_apost.dm_a + base_apost.dm_b, mol=base_apost._mol)

    assert_allclose(grid_coords2, grid_coords2_ref, rtol=1.e-3, atol=1e-8)

