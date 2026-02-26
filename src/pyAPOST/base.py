import numpy as np

import scipy as scipy

# Imports Hilbert
from pyscf import mcscf
from pyscf import lo

# Imports Real
from pyscf.dft.numint import eval_ao
from pyscf.dft.gen_grid import Grids, get_partition, gen_atomic_grids, radi


from pyAPOST.part_schemes import becke_w, tfvc_apost_w


import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

# INFO: PySCF works with NPArrayWithTag from its own library

# TODO: Use attr library to lock the value of attributes once they are initialized after a new object.
# TODO: Decide how to pass the log level. Right now it's changed on EOS function but maybe it should be pased here
# the worry is that if left unchecked, because we are parsing from pyscf Mole/Calc if they change for whatever
# reason that could be traced to the related variables inside pyAPOST. A brute force solution would be to use
# copy library but it would be annoying.

class APOSTBase(object):
    """Base class

    Arguments:
    mol: object of class Mole from PySCF
    calc: object of method calculation from PySCF
    """



    def __init__(self, mol, calc, part_name=None, frags=None):

        #DEV:saving mol
        APOSTBase._mol = mol

        logger.info("  INITIALIZING APOSTBase object for Hilbert and Real space analysis")
        logger.info("\n\n")

        # Molecule basic information from mol object
        self._natm = mol.natm
        self._charge = mol.charge
        self._spin = mol.spin
        self._atnums = [mol.atom_nelec_core(ia) + mol.atom_charge(ia) for ia in range(mol.natm)]
        self._alpha_elec = mol.nelec[0]
        self._beta_elec = mol.nelec[1]
        #
        # Basis set information
        self._nbasis = mol.nbas # Number of shells
        # self._basis_exp = None
        # self._basis_coeff = None
        # Q: shell to map: either a function and we store mol or we generate it here
        # TODO: complete _shell_to_atom and _shell_to_basis var if needed
        # for i_e, e in enumerate(mol._basis.keys()):
        #     basis = mol._basis[e]
        #     print(basis)
        #     print(len(basis))
        self._shell_to_atom = None
        self._shell_to_basis= None
        self._ovlp = mol.intor_symmetric('int1e_ovlp')



        # Calculation information
        # Q: we fill some info based on the type of calc (rest/unres, SD/Multiref)
        # Check if calc is SD
        self._kind_calc = str(type(calc))

        # initializing flags
        self._unrestr = False
        self._sd = False
        self._dft = False
        self._casscf = False
        self._ccsd = False


        # Sort calc type and set mo_coeffs, mo_occ and dm
        if any([i in self._kind_calc for i in ["RHF", "RKS", "UHF", "UKS", "KS", "dftd3"]]):
            self._sd = True
            self._mo_coeff_a = calc.mo_coeff
            self._mo_coeff_b = calc.mo_coeff
            self._occ_a = calc.mo_occ
            self._occ_b = calc.mo_occ
            self._dm_a = calc.make_rdm1()
            self._dm_b = calc.make_rdm1()
            if self._kind_calc in ["RKS", "UKS", "KS", "dftd3"]:
                self._dft = True
            # Check unrestricted
            if any([i in self._kind_calc for i in ["UHF", "UKS"]]):
                self._unrestr = True
                # Q: APOSTpy file solves for the generalized eigenvalue for this type too. Why?
                # Assuming first item is for alpha and second for beta
                self._mo_coeff_a = calc.mo_coeff[0]
                self._mo_coeff_b = calc.mo_coeff[1]
                self._occ_a = calc.mo_occ[0]
                self._occ_b = calc.mo_occ[1]
                self._dm_a, self._dm_b = calc.make_rdm1()
        elif "CASSCF" in self._kind_calc:
            self._casscf = True
            # Because PySCF does not store CASSCF mo_coeff coefficients for all orbitals together
            # we obtain it solving the general eigenvalue equation  # Q
            # Q: is CASSCF always unrestricted?
            # DEV TEST ###
            # casdm1a, casdm1b = calc.fcisolver.make_rdm1s(calc.ci, calc.ncas, calc.nelecas)
            # print('a', casdm1a)
            # print('b', casdm1b)
            ##########
            self._dm_a, self._dm_b = mcscf.addons.make_rdm1s(calc)
            # From debug: scipy.linalg.eigh((np.linalg.multi_dot((self._ovlp, self._dm_b, self._ovlp))), b=self._ovlp) gives changed signed values compared to scipy.linalg.eigh(SCR_a, b=self._ovlp)
            SCR_a = np.linalg.multi_dot((self._ovlp, self._dm_a, self._ovlp))
            SCR_b = np.linalg.multi_dot((self._ovlp, self._dm_b, self._ovlp))
            occ_a, coeff_a = scipy.linalg.eigh(SCR_a, b=self._ovlp)
            occ_b, coeff_b = scipy.linalg.eigh(SCR_b, b=self._ovlp)
            # scipy.linalg.eigh returns smallest to highest order --> flip
            # coeff_a == C_AO,MO --> flip axis=1
            # Check with original code APOSTpy: Yes
            occ_a, coeff_a = np.flip(occ_a), np.flip(coeff_a, axis=1)
            occ_b, coeff_b = np.flip(occ_b), np.flip(coeff_b, axis=1)

            self._occ_a = occ_a
            self._occ_b = occ_b
            self._mo_coeff_a = coeff_a
            self._mo_coeff_b= coeff_b


        # PySCF treats CCSD and UCCSD in different modules. For CCSD will return 1 DM in MO basis
        # and UCCSD will return DMa/DMb in MO basis.
        elif "CCSD" in self._kind_calc:
            self._ccsd = True
            if "UCCSD" in self._kind_calc:
                self._unrestr = True
                dma, dmb = calc.make_rdm1()
                occ_a, coeff_a = scipy.linalg.eigh(dma)
                occ_b, coeff_b = scipy.linalg.eigh(dmb)
                occ_a, coeff_a = np.flip(occ_a), np.flip(coeff_a, axis=1)
                occ_b, coeff_b = np.flip(occ_b), np.flip(coeff_b, axis=1)
                # Transform to AO basis. Note: PySCF returns the dm in AO basis if calc.make_rdm1(ao_repr=True), default is False
                coeff_a = np.einsum('aj, ji,->ai', calc.mo_coeff[0], coeff_a)
                coeff_b = np.einsum('aj, ji,->ai', calc.mo_coeff[1], coeff_b)
                self._mo_coeff_a = coeff_a
                self._mo_coeff_b = coeff_b
                self._occ_a = occ_a
                self._occ_b = occ_b

            else:
                dm = calc.make_rdm1()
                occ, coeff = scipy.linalg.eigh(dm)
                # Because doubly occupied, occupations dived by 2
                occ, coeff = np.flip(occ / 2.0), np.flip(coeff, axis=1)
                # Transform to AO basis. Note: PySCF returns the dm in AO basis if calc.make_rdm1(ao_repr=True), default is False
                coeff = np.einsum('aj, ji,->ai', calc.mo_coeff,  coeff)
                self._mo_coeff_a = coeff
                self._mo_coeff_b = coeff
                self._occ_a = occ
                self._occ_b = occ

        # Print Calc info
        logger.info(f'  Calc INFORMATION')
        if self._sd:
            logger.info(f'  Single Determinant : {self._sd}')
        else:
            if self._ccsd:
                logger.info(f'  Post-Hartree Fock CCSD' )
            elif self._casscf:
                logger.info(f'  Post-Hartree Fock CASSCF')
        logger.info(f'  Unrestricted : {self._unrestr}')
        logger.info(f'  DFT : {self._dft}\n')


        # Initialize Hpart or Rpart
        if part_name in ["mulliken", "lowdin", "meta_lowdin", "iao", "nao"]:
            self._part = Hpart(part_name, mol, frags=frags)
        elif part_name in ["becke", 'tfvc']:
            dm = self._dm_a + self._dm_b
            self._part = Rpart(part_name, mol, dm, self._atnums, frags=frags)
        else:
            raise ValueError("Could not identify partition. Choose from: Mulliken,Lowdin,Meta-Lowdin,IAO,NAO")

    @property
    def natm(self):
        return self._natm
    @property
    def atnums(self):
        return self._atnums
    @property
    def ovlp(self):
        return self._ovlp
    @property
    def unrestr(self):
        return self._unrestr
    @property
    def sd(self):
        return self._sd
    @property
    def dft(self):
        return self._dft
    @property
    def part(self):
        return self._part

    @property
    def alpha_elec(self):
        return self._alpha_elec

    @property
    def beta_elec(self):
        return self._beta_elec

    # Not a or b properties obtained by combining a/b ones
    @property
    def mo_coeff_a(self):
        return self._mo_coeff_a

    @property
    def mo_coeff_b(self):
        return self._mo_coeff_b

    @property
    def occ_a(self):
        return self._occ_a

    @property
    def occ_b(self):
        return self._occ_b

    @property
    def dm_a(self):
        return self._dm_a

    @property
    def dm_b(self):
        return self._dm_b

    @property
    def mo_coeff(self):
        raise NotImplementedError

    @property
    def occ(self):
        raise NotImplementedError

    def generate_mo_coeff(self, dm, overlap=None):
        """Method to obtain molecular orbital coefficients from Dm. Optional overlap"""
        # Todo: investigate about how we should do it if we wanted to update the internal attribute of the APOSTBase object
        # 2 options: 1) Don't update and just return the occ/coeff. 2) Allow for an option to update the attribute. For this we need to know the good practices
        # about updating attributes.

        # Right now code specific for dm:One-electron density matrix and overlap

        # check dm
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2 and dm.dtype == float):
            raise TypeError("One-electron density matrix must be a 2D-array float `dtype`")
        if dm.shape[0] != dm.shape[1]:
            raise ValueError("One-electron density matrix must be a square matrix.")
        if not np.allclose(dm, dm.T):
            raise ValueError("One-electron density matrix must be symmetric.")

        if isinstance(overlap, np.ndarray):
            # Todo: check for overlap too
            SCR = np.linalg.multi_dot((overlap, dm, overlap))
            occ_n, coeff_n = scipy.linalg.eigh(SCR, b=overlap)
            # scipy.linalg.eigh returns smallest to highest order --> flip
            # coeff_a == C_AO,MO --> flip axis=1
            # Check with original code APOSTpy: Yes
            occ_n, coeff_n = np.flip(occ_n), np.flip(coeff_n, axis=1)
            return occ_n, coeff_n

        # Only Dm.
        else:
            occ_n, coeff_n = scipy.linalg.eigh(dm)
            # scipy.linalg.eigh returns smallest to highest order --> flip
            # coeff_a == C_AO,MO --> flip axis=1
            # Check with original code APOSTpy: Yes
            occ_n, coeff_n = np.flip(occ_n), np.flip(coeff_n, axis=1)
            return occ_n, coeff_n

# todo: change zfrags and frags from duplicated code in Hpart/Rpart to APOSTBase class

class Rpart(object):
    """Real space part object"""
    # atomic weights
    # integration grids
    # Need to implement the partitions outside
    # Optional: allow to pass coords and weights bypassing relying on the PySCF Grids
    # todo: Allow to specify radial and angular points. in that case a dic is created to pass to atom_grid attribute before building the grid
    # todo: allow to specify radii type, radii method and empirical values (Bragg/Trehluer), and prune
    def __init__(self, part_name, mol, dm, atnums, frags=None, sort_grids=False, atom_grid_specs=None):

        self._part_name = part_name
        logger.info(f'  Initializing Rpart : {part_name.upper()} \n')


        # If frags are None then it computes it for atoms
        # First check frags argument. Needs to be a list, set or numpy array.
        # Optional: we could check for more restrictions depending on the code.
        if frags is not None:
            if not (isinstance(frags, np.ndarray) or isinstance(frags, list) or isinstance(frags, set)):
                raise TypeError(
                    "Fragment indices must be specified as a list, numpy array or set types")
            else:
                self._frags = frags
                zatoms = mol.atom_charges()
                self._zfrags = [sum(zatoms[i - 1] for i in group) for group in self._frags]
        else:
            # Force to have atoms as frags
            self._frags = np.array([[i + 1] for i in range(mol.natm)])
            zatoms = mol.atom_charges()
            self._zfrags = [sum(zatoms[i - 1] for i in group) for group in self._frags]


        # Define molecular grid and access points and weights
        # Create class Grids and set up parameters for radial and angular grids
        self._grid = Grids(mol)
        self._grid.prune = None
        self._grid.radi_method = radi.becke
        # grid.becke_scheme = gen_grid.original_becke
        # grid.radii_adjust = radi.becke_atomic_radii_adjust
        # grid.atomic_radii = radi.BRAGG_RADII
        # From pyscf/dft/gen_grid.py # Padding grids to make the AO value generated by eval_gto aligned in memory
        self._grid.alignment = 0
        if atom_grid_specs is not None and isinstance(atom_grid_specs, dict):
            # Note: if dict doesn't follow pyscf guidelines to define rad/ang it ignores it
            self._grid.atom_grid = atom_grid_specs

        # with_non0tab: stablishes a cutoff
        # from pyscf/dft/numint.py
        # screen_index: 2D uint8 array
        # How likely the AO values on grids are negligible. This array can be obtained by calling :func:`gen_grid.make_screen_index
        # sort_grids: it gets indices from arg_group_grids() it keeps the number of points the same (so far with the tests I checked)
        # but it reorganizes de points. Documentation from that function: Partition the entire space into small boxes according to the input box_size.
        #     Group the grids against these boxes.
        # self._grid.build(with_non0tab=False, sort_grids=False)
        self._grid.build(with_non0tab=False, sort_grids=sort_grids)
        self._coords = self._grid.coords
        # Weights = radial(quadrature) + angular + becke
        self._weights = self._grid.weights
        self._grid_to_atom_idx = self._grid.atm_idx
        if part_name == 'becke':
            self._aim_w = becke_w(self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx)
        elif part_name == 'tfvc':
            # self._aim_w = tfvc_w(mol, dm_a, self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx)
            self._aim_w = tfvc_apost_w(mol, dm, self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx)
        else:
            raise NotImplementedError(f"Unknown part name: {part_name}")
        # Store computed atomic orbitals in a grid
        self._ao_eval_grid = eval_ao(mol, self._coords)


    @property
    def part_name(self):
        return self._part_name

    @property
    def grid_coords(self):
        return self._coords

    @property
    def grid_weights(self):
        return self._weights

    @property
    def ao_eval_grid(self):
        return self._ao_eval_grid

    @property
    def grid_to_atom_idx(self):
        return self._grid_to_atom_idx

    @property
    def aim_w(self):
        return self._aim_w

    @property
    def frags(self):
        return self._frags

    @property
    def zfrags(self):
        return self._zfrags

class Hpart(object):
    # Eta, transformation matrix
    # Init could be modified by taking out passing mol and instead all the information that we extract from it
    # this could allow more flexibility
    def __init__(self, part_name, mol, frags=None):
        # DEV: Generate Eta matrices
        # First parse necessary data from Mole object
        self._part_name = part_name
        self.num_ao = mol.nao
        self.id_slice_atom = mol.aoslice_by_atom()

        logger.info(f'  Initializing Hpart : {part_name.upper()}')



        # If frags are None then it computes it for atoms
        # First check frags argument. Needs to be a list, set or numpy array.
        # Optional: we could check for more restrictions depending on the code.
        if frags is not None:
            if not (isinstance(frags, np.ndarray) or isinstance(frags, list) or  isinstance(frags, set)):
                raise TypeError("Fragment indices must be specified as a list, numpy array or set types")
            else:
                self._frags = frags
                zatoms = mol.atom_charges()
                self._zfrags = [sum(zatoms[i - 1] for i in group) for group in frags]
        else:
            # Force to have atoms as frags
            self._frags = np.array([[i+1] for i in range(mol.natm)])

        # Assuming always frags stored, either information passed or atoms
        eta_frags = np.zeros((len(self._frags), self.num_ao, self.num_ao))
        for ifrag, frag in enumerate(self._frags):
            for atom in frag:
                start = mol.aoslice_by_atom()[atom - 1, -2]
                end = mol.aoslice_by_atom()[atom - 1, -1]
                eta_frags[ifrag, start:end, start:end] += np.eye(end -  start)

        self._eta_frags = eta_frags

        self._T = None
        self._T_inv = None
        # Generate transformation matrix if needed by the partition
        if part_name == "lowdin":
            self._T_inv = lo.orth_ao(mol, part_name, pre_orth_ao=None)
            self._T = np.linalg.inv(self._T_inv)
        elif part_name == "meta_lowdin":
            self._T_inv = lo.orth_ao(mol, part_name, pre_orth_ao="ANO")
            self._T = np.linalg.inv(self._T_inv)
        elif part == "nao":
            self._T_inv = lo.nao.nao(mol, part_name)
            self._T = np.linalg.inv(self._T_inv)



    @property
    def part_name(self):
        return self._part_name

    @property
    def frags(self):
        return self._frags

    @property
    def zfrags(self):
        return self._zfrags

    @property
    def eta_frags(self):
        return self._eta_frags

    @property
    def T(self):
        return self._T

    @property
    def T_inv(self):
        return np.linalg.inv(self._T)





