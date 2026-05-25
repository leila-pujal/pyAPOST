import numpy as np

import scipy as scipy
# from scipy.spatial.transform import Rotation as R
import copy as copy
# Imports Hilbert
from pyscf import mcscf
from pyscf import lo

# Imports Real
from pyscf.dft.numint import eval_ao
from pyscf.dft.gen_grid import Grids, gen_atomic_grids, radi
from pyscf.dft import gen_grid


from pyAPOST.part_schemes import becke_w, tfvc_w


import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

# INFO: PySCF works with NPArrayWithTag from its own library

# TODO: Use attr library to lock the value of attributes once they are initialized after a new object.
# TODO: Decide how to pass the log level. Right now it's changed on EOS function but maybe it should be pased here
# the potential problem is that if left unchecked, because we are parsing from pyscf Mole/Calc if they change for whatever
# reason that could be traced to the related variables inside pyAPOST. A brute force solution would be to use
# copy library but it would be annoying.

class APOSTBase(object):
    """Base class

    Arguments:
    mol: object of class Mole from PySCF
    calc: object of method calculation from PySCF
    """

    def __init__(self, mol, calc, part_name=None, frags=None, rpart_specs=None):

        #DEV:saving mol
        APOSTBase._mol = mol

        logger.info("  INITIALIZING APOSTBase object for Hilbert and Real space analysis")
        logger.info("\n\n")

        # Todo: decide if to keep it stored
        self._mol = mol
        # Molecule basic information from mol object
        self._natm = mol.natm
        self._at_coords = mol.atom_coords()
        self._charge = mol.charge
        self._spin = mol.spin
        self._at_nums = [mol.atom_nelec_core(ia) + mol.atom_charge(ia) for ia in range(mol.natm)]
        self._alpha_elec = mol.nelec[0]
        self._beta_elec = mol.nelec[1]
        self._e_tot = calc.e_tot
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
        self._cas = False
        self._ccsd = False
        self._xc_code = (None, None)


        # Sort calc type and set mo_coeffs, mo_occ and dm
        # todo: Put test to be sure we fill the variables correctly
        if any([i in self._kind_calc for i in ["RHF", "RKS", "UHF", "UKS", "KS", "dftd3"]]):
            self._sd = True
            if any([i in self._kind_calc for i in ["RKS", "UKS", "KS", "dftd3"]]):
                self._dft = True
                xc_type = calc._numint._xc_type(calc.xc)
                self._xc_code = (calc.xc, xc_type)
            # Check unrestricted
            if any([i in self._kind_calc for i in ["UHF", "UKS", "ROHF"]]):
                self._unrestr = True
                # Q: APOSTpy file solves for the generalized eigenvalue for this type too.
                # Assuming first item is for alpha and second for beta
                self._mo_coeff_a = calc.mo_coeff[0]
                self._mo_coeff_b = calc.mo_coeff[1]
                self._occ_a = calc.mo_occ[0]
                self._occ_b = calc.mo_occ[1]
                self._dm_a, self._dm_b = calc.make_rdm1()
                logger.info('OCCUPATIONS (UNRESTRICTED)')
                print('ALPHA', self._occ_a)
                print('TOTAL ALPHA', np.sum(self._occ_a))
                print('BETA', self._occ_b)
                print('TOTAL BETA', np.sum(self._occ_b))
                self._s = calc.get_ovlp()
                # print(np.trace(np.dot(self._dm_a, self._s)))
            else:
                self._mo_coeff_a = calc.mo_coeff
                self._mo_coeff_b = calc.mo_coeff
                self._occ_a = calc.mo_occ
                self._occ_b = calc.mo_occ
                self._dm_a = calc.make_rdm1()
                self._dm_b = calc.make_rdm1()
                logger.info('OCCUPATIONS (RESTRICTED)')
                print('ALPHA/BETA', self._occ_a)
                print('TOTAL', np.sum(self._occ_a))

        elif ("CASSCF" in self._kind_calc) or ("CASCI" in self._kind_calc):
            self._cas = True
            # We need this to compute one electron terms correctly
            if "ROHF" in str(type(calc._scf)):
                self._unrestr = True
                self._dm_a, self._dm_b = mcscf.addons.make_rdm1s(calc)
            else:
                self._dm_a = mcscf.addons.make_rdm1(calc)
                self._dm_b = self._dm_a

            ##########
            # Obtain natural orbitals coefficients in the AO by diagonalizing generalized equation (SPS)C=SCE
            # From debug: scipy.linalg.eigh((np.linalg.multi_dot((self._ovlp, self._dm_b, self._ovlp))), b=self._ovlp) gives changed signed values compared to scipy.linalg.eigh(SCR_a, b=self._ovlp)
            # Obtain Natural orbitals from P in AO
            SCR_a = np.linalg.multi_dot((self._ovlp, self._dm_a, self._ovlp))
            SCR_b = np.linalg.multi_dot((self._ovlp, self._dm_b, self._ovlp))
            # SCR = np.linalg.multi_dot((self._ovlp, self._dm_b + self._dm_a, self._ovlp))
            occ_a, coeff_a = scipy.linalg.eigh(SCR_a, b=self._ovlp)
            occ_b, coeff_b = scipy.linalg.eigh(SCR_b, b=self._ovlp)
            # occ, coeff = scipy.linalg.eigh(SCR, b=self._ovlp)
            # scipy.linalg.eigh returns smallest to highest order --> flip
            # coeff_a == C_AO,MO --> flip axis=1
            # Check with original code APOSTpy: Yes
            occ_a, coeff_a = np.flip(occ_a), np.flip(coeff_a, axis=1)
            occ_b, coeff_b = np.flip(occ_b), np.flip(coeff_b, axis=1)
            # occ, coeff = np.flip(occ), np.flip(coeff, axis=1)

            self._occ_a = occ_a
            self._occ_b = occ_b
            self._mo_coeff_a = coeff_a
            self._mo_coeff_b= coeff_b

            # DEV:  Dic with only CAS related data.
            # For IQA-PHF we only need norb after saving the DMs.
            # Keeping it for now (as protected attribute of the class) to see if we needed for the spin
            self._cas_data = {}
            self._cas_data['nelecas'] =  calc.nelecas  # total cas electrons tuple(alpha,beta) now only taking alpha because restricted
            self._cas_data['ncore'] = calc.ncore # number of inactive
            self._cas_data['ncas'] = calc.ncas # total cas orbitals
            self._cas_data['ci'] = calc.ci # CAS space FCI coefficients (From pyscf doc CASBase class)

            # calc.ci: CAS space FCI coefficients (From pyscf doc CASBase class)
            # casdm1 = tuple with alpha and beta active space orbitals dm1
            casdm1s = calc.fcisolver.make_rdm1s(calc.ci, calc.ncas, calc.nelecas)

            # Reconstructing dm1 as in APOST. Could be potentially refactored in the future
            # dm1 --> spinorb, spinorb
            # spinorb: (number of cas orbitals + core orbitals) * 2 because spin
            spinorb = (self._cas_data['ncas'] + self._cas_data['ncore']) *2
            norb = int(spinorb/2)
            print('LOADING MULTIREFERENCE METHOD DATA')
            logger.info(f'  Post-Hartree Fock CAS :', )
            print('Number of total orbitals HF:', calc._scf.mo_occ[calc._scf.mo_occ > 0].shape[0])
            print('Occupations orbitals HF:', calc._scf.mo_occ[calc._scf.mo_occ > 0])
            print('Number of total orbitals CAS:', norb)
            # Note: nelec only works for unrestricted
            # print('Number of total electrons :', calc._scf.nelec, calc._scf.mo_occ.sum())
            print('Number of core orbitals :', self._cas_data['ncore'])
            print('Number of active space orbitals :', self._cas_data['ncas'])
            print('Number of active space electrons : ', self._cas_data['nelecas'])

            dm1 = np.zeros((spinorb, spinorb))
            dm1_a = np.zeros((norb, norb))
            dm1_b = np.zeros((norb, norb))

            # fill with alpha and beta dm1 values
            # Dev: for now we follow APOST convention:
            # dm1[0:core, 0:core] = 1 in the diagonal
            # dm1[core:spinorb, core:spinorb] = order is alternate alpha and beta --> spinorb-1=1a,spinorb2=1b, spinorbital3=2a and so on
            # Only fill combinations of indices that are the same spin. For example indices 01 corresponds to first alpha spinorbital and first beta spinorbital. This is zero
            # CORE
            for i in range(self._cas_data['ncore'] * 2):
                dm1[i, i] = 1
            for i in range(self._cas_data['ncore']):
                dm1_a[i, i] = 1
                dm1_b[i, i] = 1
            spin_ncore = self._cas_data['ncore'] * 2
            for i in range(self._cas_data['ncas']*2):
                for j in range(self._cas_data['ncas']*2):
                    if i<self._cas_data['ncas'] and j<self._cas_data['ncas']:
                        dm1_a[i+self._cas_data['ncore'],j+self._cas_data['ncore']] =  casdm1s[0][i, j]
                        dm1_b[i+self._cas_data['ncore'],j+self._cas_data['ncore']] =  casdm1s[1][i, j]
                    if i== 0 and j==0:
                        dm1[i + spin_ncore, j + spin_ncore] = casdm1s[0][i, j]
                    elif i==0 and j%2 == 0:
                        j_a = int(j/2)
                        dm1[i+spin_ncore,j+spin_ncore] = casdm1s[0][i, j_a]
                    elif i%2 == 0 and j%2 == 0:
                        i_a = int(i / 2)
                        j_a = int(j / 2)
                        dm1[i+spin_ncore,j+spin_ncore] = casdm1s[0][i_a, j_a]
                    elif i%2 == 1 and j%2 == 1:
                        i_b = int((i+1)/2)-1
                        j_b = int((j+1)/2)-1
                        dm1[i+spin_ncore,j+spin_ncore] = casdm1s[1][i_b, j_b]

            # dm1_a[abs(dm1_a) < 1e-15] = 0
            # dm1_b[abs(dm1_b) < 1e-15] = 0
            # dm1[abs(dm1) < 1e-15] = 0
            # Test idempotency
            print('TEST idempotency', np.sum(np.dot(dm1,dm1) - dm1))
            print()

            # print('Building D_ij')
            print('Building Spinless 1-RDM (DM1)')
            dij = np.zeros(calc.mo_coeff.shape) # dim: basis functions, basis functions
            dij_a = np.zeros(calc.mo_coeff.shape)
            dij_b = np.zeros(calc.mo_coeff.shape)
            for i in range(0, int(spinorb/2)):
                for j in range(0, int(spinorb/2)):
                    dij[i, j] = dm1[i*2, j*2] + dm1[i*2+1, j*2+1]
                    dij_a[i, j] = dm1[i*2, j*2]
                    dij_b[i, j] = dm1[i*2+1, j*2+1]

            # diagonalizing Dij
            # Note: By diagonalizing the DM1 (Dij) we obtain the same eigenvalues/eigenvectors obtained
            # earlier up to some threshold. The diagonalization of DM1 in AO basis adds spurious very small
            # negative values (order 10^-14) as eigenvalues. This also makes the coefficients slightly different
            # if we included them in the diagonal of DM1 and diagonalize then we recover the same coefficients.
            # occ_no, coeff_no = scipy.linalg.eigh(dij)
            occ_no, coeff_no = np.linalg.eigh(dij)
            coeff_no = np.flip(coeff_no, axis=1)
            occ_no = occ_no[::-1] # Same as flip
            print('Natural orbitals from diagonalization of Dij')
            print('Natural orbital occupations (> 1e-8)')
            print('dm1 diagonalization')
            print('   ' + ' '.join([str(round(o, 4))  for o in occ_no if o > 1e-8]))
            # print('debug; ps diagonalization')
            # print('   ' + ' '.join([str(round(o/2, 4))  for o in occ if o > 1e-8]))

            # get density from NOs as a linear combination of MO
            # coeff_no: array(MO,NO)
            # coeff: array(AO, MO)
            # i:NO, k:MO, b:AO
            coeff_no_ao_basis = np.einsum('bk,ki ->bi', calc.mo_coeff[:, occ_no>1e-8], coeff_no[occ_no>1e-8,:])
            d = 0
            dm_no = np.dot(coeff_no_ao_basis*occ_no, coeff_no_ao_basis.T)
            for mu in range(coeff_no_ao_basis.shape[0]):
                for nu in range(coeff_no_ao_basis.shape[0]):
                    d += dm_no[mu, nu] * self._ovlp[nu, mu]
            print('CHECK: Density from Natural orbitals in AO basis: ', d)

            # Now saving dm2
            dm2_act = calc.fcisolver.make_rdm2(calc.ci, calc.ncas, calc.nelecas)
            dm2 = np.zeros((norb, norb, norb, norb))
            print('Spinless DM2')
            print('CHECK DM2 dim (active, active, active, active) :', dm2_act.shape)
            # We need to reconstruct the rest of the dm2: core and core-active block
            # Note: Core densities are represented by i index (rho1/ii/core) and j (rho2/jj/core) because core are canonical orbitals
            # and weren't included in active space. rho2 for active is represented by indices kl rho2/lk/active.
            # Spinless 2-RDM is composed by three blocks, rho1(core) rho2(core), rho1(core)rho2(active)
            # rho1(active)rho2(active). rho1(active)rho2(active) it's returned by calc.fcisolver.make_rdm2(calc.ci, calc.ncas, calc.nelecas)
            # and we need to reconstruct rho1(core)rho2(core) and rho1(core)rho2(active). Note rho1 and rho2 make reference to the same core density
            # For rho1(core)rho2(core) each one has 1 index to recompose the density rho(i,i)rho(j,j) and the only combinations that survive are
            # i,i,i,i (Coulomb)(a,a,b,b/a,a,b,b) i,i,j,j (Coulomb)(a,a,a,a/b,b,b,b/aabb/bbaa) and i,j,j,i (Exchange) (a,a,a,a/b,b,b,b)
            # For rho1(core)rho2(active) now rho2(active) indices go over active orbitals k/l. For rho1 i index, for the entry of iikl we could have
            # spin combinations of aa__, bb__, where ii gives 1 and we need to add the dm1 kl for active orbitals. Here again we have Coulomb contributions (a,a,a,a/b,b,b,b/aabb/bbaa)
            # adding dm1(kl) values for the corresponding spin, and exchange (a,a,a,a/b,b,b,b) and we add -dm1(kl) value.
            # Adding the negative value of dm1 for the exchange type of contributions will make to obtain the correct sign for the exchange
            # density but the inverse sign for the BOD.

            # Core
            for i in range(self._cas_data['ncore']):
                # dm2[i, i, i, i] = 2
                for j in range(self._cas_data['ncore']):
                    # Coulomb contribution: Because i=alpha,beta and j=alpha,beta --> this gives 4 possible combinations for (ii,jj) (aa,aa)(bb,bb)(aa,bb)(bb,aa)
                    dm2[i,i,j,j] = 4
                    # Exchange contribution: Because i=alpha,beta and j=alpha,beta --> this gives 2 possible combinations for (ij,ji) (aa,aa)(bb,bb)
                    # This contributes to exchange and removing self interaction from coulomb iijj
                    dm2[i,j,j,i] -= 2


            # core-active
            for i in range(self._cas_data['ncore']):
                for k in range(self._cas_data['ncore'], norb):
                    for l in range(self._cas_data['ncore'], norb):
                        # We are going to code first like apost from the dm1. However, it might be more efficient to use the spin resolve active orbitals
                        # dm1 obtained from pyscf before combining because the indices are the same for both. APOST does it that way because it reads it from the
                        # .dm1 file the printed indices already incorporate spin information (e.g alpha=odd number/beta=paired numbers)
                        # Coulomb terms (+)
                        dm2[i, i, k, l] = 2.0*dm1[k*2, l*2] + 2.0* dm1[l*2+1, k*2+1]  # dm1(aaaa) + dm1(bbbb) + dm1(aabb) + dm1(bbaa)
                        dm2[k, l, i, i] += 2.0 *dm1[k*2, l*2] + 2.0 *dm1[l*2+1, k*2+1]
                        # Exchange terms (-)
                        dm2[i,l,k,i] -= dm1[k*2, l*2] + dm1[k*2+1, l*2+1] #dm1(aa) + dm1(bb)
                        dm2[k,i,i,l] -= dm1[k*2, l*2] + dm1[k*2+1, l*2+1]




            # fill dm2 with dm2_active from pyscf
            # change indices order from 1212 to 1122 !Q: for now keeping them the same. Check APOST
            # dm2_act = np.transpose(dm2_act, axes=(0, 2, 1,3))
            for i in range(dm2_act.shape[0]):
                for k in range(dm2_act.shape[0]):
                    for j in range(dm2_act.shape[0]):
                        for l in range(dm2_act.shape[0]):
                            i_act = i + self._cas_data['ncore']
                            j_act = j + self._cas_data['ncore']
                            k_act = k + self._cas_data['ncore']
                            l_act = l + self._cas_data['ncore']
                            if abs(dm2_act[i,k,j,l]) > 1e-10:
                                dm2[i_act,k_act,j_act,l_act] += dm2_act[i,k,j,l]
            # Check trace
            # inactive orbitals
            t1=0
            t2=0
            t3=0
            for i in range(self._cas_data['ncore']):
                # for i in range(norb):
                for j in range(self._cas_data['ncore']):
                    # for j in range(norb):
                    t1 +=dm2[i,i,j,j]
            print('CHECK TRACE INACTIVE ORBITALS (N_inact(N_inact-1) ', t1)
            for i in range(self._cas_data['ncore'], norb):
                for j in range(self._cas_data['ncore'], norb):
                    t2 += dm2[i, i, j, j]
            print('CHECK TRACE ACTIVE ORBITALS (N_active(N_active-1) ', t1)

            for i in range(norb):
                for j in range(norb):
                    t3 += dm2[i, i, j, j]
            print('TOTAL TRACE', t3)

            # Saving DM1s and DM2
            self._dm1 = dm1 # spinless
            self._dm1_a = dm1_a
            self._dm1_b = dm1_b
            self._dm2 = dm2

            # For now only protected variables of the class because we are not sure if we will use all of them consistently
            # through the code
            self._occ_no = occ_no
            self._coeff_no = coeff_no
            self._cas_mo_coeff = calc.mo_coeff


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
                coeff_a = np.einsum('aj, ji->ai', calc.mo_coeff[0], coeff_a)
                coeff_b = np.einsum('aj, ji->ai', calc.mo_coeff[1], coeff_b)
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
                coeff = np.einsum('aj, ji->ai', calc.mo_coeff,  coeff)
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
            elif self._cas:
                logger.info(f'  Post-Hartree Fock CAS')
        logger.info(f'  Unrestricted : {self._unrestr}')
        logger.info(f'  DFT : {self._dft}\n')


        # Initialize Hpart or Rpart
        if part_name in ["mulliken", "lowdin", "meta_lowdin", "iao", "nao"]:
            self._part = Hpart(part_name, mol, frags=frags)
        elif part_name in ["becke", 'tfvc']:
            if self._unrestr:
                dm = (self._dm_a, self._dm_b)
            else:
                dm = self._dm_a
            # Check if specs
            if rpart_specs:
                if not isinstance(rpart_specs, dict):
                    raise TypeError(" rpart_specs must be a dict")
            self._part = Rpart(part_name, mol, dm, self._at_nums, frags=frags, grid_specs=rpart_specs)
        else:
            raise ValueError("Could not identify partition. Choose from: Mulliken,Lowdin,Meta-Lowdin,IAO,NAO")

    @property
    def natm(self):
        return self._natm
    @property
    def at_coords(self):
        return self._at_coords
    @property
    def at_nums(self):
        return self._at_nums
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
    def xc_code(self):
        return self._xc_code
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
    def no_coeff(self):
        return self._coeff_no

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
    def dm1(self):
        if not self._cas or self._ccsd:
            raise TypeError('DM1 not saved for SD methods')
        elif self._ccsd:
            raise NotImplementedError
        else:
            return self._dm1

    @property
    def dm2(self):
        if not self._cas or self._ccsd:
            raise TypeError('DM2 not saved for SD methods')
        elif self._ccsd:
            raise NotImplementedError
        else:
            return self._dm2

    @property
    def energy(self):
        return self._e_tot
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
    # todo: allow to specify radii type, radii method and empirical values (Bragg/Trehluer)

    def __init__(self, part_name, mol, dm, atnums, frags=None, sort_grids=False, grid_specs=None):

        self._part_name = part_name
        logger.info(f'  Initializing Rpart : {part_name.upper()} \n')
        # Keeping mol for now in a protected attribute because it is needed in other places outside init
        # Note: Think of a way of accessing mol differently. Maybe Rpart a subclass of APOSTBase? I still need
        # the at_nums and at_coords. Either this is transformed as a subclass only accessible from an APOSTBase one
        # or all this information is kept.
        self._mol = mol


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
        self._grid = Grids(mol)
        # Set up options different from defaults in pyscf class Grid
        self._grid.prune = None
        self._grid.radi_method = radi.becke
        # Create class Grids and set up parameters for radial and angular grids
        logger.info(f'  Grid information', )
        # Update if specs included
        if isinstance(grid_specs, dict):
            if 'radi_method' not in grid_specs.keys():
                logger.info(f'  Radi transf : radi_becke', )
            elif 'prune' not in grid_specs.keys():
                logger.info(f'  Prune : None', )
            elif 'level' not in grid_specs.keys():
                logger.info(f'  Level : Default', )
            for k,v in grid_specs.items():
                if k == 'prune':
                    if v is None:
                        self._grid.prune = None
                    elif v == 'nwchem_prune':
                        self._grid.prune = gen_grid.nwchem_prune
                    elif v == 'sg1_prune':
                        self._grid.prune = gen_grid.sg1_prune
                    elif v == 'treutler_prune':
                        self._grid.prune = gen_grid.treutler_prune
                    else:
                        raise TypeError(f"Specified type of pruned grid not identified: {v}")
                    logger.info(f'  Prune : {v}', )

                elif k == 'radi_method':
                    if None:
                        raise TypeError("Radii method must be specified")
                    elif v == 'radi_becke':
                        # Note: Rm can't be set up outside pyscf. Corresponds to Bragg radii
                        self._grid.radi_method = radi.becke
                    elif v == 'radi_treutler':
                        self._grid.radi_method = radi.treutler
                    elif v == 'radi_delley':
                        self._grid.radi_method = radi.delley
                    elif v == 'radi_mura_knowles':
                        self._grid.radi_method = radi.mura_knowles
                    elif v == 'radi_gauss_chebyshev':
                        self._grid.radi_method = radi.gauss_chebyshev
                    else:
                        raise TypeError(f"Specified type of radii method not identified: {v}")
                    logger.info(f'  Radi transf : {v}', )

                elif k == 'level':
                    if v > 9 or not isinstance(v, int):
                        raise ValueError("Level must be and integer between 0 and 9")
                    else:
                        self._grid.level = v
                    logger.info(f'  Points Level : {self._grid.level}', )
                elif k == 'atom_grid':
                    if not isinstance(v, dict):
                        raise TypeError(f"Specs for atomic grids need to be specified as dictionary")
                    else:
                        self._grid.atom_grid = v
                    logger.info(f'  Atomic grid dic : ', )
                    for k2, v2 in v.items():
                        logger.info(f'    Atom {k2} : rad {v2[0]}, ang(size){v2[1]}', )


        else:
                logger.info(f'  Radi transf : radi_becke', )
                logger.info(f'  Prune : None', )
                logger.info(f'  Level : Default', )

        logger.info(f'  (Note: Overrides --> Prune > level)', )
        logger.info(f'  (Note: Atom grids are pruned if both atomic grid specs and prune method are specified)', )
        # grid.becke_scheme = gen_grid.original_becke
        # grid.radii_adjust = radi.becke_atomic_radii_adjust
        # grid.atomic_radii = radi.BRAGG_RADII
        # From pyscf/dft/gen_grid.py # Padding grids to make the AO value generated by eval_gto aligned in memory
        self._grid.alignment = 0
        # with_non0tab: establishes a cutoff
        # from pyscf/dft/numint.py
        # screen_index: 2D uint8 array
        # How likely the AO values on grids are negligible. This array can be obtained by calling :func:`gen_grid.make_screen_index
        # sort_grids: it gets indices from arg_group_grids() it keeps the number of points the same (so far with the tests I checked)
        # but it reorganizes de points. Documentation from that function: Partition the entire space into small boxes according to the input box_size.
        # Group the grids against these boxes.
        # self._grid.build(with_non0tab=False, sort_grids=False)
        self._grid.build(with_non0tab=False, sort_grids=sort_grids)
        self._coords = self._grid.coords
        # Weights = radial(quadrature) + angular + becke
        self._weights = self._grid.weights
        self._grid_to_atom_idx = self._grid.atm_idx
        self._quadrature_weights = self._grid.quadrature_weights
        if part_name == 'becke':
            self._aim_w = becke_w(self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx)
            self._aim_w_all_p = becke_w(self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx, all_points=True)
        elif part_name == 'tfvc':
            if isinstance(dm, tuple) and len(dm) == 2:
                dm_all = dm[0].copy()
                dm_all += dm[1]
                self._aim_w = tfvc_w(mol, dm_all, self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx)
                self._aim_w_all_p = tfvc_w(mol, dm_all, self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx, all_points=True)
            else:
                self._aim_w = tfvc_w(mol, dm, self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx)
                self._aim_w_all_p = tfvc_w(mol, dm, self._coords, mol.atom_coords(), atnums, self._grid_to_atom_idx, all_points=True)
        else:
            raise NotImplementedError(f"Unknown part name: {part_name}")

        # Todo: decide what to do with computing the density. the question is how to minimize repetition of code as much as possible.
        # Evaluate density on a grid
        # Store computed atomic orbitals in a grid
        self._ao_eval_grid = eval_ao(mol, self._coords)
        dens_ao = eval_ao(mol, self._coords)
        if isinstance(dm, tuple) and len(dm) == 2:
            dm_a, dm_b = dm
            dens_a = np.zeros((dens_ao.shape[0]))
            dens_b = np.zeros((dens_ao.shape[0]))

        elif isinstance(dm, np.ndarray) and dm.shape[0] == dens_ao.shape[1] and dm.shape[1] == dens_ao.shape[1]:
            dm_a = dm
            dm_b = None
            dens_a = np.zeros((dens_ao.shape[0]))
        else:
            raise TypeError(f"Wrong type of dm")
        for mu in range(dens_ao.shape[1]):
            for nu in range(dens_ao.shape[1]):
                dens_a += dm_a[mu, nu] * dens_ao.T[nu, :] * dens_ao.T[mu, :]
                if dm_b is not None:
                    dens_b += dm_b[mu, nu] * dens_ao.T[nu, :] * dens_ao.T[mu, :]

        if dm_b is not None:
            self._dens_grid = (dens_a, dens_b)
        else:
            self._dens_grid = dens_a


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
    def quadrature_weights(self):
        return self._quadrature_weights

    @property
    def ao_eval_grid(self):
        return self._ao_eval_grid

    @property
    def dens_grid(self):
        return self._dens_grid

    @property
    def grid_to_atom_idx(self):
        return self._grid_to_atom_idx

    @property
    def aim_w(self):
        return self._aim_w

    @property
    def aim_w_allp(self):
        return self._aim_w_all_p

    @property
    def frags(self):
        return self._frags

    @property
    def zfrags(self):
        return self._zfrags

    def rotate_grid(self, at_coords, at_nums, pha, phb, dm):
        """Method to rotate atomic grids
        it is a compose rotation through the x and z directions with -pha and -phb angles
        """
        # Todo: add the latex matrix to the docstring
        # Note: if we regenerate the AIMs we need the DM. we could add a flag to make this an option besides the roation
        # or force it to alway pass it. For now passing it.
        # Warning: keep attention if we need to copy
        # First subtract atom coords from grid_coords
        grid_coords2_pre = np.zeros((len(at_nums), self.grid_coords.shape[0], self.grid_coords.shape[1]))
        grid_coords2_pre = [copy.copy(self.grid_coords)[self.grid_to_atom_idx == ia] - at_coords[ia] for ia in range(len(at_nums))]
        print(len(grid_coords2_pre))
        print(grid_coords2_pre[0].shape)
        print(grid_coords2_pre[1].shape)
        # Todo: Once the code is more stable add debug printings

        # 1. Get all the grid points
        # coords_to_roatate = copy.copy(self._grid.coords)
        print(f'ROTATING THE GRID PHA={pha}, PHB={phb}')
        rot_mat =  scipy.spatial.transform.Rotation.from_euler('zx', [-phb, -pha], degrees=False).as_matrix()
        # grid_coords2_p0 = np.dot(grid_coords2_pre_0, rot_mat)
        # grid_coords2_p1 = np.dot(grid_coords2_pre_1, rot_mat)
        # grid_coords2_0 = grid_coords2_p0 + at_coords[0]
        # grid_coords2_1 = grid_coords2_p1 + at_coords[1]
        # Todo: I should be able to modify each element wihout indexing
        for ia in range(len(grid_coords2_pre)):
            grid_coords2_pre[ia] = np.dot(grid_coords2_pre[ia], rot_mat)
            grid_coords2_pre[ia] = grid_coords2_pre[ia] + at_coords[ia]

        # grid_coords2 = np.vstack((grid_coords2_0, grid_coords2_1))
        grid_coords2 = np.vstack(grid_coords2_pre)
        # Generate the corresponding new AIM weights
        if self.part_name == 'becke':
            aim_rot = becke_w(grid_coords2, at_coords, at_nums, self._grid_to_atom_idx)
        elif self.part_name == 'tfvc':
            aim_rot = tfvc_w(self._mol, dm, grid_coords2, at_coords, at_nums, self._grid_to_atom_idx)
        else:
            raise NotImplementedError(f"Unknown part name: {self.part_name}")

        return (grid_coords2, aim_rot)


        print(self.grid_coords.shape)
        # subset of points
        testp = self.grid_coords[0, :]
        print(testp)
        assert 5 == 6


        

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
        elif part_name == "nao":
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





