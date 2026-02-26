# pyAPOST

Python version of APOST3d developed as a PySCF extension.

```bash
# To install the repo
git clone git@github.com:leila-pujal/pyAPOST.git

cd pyAPOST
pip install -e .

```


# Dependencies

- [PySCF](https://github.com/pyscf/pyscf)

# Example

```
from pyscf import gto, scf, mcscf
from pyAPOST.base import APOSTBase
from pyAPOST.effos import EOS_calc

mol = gto.M(atom = [
            [8,( 0, 0, 0)],
            [8, (0, 0, 1.2)] ], basis = 'ccpvdz')
mol.ms = 1 # spin quantum number: triplet with ms=1
mol.charge = 0
mol.cart= True
mol.max_memory = 4000

# start with reference RHF calculation.
myhf = scf.RHF(mol)
myhf.kernel()

#then full-valence CAS TRIPLET STATE with 6 orbitals, 8 electrons
mycas = mcscf.CASSCF(myhf, 6, 8)
mycas.kernel()

# Create an APOSTBase object defining partition and fragments (if None are specified default is Atoms)
apost_obj = APOSTBase(mol, mycas, part_name="tfvc")

# Call EOS_calc function to perfom a full EOS calculation
EOS_calc(apost_obj, gen=False, vis='Molden')

```




