import numpy as np

import logging

from pyscf.dft.numint import eval_ao, eval_rho



# Note: if basis set big, for memory efficiency might be better to access already evaluated AO on the grid
# insteadof calling eval_ao every time
def eval_dens(mol, dm, coords):
    """Utility to evaluate density on a grid of points. Equivalent to pyscf/dft/numint.py eval_rho()"""
    dens_ao = eval_ao(mol, coords)
    dens = np.zeros((dens_ao.shape[0]))
    print(dens.shape,  dens_ao.shape, coords.shape)
    for mu in range(dens_ao.shape[1]):
        for nu in range(dens_ao.shape[1]):
            dens += dm[mu, nu] * dens_ao.T[mu, :] * dens_ao.T[nu, :]

    return dens

def out_summary(iqa_data, at_nums, total_energy):

    if len(iqa_data) != 2 and not isinstance(iqa_data[0], dict):
        raise ValueError(f"iqa_data must have length 1 or 2")

    logging.info('Summary IQA')
    print(f"Nucleus-Nucleus repulsion energy: {iqa_data[0]['t_nn']:.5f}")
    print('------------------------------------')
    if len(iqa_data) == 2:
        # BETA ########################
        print('ALPHA/BETA ELECTRONS (UNRESTRICTED)')
        print(f"Electron-Nucleus attraction energy: {iqa_data[1]['t_en']:.5f}")
        print(
            f"Electron-Nucleus attraction energy (PySCF/Libcint): {iqa_data[1]['en_analyt']: .5f}")
        diff = abs(iqa_data[1]['t_en'] - iqa_data[1]['en_analyt'])
        print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)

        print('Condensed atomic EN energy')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_data[1]['at_en_cond'][idx]: .5f}")
        print('Atomic EN energy matrix')
        print("         " + "       ".join([str(a) for a in at_nums]))
        for idx1, at1 in enumerate(at_nums):
            row_r = []
            for idx2, at2 in enumerate(at_nums):
                row_r.append(
                    str(round(iqa_data[1]['at_en'][idx1, idx2], 5)))
            print(f"{at1}   " + " ".join(row_r))
        print()



        print('------------------------------------')
        print(f"Kinetic energy:  {iqa_data[1]['t_kin']:.5f}")
        print(f"Kinetic energy(PySCF/Libcint):  {iqa_data[1]['kin_analyt']: .5f}")
        diff = abs(iqa_data[1]['t_kin'] - iqa_data[1]['kin_analyt'])
        print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print()
        print('Atomic Kinetic energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_data[1]['at_kin'][idx]: .5f}")

        if 't_coul' in iqa_data[1].keys():
            print('------------------------------------')
            if len(iqa_data[1]['at_coul'].shape) == 1:
                print('Electron-Electron interaction energy')
            elif len(iqa_data[1]['at_coul'].shape) == 2:
                print('Interatomic Electron-Electron interaction energy')
            print()
            print(f"Coulomb energy: {iqa_data[1]['t_coul']: .5f}", )
            print(f"Coulomb energy(PySCF/Libcint):  {iqa_data[1]['coul_analyt']: .5f}")
            diff = abs(iqa_data[1]['t_coul'] - iqa_data[1]['coul_analyt'])
            print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
            print()
            if len(iqa_data[1]['at_coul'].shape) == 1:
                print('Atomic Coulomb energy:')
                for idx, at in enumerate(at_nums):
                    print(f"{at}   {iqa_data[1]['at_coul'][idx]: .5f}")
                print()
            elif len(iqa_data[1]['at_coul'].shape) == 2:
                print('Full Coulomb energy decomposition:')
                print("         " + "       ".join([str(a) for a in at_nums]))
                for idx1, at1 in enumerate(at_nums):
                    row_r = []
                    for idx2, at2 in enumerate(at_nums):
                        row_r.append(str(round(iqa_data[1]['at_coul'][idx1, idx2], 5)))
                    print(f"{at1}   " + " ".join(row_r))
                # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
                print()
        if 't_exch' in iqa_data[1].keys():
            for si,s in enumerate(['ALPHA', 'BETA']):
                print(f'HF EXCHANGE FOR SPIN ORBITALS {s}')
                print(f"HF Exchange: {iqa_data[si]['t_exch']: .5f}")
                if iqa_data[1].get('coeff_mix', None):
                    print('COEFF MIX :', iqa_data[1]['coeff_mix'])
                    print(f"HF Exchange energy(PySCF/Libcint):  {iqa_data[si]['exch_analyt']: .5f}")
                    print(f"HF Exchange energy(PySCF/Libcint):  {iqa_data[si]['exch_analyt'] * iqa_data[1]['coeff_mix']: .5f}")
                    diff = abs(iqa_data[si]['t_exch'] - iqa_data[si]['exch_analyt'] * iqa_data[1]['coeff_mix'])
                    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
                    print()
                    if len(iqa_data[1]['at_exch'].shape) == 1:
                        print('Atomic HF Exchange energy:')
                        for idx, at in enumerate(at_nums):
                            print(f"{at}   {iqa_data[1]['at_exch'][idx] * iqa_data[1]['coeff_mix']: .5f}")
                        print()
                    elif len(iqa_data[1]['at_exch'].shape) == 2:
                        print('Full HF Exchange energy decomposition:')
                        print("         " + "       ".join([str(a) for a in at_nums]))
                        for idx1, at1 in enumerate(at_nums):
                            row_r = []
                            for idx2, at2 in enumerate(at_nums):
                                row_r.append(str(round(iqa_data[1]['at_exch'][idx1, idx2] * iqa_data[1]['coeff_mix'], 5)))
                            print(f"{at1}   " + " ".join(row_r))
                        # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
                        print()
                else:
                    print(f"HF Exchange energy(PySCF/Libcint):  {iqa_data[si]['exch_analyt']: .5f}")
                    diff = abs(iqa_data[si]['t_exch'] - iqa_data[si]['exch_analyt'])
                    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
                    print()
                    if len(iqa_data[1]['at_exch'].shape) == 1:
                        print('Atomic HF Exchange energy:')
                        for idx, at in enumerate(at_nums):
                            print(
                                f"{at}   {iqa_data[1]['at_exch'][idx]: .5f}")
                        print()
                    elif len(iqa_data[1]['at_exch'].shape) == 2:
                        print('Full HF Exchange energy decomposition:')
                        print("         " + "       ".join([str(a) for a in at_nums]))
                        for idx1, at1 in enumerate(at_nums):
                            row_r = []
                            for idx2, at2 in enumerate(at_nums):
                                row_r.append(str(round(iqa_data[1]['at_exch'][idx1, idx2], 5)))
                            print(f"{at1}   " + " ".join(row_r))
                        # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
                        print()
        print()

        if 't_dft_xc' in iqa_data[1].keys():
            print(f"DTF Exchange-Correlation energy: {iqa_data[1]['t_dft_xc']: .5f}")
            if len(iqa_data[1]['at_dft_xc'].shape) == 1:
                print('Atomic DFT Exchange-Correlation energy:')
                for idx, at in enumerate(at_nums):
                    print(
                        f"{at}   {iqa_data[1]['at_dft_xc'][idx]: .5f}")
                print()
            elif len(iqa_data[1]['at_dft_xc'].shape) == 2:
                print('BOND ORDER')
                for idx1 in range(len(at_nums)):
                    for idx2 in range(idx1 + 1):
                        if idx1 != idx2:
                            print(f"ATOMS: {at_nums[idx1]}   {at_nums[idx2]} {iqa_data[1]['bod'][idx1, idx2]: .5f}")
                print(f'Full DFT Exchange-Correlation energy decomposition ({iqa_data[1]["bod_type"].upper()} approach):')
                print("         " + "       ".join([str(a) for a in at_nums]))
                for idx1, at1 in enumerate(at_nums):
                    row_r = []
                    for idx2, at2 in enumerate(at_nums):
                        row_r.append(str(round(iqa_data[1]['at_dft_xc'][idx1, idx2], 5)))
                    print(f"{at1}   " + " ".join(row_r))
                # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
            print(f"DTF Exchange-Correlation energy: {iqa_data[1]['t_dft_xc']: .5f}")
            print()

        total_sum_energy_a = 0
        for k in iqa_data[0].keys():
            if k.startswith('t_'):
                if k == 't_nn':
                    pass
                else:
                    total_sum_energy_a += iqa_data[0][k]

        total_sum_energy_b = 0
        for k in iqa_data[1].keys():
            if k.startswith('t_'):
                if k == 't_nn':
                    pass
                else:
                    total_sum_energy_b += iqa_data[1][k]

        print('TOTAL ENERGY ALPHA (SUM OF COMPONENTS):', total_sum_energy_a)
        print('TOTAL ENERGY BETA (SUM OF COMPONENTS):', total_sum_energy_b )
        print()
        print('TOTAL ENERGY (SUM OF COMPONENTS):', total_sum_energy_a + total_sum_energy_b + iqa_data[0]['t_nn'])
        print('TOTAL ENERGY (PySCF):', total_energy)
        print('DIFF (AU) :',total_sum_energy_a + total_sum_energy_b + iqa_data[0]['t_nn'] - total_energy)
        print('DIFF (Kcal) :', (total_sum_energy_a + total_sum_energy_b + iqa_data[0]['t_nn'] - total_energy) / 0.0015936014376406278)

        print()

        ###############################
    else:
        print('ALPHA/BETA ELECTRONS (RESTRICTED)')
        print(f"Electron-Nucleus attraction energy: {iqa_data[0]['t_en']:.5f}")
        print(
            f"Electron-Nucleus attraction energy (PySCF/Libcint): {iqa_data[0]['en_analyt']: .5f}")
        diff = abs(iqa_data[0]['t_en'] - iqa_data[0]['en_analyt'])
        print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print('Condensed atomic EN energy')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_data[0]['at_en_cond'][idx]: .5f}")
        print('Atomic EN energy matrix')
        print("         " + "       ".join([str(a) for a in at_nums]))
        for idx1, at1 in enumerate(at_nums):
            row_r = []
            for idx2, at2 in enumerate(at_nums):
                row_r.append(
                    str(round(iqa_data[0]['at_en'][idx1, idx2], 5)))
            print(f"{at1}   " + " ".join(row_r))
        print()
        print('------------------------------------')
        print(f"Kinetic energy:  {iqa_data[0]['t_kin']:.5f}")
        print(f"Kinetic energy(PySCF/Libcint):  {iqa_data[0]['kin_analyt']: .5f}")
        diff = abs(iqa_data[0]['t_kin'] - iqa_data[0]['kin_analyt'])
        print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print()
        print('Atomic Kinetic energy:')
        for idx, at in enumerate(at_nums):
            print(f"{at}   {iqa_data[0]['at_kin'][idx]: .5f}")

        if 't_coul' in iqa_data[0].keys():
            print('------------------------------------')
            if len(iqa_data[0]['at_coul'].shape) == 1:
                print('Electron-Electron interaction energy')
            elif len(iqa_data[0]['at_coul'].shape) == 2:
                print('Interatomic Electron-Electron interaction energy')

            print()
            print(f"Coulomb energy: {iqa_data[0]['t_coul']: .5f}", )
            print(f"Coulomb energy(PySCF/Libcint):  {iqa_data[0]['coul_analyt']: .5f}")
            diff = abs(iqa_data[0]['t_coul'] - iqa_data[0]['coul_analyt'])
            print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
            print()
            if len(iqa_data[0]['at_coul'].shape) == 1:
                print('Atomic Coulomb energy:')
                for idx, at in enumerate(at_nums):
                    print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
                print()
            elif len(iqa_data[0]['at_coul'].shape) == 2:
                print('Full Coulomb energy decomposition:')
                print("         " + "       ".join([str(a) for a in at_nums]))
                for idx1, at1 in enumerate(at_nums):
                    row_r = []
                    for idx2, at2 in enumerate(at_nums):
                        row_r.append(str(round(iqa_data[0]['at_coul'][idx1, idx2], 5)))
                    print(f"{at1}   " + " ".join(row_r))
                # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
                print()


        if 't_exch' in iqa_data[0].keys():
            print(f"HF Exchange: {iqa_data[0]['t_exch']: .5f}")
            if iqa_data[0].get('coeff_mix', None):
                print('COEFF MIX :', iqa_data[0]['coeff_mix'])
                print(f"HF Exchange energy(PySCF/Libcint):  {iqa_data[0]['exch_analyt'] : .5f}")
                print(f"HF Exchange energy(PySCF/Libcint):  {iqa_data[0]['exch_analyt'] * iqa_data[0]['coeff_mix']: .5f}")
                diff = abs(iqa_data[0]['t_exch'] - iqa_data[0]['exch_analyt']* iqa_data[0]['coeff_mix'])
                print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
                print()
                if len(iqa_data[0]['at_exch'].shape) == 1:
                    print('Atomic HF Exchange energy:')
                    for idx, at in enumerate(at_nums):
                        print(f"{at}   {iqa_data[0]['at_exch'][idx] * iqa_data[0]['coeff_mix']: .5f}")
                    print()
                elif len(iqa_data[0]['at_exch'].shape) == 2:
                    print('Full HF Exchange energy decomposition:')
                    print("         " + "       ".join([str(a) for a in at_nums]))
                    for idx1, at1 in enumerate(at_nums):
                        row_r = []
                        for idx2, at2 in enumerate(at_nums):
                            row_r.append(str(round(iqa_data[0]['at_exch'][idx1, idx2] * iqa_data[0]['coeff_mix'], 5)))
                        print(f"{at1}   " + " ".join(row_r))
                    # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
                    print()
            else:
                print(f"HF Exchange energy(PySCF/Libcint):  {iqa_data[0]['exch_analyt']: .5f}")
                diff = abs(iqa_data[0]['t_exch'] - iqa_data[0]['exch_analyt'])
                print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
                print()
                print('Atomic HF Exchange energy:')
                if len(iqa_data[0]['at_exch'].shape) == 1:
                    for idx, at in enumerate(at_nums):
                        print(f"{at1}   {iqa_data[0]['at_exch'][idx]: .5f}")
                    print()
                elif len(iqa_data[0]['at_coul'].shape) == 2:
                    print('Full HF Exchange energy decomposition:')
                    print("         " + "       ".join([str(a) for a in at_nums]))
                    for idx1, at1 in enumerate(at_nums):
                        row_r = []
                        for idx2, at2 in enumerate(at_nums):
                            row_r.append(str(round(
                                iqa_data[0]['at_exch'][idx1, idx2], 5)))
                        print(f"{at1}   " + " ".join(row_r))
                    # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
                    print()
        if 't_dft_xc' in iqa_data[0].keys():
            print(f"DTF Exchange-Correlation energy: {iqa_data[0]['t_dft_xc']: .5f}")
            print()
            print('Atomic DFT Exchange-Correlation energy:')
            if len(iqa_data[0]['at_dft_xc'].shape) == 1:
                for idx, at in enumerate(at_nums):
                    print(f"{at1}   {iqa_data[0]['at_dft_xc'][idx]: .5f}")
                print()
            elif len(iqa_data[0]['at_dft_xc'].shape) == 2:
                print('BOND ORDER')
                for idx1 in range(len(at_nums)):
                    for idx2 in range(idx1 + 1):
                        if idx1 != idx2:
                            print(f"ATOMS: {at_nums[idx1]}   {at_nums[idx2]} {iqa_data[0]['bod'][idx1, idx2]: .5f}")
                print(f'Full DFT Exchange-Correlation energy decomposition ({iqa_data[0]["bod_type"].upper()} approach)::')
                print("         " + "       ".join([str(a) for a in at_nums]))
                for idx1, at1 in enumerate(at_nums):
                    row_r = []
                    for idx2, at2 in enumerate(at_nums):
                        row_r.append(str(round(iqa_data[0]['at_dft_xc'][idx1, idx2], 5)))
                    print(f"{at1}   " + " ".join(row_r))
                # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
                print()

        print()
        total_sum_energy = 0
        for k in iqa_data[0].keys():
            if k.startswith('t_'):
                total_sum_energy += iqa_data[0][k]

        print('TOTAL ENERGY (SUM OF COMPONENTS):', total_sum_energy)
        print('TOTAL ENERGY (PySCF):', total_energy)
        print('DIFF (AU) :', total_sum_energy - total_energy)
        print('DIFF (Kcal) :', (total_sum_energy - total_energy) / 0.0015936014376406278)
        #############################


def out_summary_phf(iqa_data, at_nums, total_energy, correlation=False):
    """WIP"""


    logging.info('Summary IQA  Post-Hartree Fock calculation')
    print('ALPHA/BETA ELECTRONS (RESTRICTED)')
    print(f"Electron-Nucleus attraction energy: {iqa_data[0]['t_en']:.5f}")
    print(
        f"Electron-Nucleus attraction energy (PySCF/Libcint): {iqa_data[0]['en_analyt']: .5f}")
    diff = abs(iqa_data[0]['t_en'] - iqa_data[0]['en_analyt'])
    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
    for idx, at in enumerate(at_nums):
        print(f"{at}   {iqa_data[0]['at_en_cond'][idx]: .5f}")
    print('------------------------------------')
    print(f"Kinetic energy:  {iqa_data[0]['t_kin']:.5f}")
    print(f"Kinetic energy(PySCF/Libcint):  {iqa_data[0]['kin_analyt']: .5f}")
    diff = abs(iqa_data[0]['t_kin'] - iqa_data[0]['kin_analyt'])
    print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
    print()
    print('Atomic Kinetic energy:')
    for idx, at in enumerate(at_nums):
        print(f"{at}   {iqa_data[0]['at_kin'][idx]: .5f}")
    # Note: For now we alway compute coulomb for iqa_phf
    if 't_coul' in iqa_data[0].keys():
        print('------------------------------------')
        # if len(iqa_data[0]['at_coul'].shape) == 1:
        #     print('Electron-Electron interaction energy')
        # elif len(iqa_data[0]['at_coul'].shape) == 2:
        print('Interatomic Electron-Electron interaction energy')

        print()
        print(f"Coulomb energy: {iqa_data[0]['t_coul']: .5f}", )
        print(f"Coulomb energy(PySCF/Libcint):  {iqa_data[0]['coul_analyt']: .5f}")
        diff = abs(iqa_data[0]['t_coul'] - iqa_data[0]['coul_analyt'])
        print('DIFF (au) ', diff)
        print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print()
        # if len(iqa_data[0]['at_coul'].shape) == 1:
        #     print('Atomic Coulomb energy:')
        #     for idx, at in enumerate(at_nums):
        #         print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
        #     print()
        # elif len(iqa_data[0]['at_coul'].shape) == 2:
        print('Full Coulomb energy decomposition:')
        print("         " + "       ".join([str(a) for a in at_nums]))
        for idx1, at1 in enumerate(at_nums):
            row_r = []
            for idx2, at2 in enumerate(at_nums):
                row_r.append(str(round(iqa_data[0]['at_coul'][idx1, idx2], 5)))
            print(f"{at1}   " + " ".join(row_r))
        # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
        print()
    if 't_xc' in iqa_data[0].keys():
        print(f"Exchange-Correlation energy: {iqa_data[0]['t_xc']: .5f}", )
        # print(f"Coulomb energy(PySCF/Libcint):  {iqa_data[0]['coul_analyt']: .5f}")
        # diff = abs(iqa_data[0]['t_coul'] - iqa_data[0]['coul_analyt'])
        # print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print()
        print('XC cumulant decomposition')
        print("         " + "       ".join([str(a) for a in at_nums]))
        for idx1, at1 in enumerate(at_nums):
            row_r = []
            for idx2, at2 in enumerate(at_nums):
                row_r.append(str(round(iqa_data[0]['at_xc'][idx1, idx2], 5)))
            print(f"{at1}   " + " ".join(row_r))
        # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
        print()
    if correlation:
        print('Independent exchange and correlation part')
        print('Exchange')
        print(f"Exchange energy: {iqa_data[0]['t_x']: .5f}", )
        print(f"Exchange energy(PySCF/Libcint):  {iqa_data[0]['exch_analyt']: .5f}")
        diff = abs(iqa_data[0]['t_x']) - abs(iqa_data[0]['exch_analyt'])
        print('DIFF (au) ', diff)
        print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print()
        print("         " + "       ".join([str(a) for a in at_nums]))
        for idx1, at1 in enumerate(at_nums):
            row_r = []
            for idx2, at2 in enumerate(at_nums):
                row_r.append(str(round(iqa_data[0]['at_x'][idx1, idx2], 5)))
            print(f"{at1}   " + " ".join(row_r))
        # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")
        print()
        print('Correlation')
        print(f"Correlation energy: {iqa_data[0]['t_c']: .5f}", )
        # print(f"Coulomb energy(PySCF/Libcint):  {iqa_data[0]['coul_analyt']: .5f}")
        # diff = abs(iqa_data[0]['t_coul'] - iqa_data[0]['coul_analyt'])
        # print('DIFF (kcal/mol) ', diff / 0.0015936014376406278)
        print()
        print("         " + "       ".join([str(a) for a in at_nums]))
        for idx1, at1 in enumerate(at_nums):
            row_r = []
            for idx2, at2 in enumerate(at_nums):
                row_r.append(str(round(iqa_data[0]['at_c'][idx1, idx2], 5)))
            print(f"{at1}   " + " ".join(row_r))
        # print(f"{at}   {iqa_data[0]['at_coul'][idx]: .5f}")

        vee_calc = iqa_data[0]['t_c'] +iqa_data[0]['t_coul'] + iqa_data[0]['t_x']
        diff = abs(vee_calc- iqa_data[0]['vee'])
        print(f"EE energy:  {vee_calc: .5f}")
        print(f"EE energy(PySCF/Libcint):  {iqa_data[0]['vee']: .5f}")
        print('DIFF (au) vee', diff)
        print('DIFF (kcal/mol) vee', diff / 0.0015936014376406278)
        print()
        total_sum_energy = 0
        for k in iqa_data[0].keys():
            if k.startswith('t_'):
                # print(k, iqa_data[0][k])
                # if k in ['t_x', 't_c']:
                #     pass
                # else:
                    total_sum_energy += iqa_data[0][k]

        print('TOTAL ENERGY (SUM OF COMPONENTS):', total_sum_energy)
        print('TOTAL ENERGY (PySCF):', total_energy)
        print('DIFF (AU) :', total_sum_energy - total_energy)
        print('DIFF (Kcal) :', (total_sum_energy - total_energy) / 0.0015936014376406278)
        #############################


def out_bod(at_nums, bod_dict):
    """Printing BOD results"""

    for scheme in bod_dict.keys():
        print(f'BOND ORDER: {scheme}')
        for idx1, at1 in enumerate(at_nums):
            row_r = []
            for idx2, at2 in enumerate(at_nums):
                row_r.append(str(round(bod_dict[scheme][idx1, idx2], 5)))
            print(f"{at1}   " + " ".join(row_r))

