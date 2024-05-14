import os
import subprocess
from glob import glob
from xyz2mol import read_xyz_file, xyz2mol
from rdkit import Chem
import shutil


def check_TS_stereochemistry():

    pwd = os.getcwd()

    ou_folder = os.path.join(pwd, 'reaction/output')
    if os.path.exists(ou_folder):

        os.chdir(ou_folder)

        ts_files = glob('TS*')
        prod_file = glob('p0*')[0]
        product = xyz_to_mol(prod_file)

        if len(ts_files) > 0:

            ts_file = [ts_file for ts_file in ts_files if not 'imag_mode' in ts_file][0]

            os.makedirs('ts_check')
            ts_check_dir = os.path.join(ou_folder, 'ts_check')

            name = ts_file.split('.')[0]

            shutil.copy(ts_file, ts_check_dir)
            os.chdir(ts_check_dir)
            run_xtb(name)
            normal_termination = check_output_xtb(f'{name}.log')
            if normal_termination:
                copy_xyz_output(f'{name}_opt.xyz')
            else:
                print('Error termination xTB')

            ts_mol = xyz_to_mol(f'{name}_opt.xyz')

            if is_the_same_stereochemistry(ts_mol, product):
                print(f"TS is leading to the correct stereo-product\n")
            else:
                print(f"TS is leading to the wrong stereo-product\n")
                ts_mol_noHs = Chem.RemoveHs(ts_mol)
                ts_smiles = Chem.MolToSmiles(ts_mol_noHs)
                ts_smiles = canonicalize_smiles(ts_smiles)
                optimize_new_product(ts_smiles)

            os.chdir(pwd)


def run_xtb(name):

    xtb_path = '/home/javialra/soft/xtb-6.6.0/bin'
    xtb_command = os.path.join(xtb_path, 'xtb')

    ou_file = f'{name}.log'

    with open(ou_file, 'w') as out:
        subprocess.run(f"{xtb_command} {name}.xyz --opt --parallel 4", shell=True, stdout=out, stderr=out)


def check_output_xtb(out_file):
    out_lines = []

    with open(out_file, 'r') as file:
        out_lines = file.readlines()

    opt_criteria = "   *** GEOMETRY OPTIMIZATION CONVERGED AFTER "

    for line in reversed(out_lines):
        if "ERROR" in line:
            return False
        if opt_criteria in line:
            return True

    return False


def copy_xyz_output(xyz_file):
    out_xyz_xtb = 'xtbopt.xyz'
    os.rename(out_xyz_xtb, xyz_file)


def xyz_to_mol(xyz_file):
    """ Convert 3D coordinates into a rkdit molecule,
    Using xyz2mol, from https://github.com/jensengroup/xyz2mol
    """

    atoms, charge, xyz_coordinates = read_xyz_file(xyz_file)
    mol = xyz2mol(atoms, xyz_coordinates, charge)[0]

    return mol


def is_the_same_stereochemistry(mol_TS, mol_prod):
    """ Compare if the TS is leading to the correct stereo_product """

    stereo_info_TS = Chem.FindMolChiralCenters(mol_TS, force=True, includeUnassigned=True, useLegacyImplementation=True)

    stereo_info_prod = Chem.FindMolChiralCenters(mol_prod, force=True, includeUnassigned=True,
                                                 useLegacyImplementation=True)

    return True if stereo_info_prod == stereo_info_TS else False


def canonicalize_smiles(smiles):
    """ Return a consistent SMILES representation for the given molecule """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def optimize_new_product(smiles):
    os.chdir('../')
    os.makedirs(f"opt_alt_p0")
    working_dir = os.path.join(os.getcwd(), f"opt_alt_p0")
    os.chdir(working_dir)
    input_ade(smiles)
    input_bash()
    run_ade()


def input_ade(smiles):
    """ Create the input for the new ade calculation """

    with open(f"p0_alt.py", 'w') as ade_input:
        ade_input.write('import autode as ade\n')
        ade_input.write('from autode.wrappers.G16 import g16\n')
        ade_input.write('ade.Config.n_cores=4\n')
        ade_input.write('ade.Config.max_core=4000\n')
        ade_input.write('g16.keywords.set_functional(\'cam-b3lyp\')\n')
        ade_input.write(f"ade.Config.G16.keywords.opt.basis_set = '6-311++G**' \n")
        ade_input.write(f"ade.Config.G16.keywords.low_opt.basis_set = '6-31G*' \n")
        ade_input.write(f"ade.Config.num_conformers=1000\n")
        ade_input.write(f"ade.Config.rmsd_threshold=0.1\n")
        ade_input.write(f"product = ade.Molecule(name='p0_alt', smiles=r\"{smiles}\")\n")
        ade_input.write('product.find_lowest_energy_conformer(hmethod=g16)\n')
        ade_input.write('product.optimise(method=g16)\n')
        ade_input.write('product.calc_thermo(method=g16)\n')
        ade_input.write('product.print_xyz_file()\n')


def input_bash():
    """ Create the input for the sh input """

    with open(f"p0_alt.sh", 'w') as bash_input:
        bash_input.write(
            '__conda_setup="$(\'/home/javialra/anaconda3/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)"\n')
        bash_input.write('if [ $? -eq 0 ]; then\n')
        bash_input.write('\teval "$__conda_setup"\n')
        bash_input.write('else\n')
        bash_input.write('\tif [ -f "/home/javialra/anaconda3/etc/profile.d/conda.sh" ]; then\n')
        bash_input.write('\t\t. "/home/javialra/anaconda3/etc/profile.d/conda.sh"\n')
        bash_input.write('\telse\n')
        bash_input.write('\t\texport PATH="/home/javialra/anaconda3/bin:$PATH"\n')
        bash_input.write('\tfi\n')
        bash_input.write('fi\n')
        bash_input.write('conda activate autode_original\n')
        bash_input.write(f"python3 p0_alt.py\n")


def run_ade():
    """ Run the ade calculation"""

    chmod = f"chmod +x p0_alt.sh"
    subprocess.run(chmod, shell=True)

    ade_command = f"./p0_alt.sh"
    out_file = f"./p0_alt.out"

    with open(out_file, 'w') as out:
        subprocess.run(ade_command, shell=True, stdout=out, stderr=out)


if __name__ == "__main__":
    check_TS_stereochemistry()
