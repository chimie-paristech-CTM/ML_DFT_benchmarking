#!/usr/bin/python
import os
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--final_dir', type=str, default='../data/autodE_input',
                    help='path to the folder with the autodE input')
parser.add_argument('--conda_env', type=str, default='autodE',
                    help='conda environment of autodE package')
parser.add_argument('--input', type=str, default=None,
                    help='path to txt file containing reactions')


def compare_workflow():

    df = pd.read_csv('../data/data_smiles_curated.csv', sep=';', index_col=0)
    rxn_benchmarked = df.sample(n=7, random_state=7) # 5%
    rxn_benchmarked.to_csv('../data/rxns_benchmarked.csv')

def create_input_autodE(final_dir, input_file, conda_env="autodE"):

    create_input(input_file, final_dir, conda_env)

    return None


def create_input(input_file, final_dir, conda_env, hmet_confor=r"True"):
    """ Create the input for autoDE"""

    with open(input_file, 'r') as file:
        lines = file.readlines()

    current_dir = os.getcwd()

    # checking directory
    if os.path.isdir(final_dir):
        os.chdir(final_dir)
    else:
        os.mkdir(final_dir)
        os.chdir(final_dir)

    for line in lines:

        idx, rxn_smile = line.split()

        directory = f"rxn_{idx}"
        os.mkdir(directory)

        create_ade_input(rxn_smile, idx, directory, hmet_confor=hmet_confor)
        create_slurm(idx, directory, conda_env)

        with open(f"{directory}/rxn_smile.txt", 'w') as rxn_file:
            rxn_file.write(rxn_smile)
    
    os.chdir(current_dir)

    return None


def create_ade_input(rxn_smile, idx, dir, hmet_confor=r"True"):
    """ Create the ade input """

    # Setting the calculation

    functional = 'cam-b3lyp'
    conf_basis_set = '6-31G*'
    basis_set = '6-311++G**'
    cores = 24
    mem = 4000
    num_conf = 1000
    rmsd = 0.3

    file_name = f"ade_{idx}.py"

    with open(f"{dir}/{file_name}", 'w') as in_ade:
        in_ade.write('import autode as ade\n')
        in_ade.write(f"ade.Config.n_cores={cores}\n")
        in_ade.write(f"ade.Config.max_core={mem}\n")
        in_ade.write(f"ade.Config.hcode=\"G16\"\n")
        in_ade.write(f"ade.Config.lcode =\"xtb\"\n")
        in_ade.write(f"rxn=ade.Reaction(r\"{rxn_smile}\")\n")
        in_ade.write(f"ade.Config.G16.keywords.set_functional('{functional}')\n")
        in_ade.write(f"ade.Config.G16.keywords.opt.basis_set = '{basis_set}' \n")
        in_ade.write("ade.Config.G16.keywords.opt.append('tight') \n")
        #in_ade.write(f"ade.Config.G16.keywords.sp.basis_set = '{final_basis_set}' \n")
        in_ade.write(f"ade.Config.G16.keywords.opt_ts.basis_set = '{basis_set}' \n")
        in_ade.write(f"ade.Config.G16.keywords.hess.basis_set = '{basis_set}' \n")
        in_ade.write(f"ade.Config.G16.keywords.low_opt.basis_set = '{conf_basis_set}' \n")
        in_ade.write(f"ade.Config.G16.keywords.low_opt.max_opt_cycles = 20\n")
        in_ade.write(f"ade.Config.num_conformers={num_conf}\n")
        in_ade.write(f"ade.Config.rmsd_threshold={rmsd}\n")
        in_ade.write(f"ade.Config.hmethod_conformers={hmet_confor}\n")
        in_ade.write('rxn.calculate_reaction_profile(free_energy=True)\n')

    return None


def create_slurm(idx, dir, conda_env):
    """ Create the slurm input """

    # Setting the calculation

    nodes = 1
    tasks_per_node = 1
    cpus_per_task = 24
    log_level = 'INFO' # {DEBUG, INFO, WARNING, ERROR}

    file_name = f"slurm_{idx}.sh"
    ade_idx = f"{idx}"

    with open(f"{dir}/{file_name}", 'w') as in_slurm:
        in_slurm.write('#!/bin/bash\n')
        in_slurm.write(f"#SBATCH --job-name=ade_{ade_idx}\n")   
        in_slurm.write(f"#SBATCH --nodes={nodes}\n")    
        in_slurm.write(f"#SBATCH --ntasks-per-node={tasks_per_node}\n")
        in_slurm.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")  
        in_slurm.write('#SBATCH --qos=qos_cpu-t3\n')   
        in_slurm.write('#SBATCH --time=20:00:00\n')
        in_slurm.write('#SBATCH --hint=nomultithread  # Disable hyperthreading\n')
        in_slurm.write(f"#SBATCH --output=ade_{ade_idx}_%j.out\n")   
        in_slurm.write(f"#SBATCH --error=ade_{ade_idx}_%j.err\n") 
        in_slurm.write(f"#SBATCH --account=qev@cpu\n")
        in_slurm.write('module purge\n')
        in_slurm.write('module load xtb/6.4.1\n') 
        in_slurm.write('module load gaussian/g16-revC01\n')
        in_slurm.write('module load python\n') 
        in_slurm.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
        in_slurm.write('export GAUSS_SCRDIR=$JOBSCRATCH\n')
        in_slurm.write(f"conda activate {conda_env}\n")
        in_slurm.write(f"export AUTODE_LOG_LEVEL={log_level}\n")
        in_slurm.write(f"export AUTODE_LOG_FILE=ade_{ade_idx}.log\n")
        in_slurm.write(f"python3 ade_{ade_idx}.py \n")

    return None


if __name__ == "__main__":
    # set up
    args = parser.parse_args()
    create_input_autodE(args.final_dir, args.input, args.conda_env)
