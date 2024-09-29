from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--smiles', type=str,
                    help='SMILES of the new product')
parser.add_argument('--conda_env', type=str, default='autode',
                    help='conda environment of autodE package')
parser.add_argument('--idx', type=str,
                    help='index of the reaction')
parser.add_argument('--ind', type=str, help='r0, r1 or p0')

def create_ade_input(smiles, idx, dir, ind, hmet_confor=r"True"):
    """ Create the ade input """

    # Setting the calculation

    functional = 'cam-b3lyp'
    conf_basis_set = '6-31G*'
    basis_set = '6-311++G**'
    cores = 24
    mem = 4000
    num_conf = 1000
    rmsd = 0.1

    file_name = f"{ind}_alt_{idx}.py"

    with open(f"{dir}/{file_name}", 'w') as in_ade:
        in_ade.write('import autode as ade\n')
        in_ade.write('from autode.wrappers.G16 import g16\n')
        in_ade.write("if __name__ == \"__main__\":  \n")
        in_ade.write(f"\tade.Config.n_cores={cores}\n")
        in_ade.write(f"\tade.Config.max_core={mem}\n")
        in_ade.write(f"\tade.Config.hcode=\"G16\"\n")
        in_ade.write(f"\tade.Config.lcode =\"xtb\"\n")
        in_ade.write(f"\tg16.keywords.set_functional('{functional}')\n")
        in_ade.write(f"\tg16.keywords.opt.basis_set = '{basis_set}' \n")
        in_ade.write(f"\tg16.keywords.hess.basis_set = '{basis_set}' \n")
        in_ade.write(f"\tg16.keywords.low_opt.basis_set = '{conf_basis_set}' \n")
        in_ade.write(f"\tg16.keywords.low_opt.max_opt_cycles = 15\n")
        in_ade.write(f"\tade.Config.num_conformers={num_conf}\n")
        in_ade.write(f"\tade.Config.rmsd_threshold={rmsd}\n")
        in_ade.write(f"\tade.Config.hmethod_conformers={hmet_confor}\n")
        in_ade.write(f"\tproduct = ade.Molecule(name='{ind}_{idx:07}', smiles=r\"{smiles}\")\n")
        in_ade.write('\tproduct.find_lowest_energy_conformer(hmethod=g16)\n')
        in_ade.write('\tproduct.optimise(method=g16)\n')
        in_ade.write('\tproduct.calc_thermo(method=g16)\n')
        in_ade.write('\tproduct.print_xyz_file()\n')

    return None


def create_slurm(idx, dir, ind, conda_env):
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
        in_slurm.write(f"python3 {ind}_alt_{ade_idx}.py \n")

    return None


if __name__ == "__main__":
    # set up
    args = parser.parse_args()
    new_path = os.path.join(os.getcwd(), f"{args.ind}_alt_{args.idx}")
    os.mkdir(new_path)
    create_ade_input(args.smiles, args.idx, new_path, args.ind)
    create_slurm(args.idx,  new_path, args.ind, args.conda_env)
