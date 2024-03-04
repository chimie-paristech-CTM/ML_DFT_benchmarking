import os
from glob import glob
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--directory', type=str, default='../data/rxns_benchmarked.csv',
                    help='path to csv file containing reactions')


def extract_info(directory):

    pwd = os.getcwd()

    # reactants and products
    os.chdir(f"{directory}/reaction/thermal")
    logs = glob(f'*.log')
    species = []
    for log in logs:
        G_abs = extract_energy(log, 'GTot')
        species.append((log.split('_')[0], G_abs))
    os.chdir(pwd)

    # transition states
    os.chdir(f"{directory}/reaction/output")
    ts = [ts_file for ts_file in glob('TS*.xyz') if 'imag' not in ts_file]
    with open(ts[0], 'r') as file:
        lines = file.readlines()
    freq_xyz = float(lines[1].split()[-2])
    os.chdir(pwd)

    os.chdir(f"{directory}/reaction/transition_states")
    tss = glob('TS*.log')
    for ts in tss:
        freq_log = freq_from_gaussian(ts)
        if abs(freq_log - freq_xyz) < 0.5:
            G_abs = extract_energy(ts, 'GTot')
            species.append((ts.split('_')[0], G_abs))
            break
    os.chdir(pwd)

    G_reac, G_prod, G_ts = 0, 0, 0

    for specie in species:
        if 'r' in specie[0]:
            G_reac += float(specie[1])
        if 'p' in specie[0]:
            G_prod += float(specie[1])
        if 'TS' in specie[0]:
            G_ts += float(specie[1])

    forward_barrier = (G_ts - G_reac) * 627.509
    reverse_barrier = (G_ts - G_prod) * 627.509
    dG_rxn = (G_prod - G_reac) * 627.509

    return forward_barrier, reverse_barrier, dG_rxn


def freq_from_gaussian(ts_file):
    """ Get the imaginary freq of the gaussian output """

    with open(ts_file, 'r') as out_gaussian:
        out_lines = out_gaussian.readlines()

    low_freqs = []

    for line in out_lines:

        if "Low frequencies ---" in line:
            low_freqs.append(line)
            break

    if low_freqs:
        frequency = float(low_freqs[0][20:].split()[0])
        return frequency

    else:
        return None


def extract_energy(file, keyword):

    with open(file, 'r') as out_gaussian:
        lines = out_gaussian.readlines()[::-1]

    block_lines = []
    append_line = False

    for line in lines:
        if (
                r"\\@" in line
                or line.startswith(" @")
                or line.startswith(r" \@")
        ):
            append_line = True

        if append_line:
            #                 Strip off new-lines and spaces
            block_lines.append(line.strip("\n").strip(" "))

        if (
                "Unable to Open any file for archive entry." in line
        ):
            break

    energy_list = [split for split in "".join(block_lines[::-1]).split(r"\\") if keyword in split][0].split("\\")
    for element in energy_list:
        if keyword in element:
            energy = float(element.split("=")[-1])
            return energy

    return "ERROR"


if __name__ == '__main__':
    args = parser.parse_args()
    os.chdir(args.directory)
    folders = os.listdir()
    info = []
    for folder in folders:
        if "rxn_" in folder:
            forward_barrier, reverse_barrier, dG_rxn = extract_info(folder)
            info.append((forward_barrier, reverse_barrier, dG_rxn, folder))

    df = pd.DataFrame(info, columns=['Forward_barrier', 'Reverse_barrier', 'Reaction_energy', 'rxn'])
    df.to_csv('results.csv')
