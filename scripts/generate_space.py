import pandas as pd
from lib.chemical_space import create_combination, bicyclic_core, tricyclic_core


if __name__ == '__main__':

    bi_cores_list = [('Diels-Alder','C=CMgC=C([Os])C([Os])=C>>C1MgC2C([Os])=C([Os])CC1'),
                     ('[3+2]cycloaddition','[Os]C=CMgC([Ir])=[N+]([O-])\C>>[Os][C@@H]1CMg[C@@]2([Ir])N(C)O1'),
                     ('[3+2]cycloaddition','[Os]C(Mg/C(C(C)C)=[N+]([O-])\[Ir])=C=O>>[Os]CMg=C1O[C@]2(C(C)C)N([Ir])O1'),
                     ('[3+2]cycloaddition','[Os]/C(MgC([Ir])=C/[Ir])=[N+]1N[C@H]2CC[C@@H]/1C2>>[Os][C@@]1MgN3[NH+]([C@@H]([C@]12[Ir])[Ir])[C@H]4CC[C@@H]3C4')]
    tri_cores_list = ['C1=C([Os])C([Os])=CMgC=CMg1>>C12C([Os])=C([Os])C3MgC3C1Mg2',]

    rxns = []
    for sub in ['C', 'C([Os])C', 'CC([Os])', 'C(=O)C(=O)', 'CC(=O)O', 'C([Os])CCCC', 'C([Os])CCCCC']:
        rxn = tricyclic_core(tri_cores_list[0], sub)
        rxns.append(('Diels-Alder', rxn))
        for bi_core in bi_cores_list:
            rxn = bicyclic_core(bi_core[1], sub)
            rxns.append((bi_core[0], rxn))

    rxns = list(set(rxns))
    df_fused_cycles = pd.DataFrame(rxns, columns=['Type', 'rxn_smiles'])

    df = pd.read_csv('../data/hypothetical_space_core.csv', sep=';')

    df = pd.concat([df, df_fused_cycles], ignore_index=True, axis=0)

    df_space = pd.DataFrame(columns=['Type', 'rxn_smiles'])
    for row in df.itertuples():
        df_space = pd.concat([df_space, create_combination(row.rxn_smiles, row.Type)], axis=0, ignore_index=True)
    df_space.to_csv('hypothetical_chemical_space.csv')