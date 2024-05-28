import pandas as pd
from lib.chemical_space import create_combination, bicyclic_core, tricyclic_core


if __name__ == '__main__':

    bi_cores_list = [('Diels-Alder','C=CMgC=C([Os])C([Ir])=C>>C1MgC2C([Os])=C([Ir])CC1'),
                     ('[3+2]cycloaddition','[Os]C=CMgC([Ir])=[N+]([O-])\C>>[Os][C@@H]1CMg[C@@]2([Ir])N(C)O1'),
                     ('[3+2]cycloaddition','[Os]C(Mg/C(C(C)C)=[N+]([O-])\[Ir])=C=O>>[Os]CMg=C1O[C@]2(C(C)C)N([Ir])O1'),
                     ('[3+2]cycloaddition','[Ir]/C(MgC([Os])=C/[Ir])=[N+]1N[C@H]2CC[C@@H]/1C2>>[Ir][C@@]1MgN3[NH+]([C@@H]([C@]12[Os])[Ir])[C@H]4CC[C@@H]3C4')]
    tri_cores_list = ['C1=C([Os])C([Ir])=CMgC=CMg1>>C12C([Os])=C([Ir])C3MgC3C1Mg2']

    rxns = []
    for sub in ['C', 'C([Os])C', 'CC([Os])', 'C(=O)C(=O)', 'CC(=O)O', 'C([Os])CCCC', 'C([Os])CCCCC']:
        rxn = tricyclic_core(tri_cores_list[0], sub)
        rxns.append(('Diels-Alder', rxn))
        for bi_core in bi_cores_list:
            rxn = bicyclic_core(bi_core[1], sub)
            rxns.append((bi_core[0], rxn))

    df_fused_cycles = pd.DataFrame(rxns, columns=['Type', 'rxn_smiles'])
    df_fused_cycles.drop_duplicates(inplace=True, subset=['rxn_smiles'])

    df = pd.read_csv('../data/hypothetical_space_core.csv', sep=';')

    df = pd.concat([df, df_fused_cycles], ignore_index=True, axis=0)

    df_space = pd.DataFrame(columns=['Type', 'rxn_smiles'])
    for row in df.itertuples():
        df_space = pd.concat([df_space, create_combination(row.rxn_smiles, row.Type)], axis=0, ignore_index=True)
    df_space.to_csv('../data/hypothetical_chemical_space_set_sanitize.csv')