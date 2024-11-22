import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def line_plot_fps_esi(log_file):

    with open(log_file, 'r') as file:
        lines = file.readlines()

    data = []
    append_line = False
    for line in lines:
        if 'Fingerprint:' in line:
            _, fps, rad, nbits = line.split()
            rad = int(rad[-2])
            nbits = int(nbits[6: -1])
            append_line = True
        if '6-fold nested-cv' in line and append_line:
            splitted_line = line.split()
            model, rmse, mae, r2 = splitted_line[7], float(splitted_line[-3]), float(splitted_line[-2]), float(
                splitted_line[-1])
            append_line = False
            data.append((model, fps, rad, nbits, rmse, mae, r2))

    df = pd.DataFrame(data, columns=['model', 'fingerprint_type', 'radius', 'nbits', 'rmse', 'mae', 'r2'])
    df = df.loc[df.nbits >= 256]
    df_drfp = df.loc[df['fingerprint_type'] == 'DRFP']
    df_morgan = df.loc[df['fingerprint_type'] == 'Morgan']

    # Plot for Morgan
    plot_metrics(df_morgan, 'morgan_fps', row_offset=0, title_prefix='Morgan')

    # Plot for DRFP (shifted by 3 rows)
    plot_metrics(df_drfp, 'drfp_fps', row_offset=0, title_prefix='DRFP')


def plot_metrics(data, name, row_offset=0, title_prefix=''):

    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    metrics = ['rmse', 'mae', 'r2']
    models = ['k-NN', 'RF', 'XGBoost']
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            sns.lineplot(data=data[data['model'] == model], y=metric, x='nbits', hue='radius', style='radius',
                         markers=True, ax=axes[i + row_offset, j])
            if i == 0:
                axes[i + row_offset, j].set_title(f'model = {model}', fontsize=12)
            if j == 0:
                if metric != 'r2':
                    axes[i + row_offset, j].set_ylabel(metric.upper() + (' (kcal/mol)'), fontsize=12)
                else:
                    axes[i + row_offset, j].set_ylabel('R$^2$', fontsize=12)
            if (i == len(metrics) - 1) and (j == 1):
                axes[i + row_offset, j].set_xlabel('Dimensionality', fontsize=12)
            else:
                axes[i + row_offset, j].set_xlabel('')
            if j != 0:
                axes[i + row_offset, j].set_ylabel('')
            axes[i + row_offset, j].set_xticks([500, 1000, 1500, 2000])
            axes[i + row_offset, j].set_xlim([150, 2100])

    plt.suptitle(f'{title_prefix} Fingerprints', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{name}.pdf', dpi=300)


def distribution_2a(csv_file_path):

    df_rxn_smiles = pd.read_csv(csv_file_path, sep=';')
    df_rxn_smiles['number_reacs'] = df_rxn_smiles['rxn_smiles'].apply(lambda x: len(x.split(">>")[0].split('.')))

    is_multi = df_rxn_smiles["Type"].value_counts() > 1
    df_rxn_smiles_filtered = df_rxn_smiles[df_rxn_smiles["Type"].isin(is_multi[is_multi].index)]
    dict_replace = {'[3+2]cycloaddition': '[3+2] cycloaddition', '[4+6]cycloaddition': '[4+6] cycloaddition',
                    '[6+4]cycloaddition': '[6+4] cycloaddition', '[8+2]cycloaddition': '[8+2] cycloaddition',
                    '[3,3]rearrangement': '[3,3] rearrangement', '[6+4]intramolecular': '[4+6] intramolecular'}
    df_rxn_smiles_filtered = df_rxn_smiles_filtered.replace(dict_replace)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    g_1 = sns.kdeplot(data=df_rxn_smiles_filtered, x="Std_DFT_forward", hue="Type", fill=True)

    axs.set_ylabel('Density', fontsize=14)
    axs.margins(x=0, y=0)
    axs.set_xlim(0.0, 8.5)
    axs.set_ylim(0.0, 0.12)

    axs.set_xlabel('$\sigma$ ($\Delta$E$^{\ddag}$) forward (kcal/mol)', fontsize=14)
    sns.move_legend(g_1, loc='upper left')

    axs.text(0.790, 0.88,
                f"Mean: {df_rxn_smiles_filtered['Std_DFT_forward'].mean():.3f} kcal/mol\n      $\sigma$: {df_rxn_smiles_filtered['Std_DFT_forward'].std():.3f} kcal/mol",
                transform=axs.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig('fig_2a.png', dpi=300)


def histogram_2b(csv_file_path):

    df_original = pd.read_csv(csv_file_path)
    df_DA = df_original.loc[df_original['Type'] == 'Diels-Alder']

    df_lower_std = df_DA.loc[df_DA['Std_DFT_forward'] < 3.5]
    df_higher_std = df_DA.loc[df_DA['Std_DFT_forward'] > 3.5]

    columns = [column for column in df_DA.columns if 'DELTA-forward' in column]

    dict_low = {}
    dict_high = {}

    for column in columns:
        functional = column[:-14]
        mae_lower = df_lower_std[column].abs().mean()
        dict_low[functional] = dict_low.get(functional, 0) + mae_lower

        mae_higher = df_higher_std[column].abs().mean()
        dict_high[functional] = dict_high.get(functional, 0) + mae_higher

    dict_low_sorted = sorted(dict_low.items(), key=lambda x: x[1])
    dict_high_sorted = sorted(dict_high.items(), key=lambda x: x[1])

    df_pics_1 = pd.DataFrame(dict_low_sorted, columns=['Functional', 'MAE'])
    df_pics_2 = pd.DataFrame(dict_high_sorted, columns=['Functional', 'MAE'])

    df_pics_2['Subset'] = ['Benchmarking subset 2'] * len(df_pics_2)
    df_pics_1['Subset'] = ['Benchmarking subset 1'] * len(df_pics_1)

    df_pics = pd.concat([df_pics_1, df_pics_2], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.set(style="white")
    plot = sns.barplot(x='Functional', y='MAE', hue='Subset', data=df_pics, palette='dark')
    plot.set_title('')
    locs, _ = plt.xticks()
    locs = [loc - 0.3 for loc in locs]
    functionals = ['$\omega$B2-PLYP', '$\omega$B97M-V', 'M06-2X', '$\omega$B97X-V', 'M06-L', 'RSX-0DH', 'B2-PLYP',
                   'M06', 'B2K-PLYP', 'B97M-V', 'PBE0-DH', 'PBE0', 'BLYP', 'B3LYP', '$\omega$B2-PLYP18', 'RSX-QIDH',
                   'PBE-QIDH', 'CAM-B3LYP', '$\omega$B97X', 'BHandHLYP']

    plt.xticks(ticks=locs, rotation=45, labels=functionals)
    plt.ylabel('MAE (kcal/mol)')

    plt.tight_layout()
    plt.savefig('fig_2b.png', dpi=300)

    df = pd.DataFrame(dict_low_sorted, columns=['Functional', 'MAE low STD'])
    df['MAE high STD'] = df['Functional'].apply(lambda x: dict_high[x])

    df.to_csv('mae_per_functional.csv')


def plot_error_cv_prediction_bo(csv_file_path):

    mae_cv = [0.520, 0.538, 0.550, 0.527, 0.550, 0.572, 0.531, 0.552]

    subtitle_a = 'Nested CV'
    subtitle_b = 'Prediction'
    subtitle_c = 'Acquired points'

    df = pd.read_csv(csv_file_path, sep=';')

    df = calculate_jitter(df)
    mae_per_round = df.groupby('round').apply(lambda x: np.mean(np.abs(x['TRUE'] - x['pred'])))

    # Plot using Seaborn
    fig, axs = plt.subplots(1, 3, figsize=(20, 4.5))

    sns.lineplot(x=[0, 1, 2, 3, 4, 5, 6, 7], y=mae_cv, alpha=0.6, ax=axs[0], color='royalblue', linewidth=2.5,  marker='o')

    axs[0].set_xlabel('')
    axs[0].set_ylabel('MAE (kcal/mol)', fontsize=14)
    axs[0].set_title(subtitle_a, fontsize=14)
    axs[0].set_xticks(range(0, 8))
    axs[0].set_xticklabels(range(0, 8))
    axs[0].set_xlim(0, 7)
    axs[0].set_ylim(0.45, 0.65)

    sns.lineplot(x=mae_per_round.index, y=mae_per_round.values, alpha=0.6, ax=axs[1], color='royalblue', linewidth=2.5,  marker='o')

    axs[1].set_xticks(range(1, df['round'].max() + 1))
    axs[1].set_xticklabels(range(1, df['round'].max() + 1))

    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_title(subtitle_b, fontsize=14)

    axs[1].set_xlim(1, df['round'].max())

    dict_replace = {'3+2': '[3+2] cycloaddition', 'DA': 'Diels-Alder', '3,3': '[3,3] rearrangement'}
    df = df.replace(dict_replace)
    df_max = extract_max_values(df)
    df_max_3_2 = df_max.loc[df_max['type'] == '[3+2] cycloaddition']
    df_max_3_3 = df_max.loc[df_max['type'] == '[3,3] rearrangement']
    df_max_DA = df_max.loc[df_max['type'] == 'Diels-Alder']

    custom_palette = {
    '[3+2] cycloaddition': '#66c2a5',  '[3,3] rearrangement': '#fc8d62', 'Diels-Alder': '#8da0cb'}

    sns.scatterplot(data=df,
                    x='round_jitter',
                    y='TRUE',
                    marker='o',
                    hue='type',
                    axes=axs[2],
                    palette=custom_palette)
    axs[2].plot(df_max_DA['round_jitter'], df_max_DA['TRUE'], marker='o', ms=0, c='steelblue',
                label='current best Diels-Alder')
    DA_max_value = 6.68
    axs[2].axhline(y=DA_max_value, color='steelblue', linewidth=0.5, linestyle='--', label='baseline Diels-Alder')

    axs[2].plot(df_max_3_3['round_jitter'], df_max_3_3['TRUE'], marker='o', ms=0, c='orange',
                label='current best [3,3] rearrangement')
    rear_max_value = 5.15
    axs[2].axhline(y=rear_max_value, color='orange', linewidth=0.5, linestyle='--',
                   label='baseline [3,3] rearrangement')

    axs[2].plot(df_max_3_2['round_jitter'], df_max_3_2['TRUE'], marker='o', ms=0, c='green',
                label='current best [3+2] cycloaddition')
    cyclo_max_value = 5.22
    axs[2].axhline(y=cyclo_max_value, color='green', linewidth=0.5, linestyle='--',
                   label='baseline [3+2] cycloaddition')

    axs[2].set_xlabel('')
    axs[2].set_ylabel('$\sigma$($\Delta$E$^{\ddag}$) forward (kcal/mol)', fontsize=16)

    for r in df['round'].unique():
        axs[2].axvline(x=r, color='gray', linewidth=0.75, linestyle='--')

    custom_x_labels = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5', 'Round 6', 'Round 7']
    rounds = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    axs[2].set_xticks(rounds)
    axs[2].set_xticklabels(custom_x_labels, rotation=45)
    axs[2].set_title(subtitle_c, fontsize=14)
    axs[2].set_xlim(1, 8)

    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.text(0.30, 0.01, 'Round', ha='center', fontsize=14)

    plt.tight_layout()

    plt.savefig(f"fig_3_top.png", dpi=300)


def calculate_jitter(df):
    df['round_jitter'] = 0.0
    for r in df['round'].unique():
        mask = df['round'] == r
        n_points = mask.sum()
        jitter_values = np.linspace(r, r + 1, n_points + 2, dtype='float')[1:-1]
        jitter_values = jitter_values.round(decimals=3)
        df.loc[mask, 'round_jitter'] = jitter_values
    return df


def extract_max_values(df):

    max_value_DA = 2.56
    max_value_3_2 = 2.56
    max_value_3_3 = 2.56
    df_max = pd.DataFrame(columns=['TRUE', 'round_jitter', 'type'])

    df_max = pd.concat([None, pd.DataFrame({'TRUE': [max_value_DA], 'round_jitter': [1.0], 'type': 'Diels-Alder'})], ignore_index=True)
    df_max = pd.concat([df_max, pd.DataFrame({'TRUE': [max_value_3_2], 'round_jitter': [1.0], 'type': '[3+2] cycloaddition'})],ignore_index=True)
    df_max = pd.concat([df_max, pd.DataFrame({'TRUE': [max_value_3_3], 'round_jitter': [1.0], 'type': '[3,3] rearrangement'})],ignore_index=True)

    for row in df.itertuples():
        if row.type == 'Diels-Alder':
            if row.TRUE > max_value_DA:
                new_row = {'TRUE': [row.TRUE], 'round_jitter': [row.round_jitter], 'type': 'Diels-Alder'}
                df_max = pd.concat([df_max, pd.DataFrame(new_row)], ignore_index=True)
                max_value_DA = row.TRUE

        if row.type == '[3+2] cycloaddition':
            if row.TRUE > max_value_3_2:
                new_row = {'TRUE': [row.TRUE], 'round_jitter': [row.round_jitter], 'type': '[3+2] cycloaddition'}
                df_max = pd.concat([df_max, pd.DataFrame(new_row)], ignore_index=True)
                max_value_3_2 = row.TRUE

        if row.type == '[3,3] rearrangement':
            if row.TRUE > max_value_3_3:
                new_row = {'TRUE': [row.TRUE], 'round_jitter': [row.round_jitter], 'type': '[3,3] rearrangement'}
                df_max = pd.concat([df_max, pd.DataFrame(new_row)], ignore_index=True)
                max_value_3_3 = row.TRUE

    df_max = pd.concat([df_max, pd.DataFrame({'TRUE': [max_value_DA], 'round_jitter': [8.0], 'type': 'Diels-Alder'})], ignore_index=True)
    df_max = pd.concat([df_max, pd.DataFrame({'TRUE': [max_value_3_2], 'round_jitter': [8.0], 'type': '[3+2] cycloaddition'})], ignore_index=True)
    df_max = pd.concat([df_max, pd.DataFrame({'TRUE': [max_value_3_3], 'round_jitter': [8.0], 'type': '[3,3] rearrangement'})], ignore_index=True)

    return df_max


def final_histogram(reference_file, final_bo_file, initial_data_file):

    df_ref = pd.read_csv(reference_file, sep="\s+")

    df_fx = pd.read_csv(final_bo_file)

    df_fx_rxn = df_fx.loc[df_fx['Name'].isin(df_ref['rxn'])]
    df_fx_rxn.reset_index(inplace=True)
    df_fx_rxn = df_fx_rxn.drop(columns=['index', 'Unnamed: 0'])

    df_fx_rxn = pd.concat([df_fx_rxn, df_ref], axis=1)

    for column in df_fx_rxn.columns[1:-10]:

        if 'forward' in column:
            reference_column = 'forward'
        elif 'reverse' in column:
            reference_column = 'reverse'
        else:
            reference_column = 'reaction'
        df_fx_rxn[f"{column}-DELTA"] = abs(df_fx_rxn[column] - df_fx_rxn[reference_column])

    columns = [column for column in df_fx_rxn.columns if 'DELTA' in column]

    dict_fx = {}

    for column in columns:

        if 'forward' in column:
            functional = column[:-18]
        elif 'reverse' in column:
            functional = column[:-18]
        else:
            functional = column[:-19]

        mae = df_fx_rxn[column].mean()
        dict_fx[functional] = dict_fx.get(functional, []) + [mae]

    df_bo = pd.DataFrame.from_dict(dict_fx, orient='index', columns=['MAE forward', 'MAE reverse', 'MAE reaction'])
    df_bo.to_csv('mae_final_functional.csv')

    dict_fx_bh9 = get_mae_full_data(initial_data_file)
    df_fx_bh9 = pd.DataFrame.from_dict(dict_fx_bh9, orient='index', columns=['MAE forward', 'MAE reverse', 'MAE reaction'])

    df_fx_bh9.sort_values(by=['MAE forward'], inplace=True)

    df_fx_bh9['Subset'] = ['BH9 dataset'] * len(df_fx_bh9)
    df_bo['Subset'] = ['Act. learning dataset'] * len(df_bo)

    df_fx_bh9.reset_index(inplace=True)
    df_bo.reset_index(inplace=True)

    df_pics = pd.concat([df_fx_bh9, df_bo], ignore_index=True)
    df_pics.rename(columns={'index': 'Functional'}, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.set(style="white")
    plot = sns.barplot(x='Functional', y='MAE forward', hue='Subset', data=df_pics, palette='dark')
    plot.set_title('')
    locs, _ = plt.xticks()
    locs = [loc - 0.3 for loc in locs]
    functionals = ['$\omega$B97M-V', '$\omega$B2-PLYP', 'B2K-PLYP', 'M06-2X', 'PBE0-DH', '$\omega$B97X-V', 'M06',
                   'B2-PLYP', 'PBE0', 'PBE-QIDH', 'RSX-0DH', 'B97M-V', 'M06-L', '$\omega$B2-PLYP18', 'RSX-QIDH',
                   '$\omega$B97X', 'CAM-B3LYP', 'B3LYP', 'BLYP', 'BHandHLYP']

    plt.xticks(ticks=locs, rotation=45, labels=functionals)
    plt.ylabel('MAE (kcal/mol)')

    plt.tight_layout()
    plt.savefig('fig_4.pdf', dpi=300)


def get_mae_full_data(initial_data_file):

    df_original = pd.read_csv(initial_data_file)
    df_bo = df_original.loc[df_original['Type'].isin(['Diels-Alder', '[3+2]cycloaddition', '[3,3]rearrangement'])]

    dict_bo = {}

    columns = [column for column in df_bo.columns if 'DELTA' in column]

    for column in columns:

        if 'forward' in column:
            functional = column[:-14]
        elif 'reverse' in column:
            functional = column[:-14]
        else:
            functional = column[:-15]

        mae_bo = df_bo[column].abs().mean()
        dict_bo[functional] = dict_bo.get(functional, []) + [mae_bo]

    return dict_bo


if __name__ == '__main__':

    line_plot_fps_esi('final_log/output.log')
    distribution_2a('../data/data_smiles_curated.csv')
    histogram_2b('../data/final_overview_data.csv')
    plot_error_cv_prediction_bo('../data/file_visualization.csv')
    final_histogram('../data/nrg23_mod.txt', '../data/raw_data_round_8/final_overview_data_8.csv',
                    '../data/final_overview_data.csv')

