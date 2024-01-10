import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plot_energy_distribution(csv_file, energy_column):
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_file}' is empty.")
        return

    # Check if the specified column exists in the DataFrame
    if energy_column not in df.columns:
        print(f"Error: Column '{energy_column}' not found in the DataFrame.")
        return

    # Plot the distribution of energy values using a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[energy_column], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {energy_column}')
    plt.xlabel('Energy Values')
    plt.ylabel('Frequency')
    plt.show()


def scatter_plot(csv_file_path, x_column, y_column):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the specified columns
    x_values = df[x_column]
    y_values = df[y_column]

    # Create a scatter plot
    plt.scatter(x_values, y_values)
    
    # Add labels and title
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')

    # Show the plot
    plt.show()


def line_plot(log_file):

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
        if '4-fold CV' in line and append_line:
            splitted_line = line.split()
            model, rmse, mae, r2 = splitted_line[7], float(splitted_line[-3]), float(splitted_line[-2]), float(
                splitted_line[-1])
            append_line = False
            data.append((model, fps, rad, nbits, rmse, mae, r2))

    df = pd.DataFrame(data, columns=['model', 'fingerprint_type', 'radius', 'nbits', 'rmse', 'mae', 'r2'])
    df_drfp = df.loc[df['fingerprint_type'] == 'DRFP']
    df_morgan = df.loc[df['fingerprint_type'] == 'Morgan']

    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'k-NN'], y='rmse', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[0, 0])
    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'RF'], y='rmse', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[0, 1])
    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'xgboost'], y='rmse', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[0, 2])

    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'k-NN'], y='mae', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[1, 0])
    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'RF'], y='mae', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[1, 1])
    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'xgboost'], y='mae', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[1, 2])

    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'k-NN'], y='r2', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[2, 0])
    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'RF'], y='r2', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[2, 1])
    sns.lineplot(data=df_morgan.loc[df_morgan['model'] == 'xgboost'], y='r2', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[2, 2])

    [axes[i, j].set_xlabel("") for i in range(3) for j in range(3)]
    [axes[i, j].set_ylabel("") for i in range(3) for j in range(1, 3)]
    [axes[i, j].set_xticks([0, 500, 1000, 1500, 2000]) for i in range(3) for j in range(3)]

    axes[0, 0].set_ylabel('RMSE (kcal/mol)', fontsize=12)
    axes[1, 0].set_ylabel('MAE (kcal/mol)', fontsize=12)
    axes[2, 0].set_ylabel('R$^2$ (kcal/mol)', fontsize=12)
    axes[2, 1].set_xlabel('Dimensionality', fontsize=12)
    axes[0, 0].set_title('model = k-NN', fontsize=12)
    axes[0, 1].set_title('model = RF', fontsize=12)
    axes[0, 2].set_title('model = XGBoost', fontsize=12)

    plt.suptitle('Morgan Fingerprints')
    plt.savefig('morgan_fps.png')

    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'k-NN'], y='rmse', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[0, 0])
    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'RF'], y='rmse', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[0, 1])
    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'xgboost'], y='rmse', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[0, 2])

    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'k-NN'], y='mae', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[1, 0])
    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'RF'], y='mae', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[1, 1])
    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'xgboost'], y='mae', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[1, 2])

    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'k-NN'], y='r2', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[2, 0])
    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'RF'], y='r2', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[2, 1])
    sns.lineplot(data=df_drfp.loc[df_drfp['model'] == 'xgboost'], y='r2', x='nbits', hue="radius", style='radius',
                 markers=True, ax=axes[2, 2])

    [axes[i, j].set_xlabel("") for i in range(3) for j in range(3)]
    [axes[i, j].set_ylabel("") for i in range(3) for j in range(1, 3)]
    [axes[i, j].set_xticks([0, 500, 1000, 1500, 2000]) for i in range(3) for j in range(3)]

    axes[0, 0].set_ylabel('RMSE (kcal/mol)', fontsize=12)
    axes[1, 0].set_ylabel('MAE (kcal/mol)', fontsize=12)
    axes[2, 0].set_ylabel('R$^2$ (kcal/mol)', fontsize=12)
    axes[2, 1].set_xlabel('Dimensionality', fontsize=12)
    axes[0, 0].set_title('model = k-NN', fontsize=12)
    axes[0, 1].set_title('model = RF', fontsize=12)
    axes[0, 2].set_title('model = XGBoost', fontsize=12)

    plt.suptitle('DRFP Fingerprints')
    plt.savefig('drfp_fps.png')


if __name__ == '__main__':
    # Example usage:
    # Replace 'your_file.csv' with the actual CSV file name and 'your_column' with the column you want to plot
    plot_energy_distribution('../final_overview_data.csv', 'Std_DFT_forward')
    scatter_plot('../final_overview_data.csv', 'Std_DFT_forward', 'Std_DFT_reverse')
    line_plot('output.log')
