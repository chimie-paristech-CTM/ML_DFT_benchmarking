import pandas as pd
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    # Example usage:
    # Replace 'your_file.csv' with the actual CSV file name and 'your_column' with the column you want to plot
    plot_energy_distribution('../final_overview_data.csv', 'Std_DFT_forward')
    scatter_plot('../final_overview_data.csv', 'Std_DFT_forward', 'Std_DFT_reverse')
