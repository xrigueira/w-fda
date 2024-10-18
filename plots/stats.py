import pandas as pd

# Get the min, q1, median, q3, max std, and variance of each variable in each dataset

# Define results dataframe
results = pd.DataFrame(columns=['column', 'min', 'q1', 'mean', 'q3', 'max', 'std', 'var'])

# Define the stations
stations = [900, 901, 906, 905, 907]

for station in stations:

    # Read the dataset
    data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

    # Get the min, q1, mean, q3, max std, and variance of each variable
    columns = data.columns[1:-2]
    for column in columns:
        minimm = data[column].min()
        if minimm < 0:
            minimm = 0
        results.loc[len(results.index)] = [column, minimm, data[column].quantile(0.25), data[column].mean(), data[column].quantile(0.75), data[column].max(), data[column].std(), data[column].var()]

# Save the results
results.to_csv(f'plots/stats.csv', sep=',', encoding='utf-8', index=True)
