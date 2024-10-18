"""Comapare the distributions of the original data with gaps before and
after being filled."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'

from scipy.stats import ks_2samp

# Read the data
station = 900
data_900_gaps = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
data_900_filled = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 901
data_901_gaps = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
data_901_filled = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 905
data_905_gaps = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
data_905_filled = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 906
data_906_gaps = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
data_906_filled = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 907
data_907_gaps = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
data_907_filled = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

#%% Get the percentage of missing values for each variable and station
missing_am_900 = data_900_gaps.ammonium_900.isna().sum() / data_900_gaps.shape[0]
missing_co_900 = data_900_gaps.conductivity_900.isna().sum() / data_900_gaps.shape[0]
missing_do_900 = data_900_gaps.dissolved_oxygen_900.isna().sum() / data_900_gaps.shape[0]
missing_ph_900 = data_900_gaps.pH_900.isna().sum() / data_900_gaps.shape[0]
missing_tu_900 = data_900_gaps.turbidity_900.isna().sum() / data_900_gaps.shape[0]
missing_wt_900 = data_900_gaps.water_temperature_900.isna().sum() / data_900_gaps.shape[0]

missing_am_901 = data_901_gaps.ammonium_901.isna().sum() / data_901_gaps.shape[0]
missing_co_901 = data_901_gaps.conductivity_901.isna().sum() / data_901_gaps.shape[0]
missing_do_901 = data_901_gaps.dissolved_oxygen_901.isna().sum() / data_901_gaps.shape[0]
missing_ph_901 = data_901_gaps.pH_901.isna().sum() / data_901_gaps.shape[0]
missing_tu_901 = data_901_gaps.turbidity_901.isna().sum() / data_901_gaps.shape[0]
missing_wt_901 = data_901_gaps.water_temperature_901.isna().sum() / data_901_gaps.shape[0]

missing_am_905 = data_905_gaps.ammonium_905.isna().sum() / data_905_gaps.shape[0]
missing_co_905 = data_905_gaps.conductivity_905.isna().sum() / data_905_gaps.shape[0]
missing_do_905 = data_905_gaps.dissolved_oxygen_905.isna().sum() / data_905_gaps.shape[0]
missing_ph_905 = data_905_gaps.pH_905.isna().sum() / data_905_gaps.shape[0]
missing_tu_905 = data_905_gaps.turbidity_905.isna().sum() / data_905_gaps.shape[0]
missing_wt_905 = data_905_gaps.water_temperature_905.isna().sum() / data_905_gaps.shape[0]

missing_am_906 = data_906_gaps.ammonium_906.isna().sum() / data_906_gaps.shape[0]
missing_co_906 = data_906_gaps.conductivity_906.isna().sum() / data_906_gaps.shape[0]
missing_do_906 = data_906_gaps.dissolved_oxygen_906.isna().sum() / data_906_gaps.shape[0]
missing_ph_906 = data_906_gaps.pH_906.isna().sum() / data_906_gaps.shape[0]
missing_tu_906 = data_906_gaps.turbidity_906.isna().sum() / data_906_gaps.shape[0]
missing_wt_906 = data_906_gaps.water_temperature_906.isna().sum() / data_906_gaps.shape[0]

missing_am_907 = data_907_gaps.ammonium_907.isna().sum() / data_907_gaps.shape[0]
missing_co_907 = data_907_gaps.conductivity_907.isna().sum() / data_907_gaps.shape[0]
missing_do_907 = data_907_gaps.dissolved_oxygen_907.isna().sum() / data_907_gaps.shape[0]
missing_ph_907 = data_907_gaps.pH_907.isna().sum() / data_907_gaps.shape[0]
missing_tu_907 = data_907_gaps.turbidity_907.isna().sum() / data_907_gaps.shape[0]
missing_wt_907 = data_907_gaps.water_temperature_907.isna().sum() / data_907_gaps.shape[0]

# Store the results in a DataFrame
missing_values = pd.DataFrame({
    'station': [900, 900, 900, 900, 900, 900, 901, 901, 901, 901, 901, 901, 905, 905, 905, 905, 905, 905, 906, 906, 906, 906, 906, 906, 907, 907, 907, 907, 907, 907],
    'variable': ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature'],
    'missing_values': [missing_am_900, missing_co_900, missing_do_900, missing_ph_900, missing_tu_900, missing_wt_900,
                    missing_am_901, missing_co_901, missing_do_901, missing_ph_901, missing_tu_901, missing_wt_901,
                    missing_am_905, missing_co_905, missing_do_905, missing_ph_905, missing_tu_905, missing_wt_905,
                    missing_am_906, missing_co_906, missing_do_906, missing_ph_906, missing_tu_906, missing_wt_906,
                    missing_am_907, missing_co_907, missing_do_907, missing_ph_907, missing_tu_907, missing_wt_907]
    })

print(missing_values)

#%% Extract the data for each variable and station before and after filling. Remove the NaNs for the original data
am_900_gaps = data_900_gaps.ammonium_900.to_numpy()
am_900_gaps = am_900_gaps[~np.isnan(am_900_gaps)]
co_900_gaps = data_900_gaps.conductivity_900.to_numpy()
co_900_gaps = co_900_gaps[~np.isnan(co_900_gaps)]
do_900_gaps = data_900_gaps.dissolved_oxygen_900.to_numpy()
do_900_gaps = do_900_gaps[~np.isnan(do_900_gaps)]
ph_900_gaps = data_900_gaps.pH_900.to_numpy()
ph_900_gaps = ph_900_gaps[~np.isnan(ph_900_gaps)]
tu_900_gaps = data_900_gaps.turbidity_900.to_numpy()
tu_900_gaps = tu_900_gaps[~np.isnan(tu_900_gaps)]
wt_900_gaps = data_900_gaps.water_temperature_900.to_numpy()
wt_900_gaps = wt_900_gaps[~np.isnan(wt_900_gaps)]

am_900_filled = data_900_filled.ammonium_900.to_numpy()
am_900_filled = am_900_filled[~np.isnan(am_900_filled)]
co_900_filled = data_900_filled.conductivity_900.to_numpy()
co_900_filled = co_900_filled[~np.isnan(co_900_filled)]
do_900_filled = data_900_filled.dissolved_oxygen_900.to_numpy()
do_900_filled = do_900_filled[~np.isnan(do_900_filled)]
ph_900_filled = data_900_filled.pH_900.to_numpy()
ph_900_filled = ph_900_filled[~np.isnan(ph_900_filled)]
tu_900_filled = data_900_filled.turbidity_900.to_numpy()
tu_900_filled = tu_900_filled[~np.isnan(tu_900_filled)]
wt_900_filled = data_900_filled.water_temperature_900.to_numpy()
wt_900_filled = wt_900_filled[~np.isnan(wt_900_filled)]

am_901_gaps = data_901_gaps.ammonium_901.to_numpy()
am_901_gaps = am_901_gaps[~np.isnan(am_901_gaps)]
co_901_gaps = data_901_gaps.conductivity_901.to_numpy()
co_901_gaps = co_901_gaps[~np.isnan(co_901_gaps)]
do_901_gaps = data_901_gaps.dissolved_oxygen_901.to_numpy()
do_901_gaps = do_901_gaps[~np.isnan(do_901_gaps)]
ph_901_gaps = data_901_gaps.pH_901.to_numpy()
ph_901_gaps = ph_901_gaps[~np.isnan(ph_901_gaps)]
tu_901_gaps = data_901_gaps.turbidity_901.to_numpy()
tu_901_gaps = tu_901_gaps[~np.isnan(tu_901_gaps)]
wt_901_gaps = data_901_gaps.water_temperature_901.to_numpy()
wt_901_gaps = wt_901_gaps[~np.isnan(wt_901_gaps)]

am_901_filled = data_901_filled.ammonium_901.to_numpy()
am_901_filled = am_901_filled[~np.isnan(am_901_filled)]
co_901_filled = data_901_filled.conductivity_901.to_numpy()
co_901_filled = co_901_filled[~np.isnan(co_901_filled)]
do_901_filled = data_901_filled.dissolved_oxygen_901.to_numpy()
do_901_filled = do_901_filled[~np.isnan(do_901_filled)]
ph_901_filled = data_901_filled.pH_901.to_numpy()
ph_901_filled = ph_901_filled[~np.isnan(ph_901_filled)]
tu_901_filled = data_901_filled.turbidity_901.to_numpy()
tu_901_filled = tu_901_filled[~np.isnan(tu_901_filled)]
wt_901_filled = data_901_filled.water_temperature_901.to_numpy()
wt_901_filled = wt_901_filled[~np.isnan(wt_901_filled)]

am_905_gaps = data_905_gaps.ammonium_905.to_numpy()
am_905_gaps = am_905_gaps[~np.isnan(am_905_gaps)]
co_905_gaps = data_905_gaps.conductivity_905.to_numpy()
co_905_gaps = co_905_gaps[~np.isnan(co_905_gaps)]
do_905_gaps = data_905_gaps.dissolved_oxygen_905.to_numpy()
do_905_gaps = do_905_gaps[~np.isnan(do_905_gaps)]
ph_905_gaps = data_905_gaps.pH_905.to_numpy()
ph_905_gaps = ph_905_gaps[~np.isnan(ph_905_gaps)]
tu_905_gaps = data_905_gaps.turbidity_905.to_numpy()
tu_905_gaps = tu_905_gaps[~np.isnan(tu_905_gaps)]
wt_905_gaps = data_905_gaps.water_temperature_905.to_numpy()
wt_905_gaps = wt_905_gaps[~np.isnan(wt_905_gaps)]

am_905_filled = data_905_filled.ammonium_905.to_numpy()
am_905_filled = am_905_filled[~np.isnan(am_905_filled)]
co_905_filled = data_905_filled.conductivity_905.to_numpy()
co_905_filled = co_905_filled[~np.isnan(co_905_filled)]
do_905_filled = data_905_filled.dissolved_oxygen_905.to_numpy()
do_905_filled = do_905_filled[~np.isnan(do_905_filled)]
ph_905_filled = data_905_filled.pH_905.to_numpy()
ph_905_filled = ph_905_filled[~np.isnan(ph_905_filled)]
tu_905_filled = data_905_filled.turbidity_905.to_numpy()
tu_905_filled = tu_905_filled[~np.isnan(tu_905_filled)]
wt_905_filled = data_905_filled.water_temperature_905.to_numpy()
wt_905_filled = wt_905_filled[~np.isnan(wt_905_filled)]

am_906_gaps = data_906_gaps.ammonium_906.to_numpy()
am_906_gaps = am_906_gaps[~np.isnan(am_906_gaps)]
co_906_gaps = data_906_gaps.conductivity_906.to_numpy()
co_906_gaps = co_906_gaps[~np.isnan(co_906_gaps)]
do_906_gaps = data_906_gaps.dissolved_oxygen_906.to_numpy()
do_906_gaps = do_906_gaps[~np.isnan(do_906_gaps)]
ph_906_gaps = data_906_gaps.pH_906.to_numpy()
ph_906_gaps = ph_906_gaps[~np.isnan(ph_906_gaps)]
tu_906_gaps = data_906_gaps.turbidity_906.to_numpy()
tu_906_gaps = tu_906_gaps[~np.isnan(tu_906_gaps)]
wt_906_gaps = data_906_gaps.water_temperature_906.to_numpy()
wt_906_gaps = wt_906_gaps[~np.isnan(wt_906_gaps)]

am_906_filled = data_906_filled.ammonium_906.to_numpy()
am_906_filled = am_906_filled[~np.isnan(am_906_filled)]
co_906_filled = data_906_filled.conductivity_906.to_numpy()
co_906_filled = co_906_filled[~np.isnan(co_906_filled)]
do_906_filled = data_906_filled.dissolved_oxygen_906.to_numpy()
do_906_filled = do_906_filled[~np.isnan(do_906_filled)]
ph_906_filled = data_906_filled.pH_906.to_numpy()
ph_906_filled = ph_906_filled[~np.isnan(ph_906_filled)]
tu_906_filled = data_906_filled.turbidity_906.to_numpy()
tu_906_filled = tu_906_filled[~np.isnan(tu_906_filled)]
wt_906_filled = data_906_filled.water_temperature_906.to_numpy()
wt_906_filled = wt_906_filled[~np.isnan(wt_906_filled)]

am_907_gaps = data_907_gaps.ammonium_907.to_numpy()
am_907_gaps = am_907_gaps[~np.isnan(am_907_gaps)]
co_907_gaps = data_907_gaps.conductivity_907.to_numpy()
co_907_gaps = co_907_gaps[~np.isnan(co_907_gaps)]
do_907_gaps = data_907_gaps.dissolved_oxygen_907.to_numpy()
do_907_gaps = do_907_gaps[~np.isnan(do_907_gaps)]
ph_907_gaps = data_907_gaps.pH_907.to_numpy()
ph_907_gaps = ph_907_gaps[~np.isnan(ph_907_gaps)]
tu_907_gaps = data_907_gaps.turbidity_907.to_numpy()
tu_907_gaps = tu_907_gaps[~np.isnan(tu_907_gaps)]
wt_907_gaps = data_907_gaps.water_temperature_907.to_numpy()
wt_907_gaps = wt_907_gaps[~np.isnan(wt_907_gaps)]

am_907_filled = data_907_filled.ammonium_907.to_numpy()
am_907_filled = am_907_filled[~np.isnan(am_907_filled)]
co_907_filled = data_907_filled.conductivity_907.to_numpy()
co_907_filled = co_907_filled[~np.isnan(co_907_filled)]
do_907_filled = data_907_filled.dissolved_oxygen_907.to_numpy()
do_907_filled = do_907_filled[~np.isnan(do_907_filled)]
ph_907_filled = data_907_filled.pH_907.to_numpy()
ph_907_filled = ph_907_filled[~np.isnan(ph_907_filled)]
tu_907_filled = data_907_filled.turbidity_907.to_numpy()
tu_907_filled = tu_907_filled[~np.isnan(tu_907_filled)]
wt_907_filled = data_907_filled.water_temperature_907.to_numpy()
wt_907_filled = wt_907_filled[~np.isnan(wt_907_filled)]

#%% Plot the empirical cumulative distributions of each variable comparing the data with gaps and the filled version for each station
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

sns.ecdfplot(data=am_900_gaps, linestyle='dotted', linewidth=5, color='lightcoral', ax=axes[0, 0], label='Original')
sns.ecdfplot(data=am_900_filled, color='red', ax=axes[0, 0], label='Imputed')
axes[0, 0].legend(loc='lower right', fontsize=12)

sns.ecdfplot(data=co_900_gaps, linestyle='dotted', linewidth=5, color='cornflowerblue', ax=axes[0, 1], label='Original')
sns.ecdfplot(data=co_900_filled, color='blue', ax=axes[0, 1], label='Imputed')
axes[0, 1].legend(loc='lower right', fontsize=12)

sns.ecdfplot(data=do_900_gaps, linestyle='dotted', linewidth=5, color='mediumpurple', ax=axes[0, 2], label='Original')
sns.ecdfplot(data=do_900_filled, color='purple', ax=axes[0, 2], label='Imputed')
axes[0, 2].legend(loc='lower right', fontsize=12)

sns.ecdfplot(data=ph_900_gaps, linestyle='dotted', linewidth=5, color='dimgray', ax=axes[1, 0], label='Original')
sns.ecdfplot(data=ph_900_filled, color='darkgray', ax=axes[1, 0], label='Imputed')
axes[1, 0].legend(loc='lower left', fontsize=12)

sns.ecdfplot(data=tu_900_gaps, linestyle='dotted', linewidth=5, color='gold', ax=axes[1, 1], label='Original')
sns.ecdfplot(data=tu_900_filled, color='goldenrod', ax=axes[1, 1], label='Imputed')
axes[1, 1].legend(loc='lower right', fontsize=12)

sns.ecdfplot(data=wt_900_gaps, linestyle='dotted', linewidth=5, color='limegreen', ax=axes[1, 2], label='Original')
sns.ecdfplot(data=wt_900_filled, color='green', ax=axes[1, 2], label='Imputed')
axes[1, 2].legend(loc='lower right', fontsize=12)

# Clean default y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)

# Set the title for each variable
var_names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
for i, ax in enumerate(axes.flat):
    ax.set_title(var_names[i], fontname='Arial', fontsize=18)

# Set the x label for each variable
for ax in axes[1, :]:
    ax.set_xlabel('Value', fontname='Arial', fontsize=16)

# Set the y label for each variable
for ax in axes[:, 0]:
    ax.set_ylabel('Cumulative probability', fontname='Arial', fontsize=16)

fig.suptitle('ECD before and after imputation', fontname='Arial', fontsize=22)
plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig('plots/imputation.pdf', format='pdf', dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

sns.ecdfplot(data=am_901_gaps, linestyle='dotted', linewidth=5, color='lightcoral', ax=axes[0, 0])
sns.ecdfplot(data=am_901_filled, color='red', ax=axes[0, 0])

sns.ecdfplot(data=co_901_gaps, linestyle='dotted', linewidth=5, color='cornflowerblue', ax=axes[0, 1])
sns.ecdfplot(data=co_901_filled, color='blue', ax=axes[0, 1])

sns.ecdfplot(data=do_901_gaps, linestyle='dotted', linewidth=5, color='mediumpurple', ax=axes[0, 2])
sns.ecdfplot(data=do_901_filled, color='purple', ax=axes[0, 2])

sns.ecdfplot(data=ph_901_gaps, linestyle='dotted', linewidth=5, color='dimgray', ax=axes[1, 0])
sns.ecdfplot(data=ph_901_filled, color='darkgray', ax=axes[1, 0])

sns.ecdfplot(data=tu_901_gaps, linestyle='dotted', linewidth=5, color='gold', ax=axes[1, 1])
sns.ecdfplot(data=tu_901_filled, color='goldenrod', ax=axes[1, 1])

sns.ecdfplot(data=wt_901_gaps, linestyle='dotted', linewidth=5, color='limegreen', ax=axes[1, 2])
sns.ecdfplot(data=wt_901_filled, color='green', ax=axes[1, 2])

# Clean default y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)

# Set the title for each variable
var_names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
for i, ax in enumerate(axes.flat):
    ax.set_title(var_names[i], fontname='Arial', fontsize=18)

# Set the x label for each variable
for ax in axes[1, :]:
    ax.set_xlabel('Value', fontname='Arial', fontsize=16)

# Set the y label for each variable
for ax in axes[:, 0]:
    ax.set_ylabel('Cumulative probability', fontname='Arial', fontsize=16)

fig.suptitle('ECD before and after imputation for station 901', fontname='Arial', fontsize=22)
plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig('plots/imputation_901.pdf', format='pdf', dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

sns.ecdfplot(data=am_905_gaps, linestyle='dotted', linewidth=5, color='lightcoral', ax=axes[0, 0])
sns.ecdfplot(data=am_905_filled, color='red', ax=axes[0, 0])

sns.ecdfplot(data=co_905_gaps, linestyle='dotted', linewidth=5, color='cornflowerblue', ax=axes[0, 1])
sns.ecdfplot(data=co_905_filled, color='blue', ax=axes[0, 1])

sns.ecdfplot(data=do_905_gaps, linestyle='dotted', linewidth=5, color='mediumpurple', ax=axes[0, 2])
sns.ecdfplot(data=do_905_filled, color='purple', ax=axes[0, 2])

sns.ecdfplot(data=ph_905_gaps, linestyle='dotted', linewidth=5, color='dimgray', ax=axes[1, 0])
sns.ecdfplot(data=ph_905_filled, color='darkgray', ax=axes[1, 0])

sns.ecdfplot(data=tu_905_gaps, linestyle='dotted', linewidth=5, color='gold', ax=axes[1, 1])
sns.ecdfplot(data=tu_905_filled, color='goldenrod', ax=axes[1, 1])

sns.ecdfplot(data=wt_905_gaps, linestyle='dotted', linewidth=5, color='limegreen', ax=axes[1, 2])
sns.ecdfplot(data=wt_905_filled, color='green', ax=axes[1, 2])

# Clean default y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)

# Set the title for each variable
var_names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
for i, ax in enumerate(axes.flat):
    ax.set_title(var_names[i], fontname='Arial', fontsize=18)

# Set the x label for each variable
for ax in axes[1, :]:
    ax.set_xlabel('Value', fontname='Arial', fontsize=16)

# Set the y label for each variable
for ax in axes[:, 0]:
    ax.set_ylabel('Cumulative probability', fontname='Arial', fontsize=16)

fig.suptitle('ECD before and after imputation for station 905', fontname='Arial', fontsize=22)
plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig('plots/imputation_905.pdf', format='pdf', dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

sns.ecdfplot(data=am_906_gaps, linestyle='dotted', linewidth=5, color='lightcoral', ax=axes[0, 0], label='Original')
sns.ecdfplot(data=am_906_filled, color='red', ax=axes[0, 0], label='Imputed')
axes[0, 0].legend(loc='lower right', fontsize=12)

sns.ecdfplot(data=co_906_gaps, linestyle='dotted', linewidth=5, color='cornflowerblue', ax=axes[0, 1], label='Original')
sns.ecdfplot(data=co_906_filled, color='blue', ax=axes[0, 1], label='Imputed')
axes[0, 1].legend(loc='lower right', fontsize=12)

sns.ecdfplot(data=do_906_gaps, linestyle='dotted', linewidth=5, color='mediumpurple', ax=axes[0, 2], label='Original')
sns.ecdfplot(data=do_906_filled, color='purple', ax=axes[0, 2], label='Imputed')
axes[0, 2].legend(loc='lower right', fontsize=12)

sns.ecdfplot(data=ph_906_gaps, linestyle='dotted', linewidth=5, color='dimgray', ax=axes[1, 0], label='Original')
sns.ecdfplot(data=ph_906_filled, color='darkgray', ax=axes[1, 0], label='Imputed')
axes[1, 0].legend(loc='lower left', fontsize=12)

sns.ecdfplot(data=tu_906_gaps, linestyle='dotted', linewidth=5, color='gold', ax=axes[1, 1], label='Original')
sns.ecdfplot(data=tu_906_filled, color='goldenrod', ax=axes[1, 1], label='Imputed')
axes[1, 1].legend(loc='lower right', fontsize=12)

sns.ecdfplot(data=wt_906_gaps, linestyle='dotted', linewidth=5, color='limegreen', ax=axes[1, 2], label='Original')
sns.ecdfplot(data=wt_906_filled, color='green', ax=axes[1, 2], label='Imputed')
axes[1, 2].legend(loc='lower right', fontsize=12)

# Clean default y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)

# Set the title for each variable
var_names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
for i, ax in enumerate(axes.flat):
    ax.set_title(var_names[i], fontname='Arial', fontsize=18)

# Set the x label for each variable
for ax in axes[1, :]:
    ax.set_xlabel('Value', fontname='Arial', fontsize=16)

# Set the y label for each variable
for ax in axes[:, 0]:
    ax.set_ylabel('Cumulative probability', fontname='Arial', fontsize=16)

fig.suptitle('ECD before and after imputation', fontname='Arial', fontsize=22)
plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig('plots/imputation_906.pdf', format='pdf', dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

sns.ecdfplot(data=am_907_gaps, linestyle='dotted', linewidth=5, color='lightcoral', ax=axes[0, 0])
sns.ecdfplot(data=am_907_filled, color='red', ax=axes[0, 0])

sns.ecdfplot(data=co_907_gaps, linestyle='dotted', linewidth=5, color='cornflowerblue', ax=axes[0, 1])
sns.ecdfplot(data=co_907_filled, color='blue', ax=axes[0, 1])

sns.ecdfplot(data=do_907_gaps, color='mediumpurple', ax=axes[0, 2])
sns.ecdfplot(data=do_907_filled, color='purple', ax=axes[0, 2])

sns.ecdfplot(data=ph_907_gaps, linestyle='dotted', linewidth=5, color='dimgray', ax=axes[1, 0])
sns.ecdfplot(data=ph_907_filled, color='darkgray', ax=axes[1, 0])

sns.ecdfplot(data=tu_907_gaps, linestyle='dotted', linewidth=5, color='gold', ax=axes[1, 1])
sns.ecdfplot(data=tu_907_filled, color='goldenrod', ax=axes[1, 1])

sns.ecdfplot(data=wt_907_gaps, linestyle='dotted', linewidth=5, color='limegreen', ax=axes[1, 2])
sns.ecdfplot(data=wt_907_filled, color='green', ax=axes[1, 2])

# Clean default y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)

# Set the title for each variable
var_names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
for i, ax in enumerate(axes.flat):
    ax.set_title(var_names[i], fontname='Arial', fontsize=18)

# Set the x label for each variable
for ax in axes[1, :]:
    ax.set_xlabel('Value', fontname='Arial', fontsize=16)

# Set the y label for each variable
for ax in axes[:, 0]:
    ax.set_ylabel('Cumulative probability', fontname='Arial', fontsize=16)

fig.suptitle('ECD before and after imputation for station 907', fontname='Arial', fontsize=22)
plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig('plots/imputation_907.pdf', format='pdf', dpi=300, bbox_inches='tight')

#%% Get the mean and standard deviation for each variable and station before and after filling
mean_900_am_gaps, std_900_am_gaps, mean_900_am_filled, std_900_am_filled = np.mean(am_900_gaps), np.std(am_900_gaps), np.mean(am_900_filled), np.std(am_900_filled)
mean_900_co_gaps, std_900_co_gaps, mean_900_co_filled, std_900_co_filled = np.mean(co_900_gaps), np.std(co_900_gaps), np.mean(co_900_filled), np.std(co_900_filled)
mean_900_do_gaps, std_900_do_gaps, mean_900_do_filled, std_900_do_filled = np.mean(do_900_gaps), np.std(do_900_gaps), np.mean(do_900_filled), np.std(do_900_filled)
mean_900_ph_gaps, std_900_ph_gaps, mean_900_ph_filled, std_900_ph_filled = np.mean(ph_900_gaps), np.std(ph_900_gaps), np.mean(ph_900_filled), np.std(ph_900_filled)
mean_900_tu_gaps, std_900_tu_gaps, mean_900_tu_filled, std_900_tu_filled = np.mean(tu_900_gaps), np.std(tu_900_gaps), np.mean(tu_900_filled), np.std(tu_900_filled)
mean_900_wt_gaps, std_900_wt_gaps, mean_900_wt_filled, std_900_wt_filled = np.mean(wt_900_gaps), np.std(wt_900_gaps), np.mean(wt_900_filled), np.std(wt_900_filled)

mean_901_am_gaps, std_901_am_gaps, mean_901_am_filled, std_901_am_filled = np.mean(am_901_gaps), np.std(am_901_gaps), np.mean(am_901_filled), np.std(am_901_filled)
mean_901_co_gaps, std_901_co_gaps, mean_901_co_filled, std_901_co_filled = np.mean(co_901_gaps), np.std(co_901_gaps), np.mean(co_901_filled), np.std(co_901_filled)
mean_901_do_gaps, std_901_do_gaps, mean_901_do_filled, std_901_do_filled = np.mean(do_901_gaps), np.std(do_901_gaps), np.mean(do_901_filled), np.std(do_901_filled)
mean_901_ph_gaps, std_901_ph_gaps, mean_901_ph_filled, std_901_ph_filled = np.mean(ph_901_gaps), np.std(ph_901_gaps), np.mean(ph_901_filled), np.std(ph_901_filled)
mean_901_tu_gaps, std_901_tu_gaps, mean_901_tu_filled, std_901_tu_filled = np.mean(tu_901_gaps), np.std(tu_901_gaps), np.mean(tu_901_filled), np.std(tu_901_filled)
mean_901_wt_gaps, std_901_wt_gaps, mean_901_wt_filled, std_901_wt_filled = np.mean(wt_901_gaps), np.std(wt_901_gaps), np.mean(wt_901_filled), np.std(wt_901_filled)

mean_905_am_gaps, std_905_am_gaps, mean_905_am_filled, std_905_am_filled = np.mean(am_905_gaps), np.std(am_905_gaps), np.mean(am_905_filled), np.std(am_905_filled)
mean_905_co_gaps, std_905_co_gaps, mean_905_co_filled, std_905_co_filled = np.mean(co_905_gaps), np.std(co_905_gaps), np.mean(co_905_filled), np.std(co_905_filled)
mean_905_do_gaps, std_905_do_gaps, mean_905_do_filled, std_905_do_filled = np.mean(do_905_gaps), np.std(do_905_gaps), np.mean(do_905_filled), np.std(do_905_filled)
mean_905_ph_gaps, std_905_ph_gaps, mean_905_ph_filled, std_905_ph_filled = np.mean(ph_905_gaps), np.std(ph_905_gaps), np.mean(ph_905_filled), np.std(ph_905_filled)
mean_905_tu_gaps, std_905_tu_gaps, mean_905_tu_filled, std_905_tu_filled = np.mean(tu_905_gaps), np.std(tu_905_gaps), np.mean(tu_905_filled), np.std(tu_905_filled)
mean_905_wt_gaps, std_905_wt_gaps, mean_905_wt_filled, std_905_wt_filled = np.mean(wt_905_gaps), np.std(wt_905_gaps), np.mean(wt_905_filled), np.std(wt_905_filled)

mean_906_am_gaps, std_906_am_gaps, mean_906_am_filled, std_906_am_filled = np.mean(am_906_gaps), np.std(am_906_gaps), np.mean(am_906_filled), np.std(am_906_filled)
mean_906_co_gaps, std_906_co_gaps, mean_906_co_filled, std_906_co_filled = np.mean(co_906_gaps), np.std(co_906_gaps), np.mean(co_906_filled), np.std(co_906_filled)
mean_906_do_gaps, std_906_do_gaps, mean_906_do_filled, std_906_do_filled = np.mean(do_906_gaps), np.std(do_906_gaps), np.mean(do_906_filled), np.std(do_906_filled)
mean_906_ph_gaps, std_906_ph_gaps, mean_906_ph_filled, std_906_ph_filled = np.mean(ph_906_gaps), np.std(ph_906_gaps), np.mean(ph_906_filled), np.std(ph_906_filled)
mean_906_tu_gaps, std_906_tu_gaps, mean_906_tu_filled, std_906_tu_filled = np.mean(tu_906_gaps), np.std(tu_906_gaps), np.mean(tu_906_filled), np.std(tu_906_filled)
mean_906_wt_gaps, std_906_wt_gaps, mean_906_wt_filled, std_906_wt_filled = np.mean(wt_906_gaps), np.std(wt_906_gaps), np.mean(wt_906_filled), np.std(wt_906_filled)

mean_907_am_gaps, std_907_am_gaps, mean_907_am_filled, std_907_am_filled = np.mean(am_907_gaps), np.std(am_907_gaps), np.mean(am_907_filled), np.std(am_907_filled)
mean_907_co_gaps, std_907_co_gaps, mean_907_co_filled, std_907_co_filled = np.mean(co_907_gaps), np.std(co_907_gaps), np.mean(co_907_filled), np.std(co_907_filled)
mean_907_do_gaps, std_907_do_gaps, mean_907_do_filled, std_907_do_filled = np.mean(do_907_gaps), np.std(do_907_gaps), np.mean(do_907_filled), np.std(do_907_filled)
mean_907_ph_gaps, std_907_ph_gaps, mean_907_ph_filled, std_907_ph_filled = np.mean(ph_907_gaps), np.std(ph_907_gaps), np.mean(ph_907_filled), np.std(ph_907_filled)
mean_907_tu_gaps, std_907_tu_gaps, mean_907_tu_filled, std_907_tu_filled = np.mean(tu_907_gaps), np.std(tu_907_gaps), np.mean(tu_907_filled), np.std(tu_907_filled)
mean_907_wt_gaps, std_907_wt_gaps, mean_907_wt_filled, std_907_wt_filled = np.mean(wt_907_gaps), np.std(wt_907_gaps), np.mean(wt_907_filled), np.std(wt_907_filled)

# Store the results in a DataFrame
results = pd.DataFrame({
    'station': [900, 900, 900, 900, 900, 900, 901, 901, 901, 901, 901, 901, 905, 905, 905, 905, 905, 905, 906, 906, 906, 906, 906, 906, 907, 907, 907, 907, 907, 907],
    'variable': ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature'],
    'mean_gaps': [mean_900_am_gaps, mean_900_co_gaps, mean_900_do_gaps, mean_900_ph_gaps, mean_900_tu_gaps, mean_900_wt_gaps,
                mean_901_am_gaps, mean_901_co_gaps, mean_901_do_gaps, mean_901_ph_gaps, mean_901_tu_gaps, mean_901_wt_gaps,
                mean_905_am_gaps, mean_905_co_gaps, mean_905_do_gaps, mean_905_ph_gaps, mean_905_tu_gaps, mean_905_wt_gaps,
                mean_906_am_gaps, mean_906_co_gaps, mean_906_do_gaps, mean_906_ph_gaps, mean_906_tu_gaps, mean_906_wt_gaps,
                mean_907_am_gaps, mean_907_co_gaps, mean_907_do_gaps, mean_907_ph_gaps, mean_907_tu_gaps, mean_907_wt_gaps],
    'std_gaps': [std_900_am_gaps, std_900_co_gaps, std_900_do_gaps, std_900_ph_gaps, std_900_tu_gaps, std_900_wt_gaps,
                std_901_am_gaps, std_901_co_gaps, std_901_do_gaps, std_901_ph_gaps, std_901_tu_gaps, std_901_wt_gaps,
                std_905_am_gaps, std_905_co_gaps, std_905_do_gaps, std_905_ph_gaps, std_905_tu_gaps, std_905_wt_gaps,
                std_906_am_gaps, std_906_co_gaps, std_906_do_gaps, std_906_ph_gaps, std_906_tu_gaps, std_906_wt_gaps,
                std_907_am_gaps, std_907_co_gaps, std_907_do_gaps, std_907_ph_gaps, std_907_tu_gaps, std_907_wt_gaps],
    'mean_filled': [mean_900_am_filled, mean_900_co_filled, mean_900_do_filled, mean_900_ph_filled, mean_900_tu_filled, mean_900_wt_filled,
                mean_901_am_filled, mean_901_co_filled, mean_901_do_filled, mean_901_ph_filled, mean_901_tu_filled, mean_901_wt_filled,
                mean_905_am_filled, mean_905_co_filled, mean_905_do_filled, mean_905_ph_filled, mean_905_tu_filled, mean_905_wt_filled,
                mean_906_am_filled, mean_906_co_filled, mean_906_do_filled, mean_906_ph_filled, mean_906_tu_filled, mean_906_wt_filled,
                mean_907_am_filled, mean_907_co_filled, mean_907_do_filled, mean_907_ph_filled, mean_907_tu_filled, mean_907_wt_filled],
    'std_filled': [std_900_am_filled, std_900_co_filled, std_900_do_filled, std_900_ph_filled, std_900_tu_filled, std_900_wt_filled,
                std_901_am_filled, std_901_co_filled, std_901_do_filled, std_901_ph_filled, std_901_tu_filled, std_901_wt_filled,
                std_905_am_filled, std_905_co_filled, std_905_do_filled, std_905_ph_filled, std_905_tu_filled, std_905_wt_filled,
                std_906_am_filled, std_906_co_filled, std_906_do_filled, std_906_ph_filled, std_906_tu_filled, std_906_wt_filled,
                std_907_am_filled, std_907_co_filled, std_907_do_filled, std_907_ph_filled, std_907_tu_filled, std_907_wt_filled]
    })

print(results)

#%% Getthe percent difference between the mean and standard deviation of the original and filled data
percent_diff_mean_900_am = (mean_900_am_filled - mean_900_am_gaps) / mean_900_am_gaps * 100
percent_diff_mean_900_co = (mean_900_co_filled - mean_900_co_gaps) / mean_900_co_gaps * 100
percent_diff_mean_900_do = (mean_900_do_filled - mean_900_do_gaps) / mean_900_do_gaps * 100
percent_diff_mean_900_ph = (mean_900_ph_filled - mean_900_ph_gaps) / mean_900_ph_gaps * 100
percent_diff_mean_900_tu = (mean_900_tu_filled - mean_900_tu_gaps) / mean_900_tu_gaps * 100
percent_diff_mean_900_wt = (mean_900_wt_filled - mean_900_wt_gaps) / mean_900_wt_gaps * 100

percent_diff_mean_901_am = (mean_901_am_filled - mean_901_am_gaps) / mean_901_am_gaps * 100
percent_diff_mean_901_co = (mean_901_co_filled - mean_901_co_gaps) / mean_901_co_gaps * 100
percent_diff_mean_901_do = (mean_901_do_filled - mean_901_do_gaps) / mean_901_do_gaps * 100
percent_diff_mean_901_ph = (mean_901_ph_filled - mean_901_ph_gaps) / mean_901_ph_gaps * 100
percent_diff_mean_901_tu = (mean_901_tu_filled - mean_901_tu_gaps) / mean_901_tu_gaps * 100
percent_diff_mean_901_wt = (mean_901_wt_filled - mean_901_wt_gaps) / mean_901_wt_gaps * 100

percent_diff_mean_905_am = (mean_905_am_filled - mean_905_am_gaps) / mean_905_am_gaps * 100
percent_diff_mean_905_co = (mean_905_co_filled - mean_905_co_gaps) / mean_905_co_gaps * 100
percent_diff_mean_905_do = (mean_905_do_filled - mean_905_do_gaps) / mean_905_do_gaps * 100
percent_diff_mean_905_ph = (mean_905_ph_filled - mean_905_ph_gaps) / mean_905_ph_gaps * 100
percent_diff_mean_905_tu = (mean_905_tu_filled - mean_905_tu_gaps) / mean_905_tu_gaps * 100
percent_diff_mean_905_wt = (mean_905_wt_filled - mean_905_wt_gaps) / mean_905_wt_gaps * 100

percent_diff_mean_907_am = (mean_907_am_filled - mean_907_am_gaps) / mean_907_am_gaps * 100
percent_diff_mean_907_co = (mean_907_co_filled - mean_907_co_gaps) / mean_907_co_gaps * 100
percent_diff_mean_907_do = (mean_907_do_filled - mean_907_do_gaps) / mean_907_do_gaps * 100
percent_diff_mean_907_ph = (mean_907_ph_filled - mean_907_ph_gaps) / mean_907_ph_gaps * 100
percent_diff_mean_907_tu = (mean_907_tu_filled - mean_907_tu_gaps) / mean_907_tu_gaps * 100
percent_diff_mean_907_wt = (mean_907_wt_filled - mean_907_wt_gaps) / mean_907_wt_gaps * 100

# Store the results in a DataFrame
percent_diff_mean = pd.DataFrame({
    'station': [900, 900, 900, 900, 900, 900, 901, 901, 901, 901, 901, 901, 905, 905, 905, 905, 905, 905, 907, 907, 907, 907, 907, 907],
    'variable': ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature'],
    'percent_diff_mean': [percent_diff_mean_900_am, percent_diff_mean_900_co, percent_diff_mean_900_do, percent_diff_mean_900_ph, percent_diff_mean_900_tu, percent_diff_mean_900_wt,
                        percent_diff_mean_901_am, percent_diff_mean_901_co, percent_diff_mean_901_do, percent_diff_mean_901_ph, percent_diff_mean_901_tu, percent_diff_mean_901_wt,
                        percent_diff_mean_905_am, percent_diff_mean_905_co, percent_diff_mean_905_do, percent_diff_mean_905_ph, percent_diff_mean_905_tu, percent_diff_mean_905_wt,
                        percent_diff_mean_907_am, percent_diff_mean_907_co, percent_diff_mean_907_do, percent_diff_mean_907_ph, percent_diff_mean_907_tu, percent_diff_mean_907_wt]
    })

print(percent_diff_mean)

#%% Test if the original and filled data come from the same distribution using the Kolmogorov-Smirnov test
# stat_900_am, p_900_am = ks_2samp(am_900_gaps[:500], am_900_filled[:500])
# stat_900_co, p_900_co = ks_2samp(co_900_gaps, co_900_filled)
# stat_900_do, p_900_do = ks_2samp(do_900_gaps, do_900_filled)
# stat_900_ph, p_900_ph = ks_2samp(ph_900_gaps, ph_900_filled)
# stat_900_tu, p_900_tu = ks_2samp(tu_900_gaps, tu_900_filled)
# stat_900_wt, p_900_wt = ks_2samp(wt_900_gaps, wt_900_filled)

# stat_901_am, p_901_am = ks_2samp(am_901_gaps[:500], am_901_filled[:500])
# stat_901_co, p_901_co = ks_2samp(co_901_gaps, co_901_filled)
# stat_901_do, p_901_do = ks_2samp(do_901_gaps, do_901_filled)
# stat_901_ph, p_901_ph = ks_2samp(ph_901_gaps, ph_901_filled)
# stat_901_tu, p_901_tu = ks_2samp(tu_901_gaps, tu_901_filled)
# stat_901_wt, p_901_wt = ks_2samp(wt_901_gaps, wt_901_filled)

# stat_905_am, p_905_am = ks_2samp(am_905_gaps, am_905_filled)
# stat_905_co, p_905_co = ks_2samp(co_905_gaps, co_905_filled)
# stat_905_do, p_905_do = ks_2samp(do_905_gaps, do_905_filled)
# stat_905_ph, p_905_ph = ks_2samp(ph_905_gaps, ph_905_filled)
# stat_905_tu, p_905_tu = ks_2samp(tu_905_gaps, tu_905_filled)
# stat_905_wt, p_905_wt = ks_2samp(wt_905_gaps, wt_905_filled)

# stat_906_am, p_906_am = ks_2samp(am_906_gaps, am_906_filled)
# stat_906_co, p_906_co = ks_2samp(co_906_gaps, co_906_filled)
# stat_906_do, p_906_do = ks_2samp(do_906_gaps, do_906_filled)
# stat_906_ph, p_906_ph = ks_2samp(ph_906_gaps, ph_906_filled)
# stat_906_tu, p_906_tu = ks_2samp(tu_906_gaps, tu_906_filled)
# stat_906_wt, p_906_wt = ks_2samp(wt_906_gaps, wt_906_filled)

# stat_907_am, p_907_am = ks_2samp(am_907_gaps, am_907_filled)
# stat_907_co, p_907_co = ks_2samp(co_907_gaps, co_907_filled)
# stat_907_do, p_907_do = ks_2samp(do_907_gaps, do_907_filled)
# stat_907_ph, p_907_ph = ks_2samp(ph_907_gaps, ph_907_filled)
# stat_907_tu, p_907_tu = ks_2samp(tu_907_gaps, tu_907_filled)
# stat_907_wt, p_907_wt = ks_2samp(wt_907_gaps, wt_907_filled)

# # Store the results in a DataFrame
# results = pd.DataFrame({
#     'station': [900, 900, 900, 900, 900, 900, 901, 901, 901, 901, 901, 901, 905, 905, 905, 905, 905, 905, 906, 906, 906, 906, 906, 906, 907, 907, 907, 907, 907, 907],
#     'variable': ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
#                 'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
#                 'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
#                 'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
#                 'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature'],
#     'statistic': [stat_900_am, stat_900_co, stat_900_do, stat_900_ph, stat_900_tu, stat_900_wt,
#                 stat_901_am, stat_901_co, stat_901_do, stat_901_ph, stat_901_tu, stat_901_wt,
#                 stat_905_am, stat_905_co, stat_905_do, stat_905_ph, stat_905_tu, stat_905_wt,
#                 stat_906_am, stat_906_co, stat_906_do, stat_906_ph, stat_906_tu, stat_906_wt,
#                 stat_907_am, stat_907_co, stat_907_do, stat_907_ph, stat_907_tu, stat_907_wt],
#     'p-value': [p_900_am, p_900_co, p_900_do, p_900_ph, p_900_tu, p_900_wt,
#                 p_901_am, p_901_co, p_901_do, p_901_ph, p_901_tu, p_901_wt,
#                 p_905_am, p_905_co, p_905_do, p_905_ph, p_905_tu, p_905_wt,
#                 p_906_am, p_906_co, p_906_do, p_906_ph, p_906_tu, p_906_wt,
#                 p_907_am, p_907_co, p_907_do, p_907_ph, p_907_tu, p_907_wt]
# })

# print(results)

# NOTE: The p-value is the probability of observing the given statistic if the null hypothesis is true. If the p-value is less than the significance level (0.05), we reject the null hypothesis. In this case, the null hypothesis is that the two samples come from the same distribution. If the p-value is greater than 0.05, we fail to reject the null hypothesis.
# In other words, if p-value < 0.05, we can conclude that the original and filled data come from different distributions. If p-value > 0.05, we can conclude that the original and filled data come from the same distribution.
