import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'

# Read the data
station = 901
data_901 = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 905
data_905 = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 906
data_906 = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 907
data_907 = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

# Subset the anomalies
data_anomalies_901 = data_901[data_901['label'] == 1]
data_anomalies_905 = data_905[data_905['label'] == 1]
data_anomalies_906 = data_906[data_906['label'] == 1]
data_anomalies_907 = data_907[data_907['label'] == 1]

# Subset the normal data
data_normal_901 = data_901[data_901['label'] == 0]
data_normal_905 = data_905[data_905['label'] == 0]
data_normal_906 = data_906[data_906['label'] == 0]
data_normal_907 = data_907[data_907['label'] == 0]

# Extract the anomalous data for each variable and station
am_901_anomalies = data_anomalies_901.ammonium_901.to_numpy()
co_901_anomalies = data_anomalies_901.conductivity_901.to_numpy()
do_901_anomalies = data_anomalies_901.dissolved_oxygen_901.to_numpy()
ph_901_anomalies = data_anomalies_901.pH_901.to_numpy()
tu_901_anomalies = data_anomalies_901.turbidity_901.to_numpy()
wt_901_anomalies = data_anomalies_901.water_temperature_901.to_numpy()

am_905_anomalies = data_anomalies_905.ammonium_905.to_numpy()
co_905_anomalies = data_anomalies_905.conductivity_905.to_numpy()
do_905_anomalies = data_anomalies_905.dissolved_oxygen_905.to_numpy()
ph_905_anomalies = data_anomalies_905.pH_905.to_numpy()
tu_905_anomalies = data_anomalies_905.turbidity_905.to_numpy()
wt_905_anomalies = data_anomalies_905.water_temperature_905.to_numpy()

am_906_anomalies = data_anomalies_906.ammonium_906.to_numpy()
co_906_anomalies = data_anomalies_906.conductivity_906.to_numpy()
do_906_anomalies = data_anomalies_906.dissolved_oxygen_906.to_numpy()
ph_906_anomalies = data_anomalies_906.pH_906.to_numpy()
tu_906_anomalies = data_anomalies_906.turbidity_906.to_numpy()
wt_906_anomalies = data_anomalies_906.water_temperature_906.to_numpy()

am_907_anomalies = data_anomalies_907.ammonium_907.to_numpy()
co_907_anomalies = data_anomalies_907.conductivity_907.to_numpy()
do_907_anomalies = data_anomalies_907.dissolved_oxygen_907.to_numpy()
ph_907_anomalies = data_anomalies_907.pH_907.to_numpy()
tu_907_anomalies = data_anomalies_907.turbidity_907.to_numpy()
wt_907_anomalies = data_anomalies_907.water_temperature_907.to_numpy()

# Extract the normal data for each variable and station
am_901_background = data_normal_901.ammonium_901.to_numpy()
co_901_background = data_normal_901.conductivity_901.to_numpy()
do_901_background = data_normal_901.dissolved_oxygen_901.to_numpy()
ph_901_background = data_normal_901.pH_901.to_numpy()
tu_901_background = data_normal_901.turbidity_901.to_numpy()
wt_901_background = data_normal_901.water_temperature_901.to_numpy()

am_905_background = data_normal_905.ammonium_905.to_numpy()
co_905_background = data_normal_905.conductivity_905.to_numpy()
do_905_background = data_normal_905.dissolved_oxygen_905.to_numpy()
ph_905_background = data_normal_905.pH_905.to_numpy()
tu_905_background = data_normal_905.turbidity_905.to_numpy()
wt_905_background = data_normal_905.water_temperature_905.to_numpy()

am_906_background = data_normal_906.ammonium_906.to_numpy()
co_906_background = data_normal_906.conductivity_906.to_numpy()
do_906_background = data_normal_906.dissolved_oxygen_906.to_numpy()
ph_906_background = data_normal_906.pH_906.to_numpy()
tu_906_background = data_normal_906.turbidity_906.to_numpy()
wt_906_background = data_normal_906.water_temperature_906.to_numpy()

am_907_background = data_normal_907.ammonium_907.to_numpy()
co_907_background = data_normal_907.conductivity_907.to_numpy()
do_907_background = data_normal_907.dissolved_oxygen_907.to_numpy()
ph_907_background = data_normal_907.pH_907.to_numpy()
tu_907_background = data_normal_907.turbidity_907.to_numpy()
wt_907_background = data_normal_907.water_temperature_907.to_numpy()

# Plot the distribution of the each variable for the ground truth and each model's prediction
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(18, 10))

sns.kdeplot(am_901_background, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 0])
sns.kdeplot(am_901_anomalies, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 0])

sns.kdeplot(co_901_background, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[0, 1])
sns.kdeplot(co_901_anomalies, bw_adjust=0.65, fill=True, color='blue', ax=axes[0, 1])

sns.kdeplot(do_901_background, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[0, 2])
sns.kdeplot(do_901_anomalies, bw_adjust=0.65, fill=True, color='purple', ax=axes[0, 2])

sns.kdeplot(ph_901_background, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[0, 3])
sns.kdeplot(ph_901_anomalies, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[0, 3])

sns.kdeplot(tu_901_background, bw_adjust=0.65, fill=True, color='gold', ax=axes[0, 4])
sns.kdeplot(tu_901_anomalies, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[0, 4])

sns.kdeplot(wt_901_background, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[0, 5])
sns.kdeplot(wt_901_anomalies, bw_adjust=0.65, fill=True, color='green', ax=axes[0, 5])

sns.kdeplot(am_905_background, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[1, 0])
sns.kdeplot(am_905_anomalies, bw_adjust=0.65, fill=True, color='red', ax=axes[1, 0])

sns.kdeplot(co_905_background, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 1])
sns.kdeplot(co_905_anomalies, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 1])

sns.kdeplot(do_905_background, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[1, 2])
sns.kdeplot(do_905_anomalies, bw_adjust=0.65, fill=True, color='purple', ax=axes[1, 2])

sns.kdeplot(ph_905_background, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[1, 3])
sns.kdeplot(ph_905_anomalies, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[1, 3])

sns.kdeplot(tu_905_background, bw_adjust=0.65, fill=True, color='gold', ax=axes[1, 4])
sns.kdeplot(tu_905_anomalies, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[1, 4])

sns.kdeplot(wt_905_background, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[1, 5])
sns.kdeplot(wt_905_anomalies, bw_adjust=0.65, fill=True, color='green', ax=axes[1, 5])

sns.kdeplot(am_906_background, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[2, 0])
sns.kdeplot(am_906_anomalies, bw_adjust=0.65, fill=True, color='red', ax=axes[2, 0])

sns.kdeplot(co_906_background, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[2, 1])
sns.kdeplot(co_906_anomalies, bw_adjust=0.65, fill=True, color='blue', ax=axes[2, 1])

sns.kdeplot(do_906_background, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 2])
sns.kdeplot(do_906_anomalies, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 2])

sns.kdeplot(ph_906_background, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[2, 3])
sns.kdeplot(ph_906_anomalies, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[2, 3])

sns.kdeplot(tu_906_background, bw_adjust=0.65, fill=True, color='gold', ax=axes[2, 4])
sns.kdeplot(tu_906_anomalies, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[2, 4])

sns.kdeplot(wt_906_background, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[2, 5])
sns.kdeplot(wt_906_anomalies, bw_adjust=0.65, fill=True, color='green', ax=axes[2, 5])

sns.kdeplot(am_907_background, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[3, 0])
sns.kdeplot(am_907_anomalies, bw_adjust=0.65, fill=True, color='red', ax=axes[3, 0])

sns.kdeplot(co_907_background, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[3, 1])
sns.kdeplot(co_907_anomalies, bw_adjust=0.65, fill=True, color='blue', ax=axes[3, 1])

sns.kdeplot(do_907_background, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[3, 2])
sns.kdeplot(do_907_anomalies, bw_adjust=0.65, fill=True, color='purple', ax=axes[3, 2])

sns.kdeplot(ph_907_background, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 3])
sns.kdeplot(ph_907_anomalies, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 3])

sns.kdeplot(tu_907_background, bw_adjust=0.65, fill=True, color='gold', ax=axes[3, 4])
sns.kdeplot(tu_907_anomalies, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[3, 4])

sns.kdeplot(wt_907_background, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[3, 5])
sns.kdeplot(wt_907_anomalies, bw_adjust=0.65, fill=True, color='green', ax=axes[3, 5])

# Clean default y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)

# Set the title for each variable
var_names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
for i, ax in enumerate(axes[0]):
    ax.set_title(var_names[i], fontfamily='Arial', fontsize=18)

# Set the x label for each variable
for ax in axes[3, :]:
    ax.set_xlabel('Value', fontname='Arial', fontsize=16)

# Set the y label for each variable
stations = ['Station 901', 'Station 905', 'Station 906', 'Station 907']
for i, ax in enumerate(axes):
    ax[0].set_ylabel(stations[i], fontname='Arial', fontsize=18)

# fig.suptitle(f'Distributions of anomalies and background events', fontname='Arial', fontsize=22)
plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig(f'plots/distributions_var.pdf', format="pdf", dpi=300, bbox_inches='tight')
