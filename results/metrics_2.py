import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'

# Simmulation results
contamination_levels = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]
metrics = {
    "Accuracy": {
        "MOUT": [0.999, 0.995, 1.000, 0.990, 0.990, 0.962],
        "MUOD": [0.943, 0.948, 0.948, 0.967, 0.976, 0.990],
        "MS": [0.990, 1.000, 0.990, 0.962, 0.967, 0.957],
        "MMSA": [1.000, 0.995, 0.995, 0.995, 0.990, 0.962],
    },
    "Precision": {
        "MOUT": [1.000, 0.500, 1.000, 0.800, 0.800, 0.200],
        "MUOD": [1.000, 1.000, 1.000, 1.000, 1.000, 0.900],
        "MS": [1.000, 1.000, 0.500, 0.200, 0.300, 0.100],
        "MMSA": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    },
    "Recall": {
        "MOUT": [0.990, 1.000, 1.000, 1.000, 1.000, 1.000],
        "MUOD": [0.000, 0.154, 0.267, 0.588, 0.667, 0.900],
        "MS": [0.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        "MMSA": [1.000, 0.667, 0.800, 0.909, 0.833, 0.556],
    },
    "F1-Score": {
        "MOUT": [0.990, 0.667, 1.000, 0.889, 0.889, 0.333],
        "MUOD": [0.000, 0.267, 0.421, 0.741, 0.800, 0.900],
        "MS": [0.000, 1.000, 0.667, 0.333, 0.462, 0.182],
        "MMSA": [1.000, 0.800, 0.889, 0.952, 0.909, 0.714],
    },
    # "Error Rates": {
    #     "MOUT": [0.000, 0.005, 0.010, 0.010, 0.010, 0.038],
    #     "MUOD": [0.057, 0.052, 0.052, 0.033, 0.024, 0.010],
    #     "MS": [0.010, 0.010, 0.010, 0.038, 0.033, 0.043],
    #     "MMSA": [0.000, 0.005, 0.005, 0.005, 0.010, 0.038],
    # },
}

# Plot setup
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
titles = list(metrics.keys())

# Colors and labels
methods = ["MOUT", "MUOD", "MS", "MMSA"]
colors = ['darkgrey', 'skyblue', 'cornflowerblue', 'royalblue']
bar_width = 0.16

# Plot each metric
for i, ax in enumerate(axes.flat):
    metric = titles[i]
    data = metrics[metric]
    x = np.arange(len(contamination_levels))  # Positions for x-axis

    for j, method in enumerate(methods):
        ax.bar(x + j * bar_width, data[method], bar_width, label=method, color=colors[j])
    
    ax.set_xlabel("Contamination level", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels([str(level) for level in contamination_levels], fontsize=10)

fig.suptitle('Simulation performance', fontsize=16, fontname='Arial')

# Create a single legend at the bottom center
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(methods))

# Adjust layout to make space for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)  # Add extra space at the bottom for the legend

# plt.show()

# Save plot
plt.savefig('results/metrics_sim.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Results on real data
metrics = {
    "Station 901": {
        "Accuracy": {
            "MOUT": 0.99258,
            "MUOD": 0.90428,
            "MS": 0.99351,
            "MMSA": 0.85228,
            "OC-SVM": 0.91826,
            "LR": 0.97794,
            "RF": 0.99881,
        },
        "Precision": {
            "MOUT": 0.09524,
            "MUOD": 0.03893,
            "MS": 0.00000,
            "MMSA": 0.02870,
            "OC-SVM": 0.02895,
            "LR": 0.12579,
            "RF": 0.84071,
        },
        "Recall": {
            "MOUT": 0.02105,
            "MUOD": 0.60000,
            "MS": 0.00000,
            "MMSA": 0.68421,
            "OC-SVM": 0.36842,
            "LR": 0.42105,
            "RF": 1.00000,
        },
        "F1-Score": {
            "MOUT": 0.03448,
            "MUOD": 0.07312,
            "MS": 0.00000,
            "MMSA": 0.05509,
            "OC-SVM": 0.05368,
            "LR": 0.19371,
            "RF": 0.91346,
        },
        # "Error Rates": {
        #     "MOUT": 0.00742,
        #     "MUOD": 0.09572,
        #     "MS": 0.00649,
        #     "MMSA": 0.14772,
        #     "OC-SVM": 0.08174,
        #     "LR": 0.02206,
        #     "RF": 0.00119,
        # },
    },
    "Station 905": {
        "Accuracy": {
            "MOUT": 0.97253,
            "MUOD": 0.84494,
            "MS": 0.97237,
            "MMSA": 0.83279,
            "OC-SVM": 0.87216,
            "LR": 0.97353,
            "RF": 0.99983,
        },
        "Precision": {
            "MOUT": 0.00000,
            "MUOD": 0.00517,
            "MS": 0.00000,
            "MMSA": 0.03439,
            "OC-SVM": 0.01055,
            "LR": 0.51911,
            "RF": 0.99398,
        },
        "Recall": {
            "MOUT": 0.00000,
            "MUOD": 0.02424,
            "MS": 0.00000,
            "MMSA": 0.18788,
            "OC-SVM": 0.03939,
            "LR": 0.49394,
            "RF": 1.00000,
        },
        "F1-Score": {
            "MOUT": 0.00000,
            "MUOD": 0.00852,
            "MS": 0.00000,
            "MMSA": 0.05813,
            "OC-SVM": 0.01665,
            "LR": 0.50621,
            "RF": 0.99698,
        },
        # "Error Rates": {
        #     "MOUT": 0.02747,
        #     "MUOD": 0.15506,
        #     "MS": 0.02763,
        #     "MMSA": 0.16721,
        #     "OC-SVM": 0.12784,
        #     "LR": 0.02647,
        #     "RF": 0.00017,
        # },
    },
    "Station 906": {
        "Accuracy": {
            "MOUT": 0.99354,
            "MUOD": 0.89482,
            "MS": 0.99845,
            "MMSA": 0.85105,
            "OC-SVM": 0.91583,
            "LR": 0.98869,
            "RF": 0.99938,
        },
        "Precision": {
            "MOUT": 0.00000,
            "MUOD": 0.01227,
            "MS": 0.00000,
            "MMSA": 0.00829,
            "OC-SVM": 0.00594,
            "LR": 0.08808,
            "RF": 0.69697,
        },
        "Recall": {
            "MOUT": 0.00000,
            "MUOD": 0.91304,
            "MS": 0.00000,
            "MMSA": 0.86957,
            "OC-SVM": 0.34783,
            "LR": 0.73913,
            "RF": 1.00000,
        },
        "F1-Score": {
            "MOUT": 0.00000,
            "MUOD": 0.02422,
            "MS": 0.00000,
            "MMSA": 0.01642,
            "OC-SVM": 0.01168,
            "LR": 0.15741,
            "RF": 0.82143,
        },
        # "Error Rates": {
        #     "MOUT": 0.00647,
        #     "MUOD": 0.10519,
        #     "MS": 0.00155,
        #     "MMSA": 0.14895,
        #     "OC-SVM": 0.08417,
        #     "LR": 0.01131,
        #     "RF": 0.00062,
        # },
    },
    "Station 907": {
        "Accuracy": {
            "MOUT": 0.98145,
            "MUOD": 0.86427,
            "MS": 0.98131,
            "MMSA": 0.84339,
            "OC-SVM": 0.87979,
            "LR": 0.97940,
            "RF": 0.99972,
        },
        "Precision": {
            "MOUT": 0.00000,
            "MUOD": 0.04343,
            "MS": 0.00000,
            "MMSA": 0.03996,
            "OC-SVM": 0.00617,
            "LR": 0.25424,
            "RF": 0.98502,
        },
        "Recall": {
            "MOUT": 0.00000,
            "MUOD": 0.30038,
            "MS": 0.00000,
            "MMSA": 0.32319,
            "OC-SVM": 0.03422,
            "LR": 0.05703,
            "RF": 1.00000,
        },
        "F1-Score": {
            "MOUT": 0.00000,
            "MUOD": 0.07589,
            "MS": 0.00000,
            "MMSA": 0.07113,
            "OC-SVM": 0.01045,
            "LR": 0.09317,
            "RF": 0.99245,
        },
        # "Error Rates": {
        #     "MOUT": 0.01855,
        #     "MUOD": 0.13573,
        #     "MS": 0.01870,
        #     "MMSA": 0.15661,
        #     "OC-SVM": 0.12021,
        #     "LR": 0.02060,
        #     "RF": 0.00028,
        # },
    },
}

# Plot setup
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
titles = list(metrics.keys())

# Colors and labels
methods = ["MOUT", "MUOD", "MS", "MMSA", "OC-SVM", "LR", "RF"]
colors = ['darkgrey', 'skyblue', 'cornflowerblue', 'royalblue', 'lightcoral', 'indianred', 'firebrick']
bar_width = 0.08

# Plot each metric
for i, ax in enumerate(axes.flat):
    station = titles[i]
    data = metrics[station]
    x = np.arange(len(data))  # Positions for x-axis

    for j, method in enumerate(methods):
        ax.bar(x + j * bar_width, [data[metric][method] for metric in data], bar_width, label=method, color=colors[j])
    
    # ax.set_xlabel("Metrics")
    ax.set_ylabel(station, fontsize=12)
    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(list(data.keys()), fontsize=10, rotation=0)

fig.suptitle('Performance on real data', fontsize=16, fontname='Arial')

# Create a single legend at the bottom center
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(methods))

# Adjust layout to make space for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.16)  # Add extra space at the bottom for the legend

# plt.show()

plt.savefig('results/metrics_real_data.pdf', format='pdf', dpi=300, bbox_inches='tight')
