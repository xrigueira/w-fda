import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# This file plots the results

data = np.load('GAN/data/synthetic_data.npy')

pH_syn = []
for i in data:
    pH_syn.append(np.average(i))

df = pd.read_csv('GAN/data/pH_cle.csv', sep=';', encoding='utf-8')
pH_ori = df.value.tolist()[:len(pH_syn)]

plt.plot(pH_ori)
plt.show()

plt.plot(pH_syn)

plt.show()
