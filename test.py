import pandas as pd

# Read the data
data = pd.read_csv(f'data/labeled_901_pro.csv', sep=',', encoding='utf-8')



# Count number of 1s and 0s.

ones = list(data.label).count(1)
zeros = list(data.label).count(0)
total = len(data.label)

print(ones, zeros, total)

cont = (ones / total) * 100

print(cont)