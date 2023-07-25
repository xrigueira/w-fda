import pandas as pd
from random import shuffle

# This file shuffles the database by day correctly

# Read the data
station = 901
data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

# Group the data by day
data['date'] = pd.to_datetime(data['date'])
grouped_by_day = data.groupby(data['date'].dt.date)

# Shuffle order of the groups
group_keys = list(grouped_by_day.groups.keys())
shuffle(group_keys)

# Combine the suffle groups back into a single data frame
shuffled_df = pd.concat([grouped_by_day.get_group(key) for key in group_keys])

# Reset the index of the suffled data frame
shuffled_df.reset_index(drop=True, inplace=True)

shuffled_df.to_csv(f'data/labeled_{station}_shu.csv', sep=',', index=False)
