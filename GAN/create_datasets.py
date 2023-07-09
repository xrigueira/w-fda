import warnings
warnings.filterwarnings('ignore')

import requests
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml

"""I used this program to create assets.h5, which is now stored in One Drive
because it is too heavy. I just needed assets.h5 to make sure the tgan implementation 
worked as in the example provided by Stefan Jansen.

To get assets.h5 I have to follow the intructions here:
https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/data/create_datasets.ipynb
is the "Quandl Wiki Prices" section and once I have tke wiki_prices.csv run this program
to finally compile assets.h5"""

pd.set_option('display.expand_frame_repr', False)

DATA_STORE = Path('assets.h5') # Saved in Drive folder: /Data

# Create dataset with Quandl Wiki Prices
df = (pd.read_csv('wiki_prices.csv',
                parse_dates=['date'],
                index_col=['date', 'ticker'],
                infer_datetime_format=True)
    .sort_index())

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)