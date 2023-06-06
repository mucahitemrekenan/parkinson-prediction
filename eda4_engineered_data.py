import pandas as pd
import numpy as np


data = pd.read_csv('data/engineered_data.csv', low_memory=False)

corr = data.corr()