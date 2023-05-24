import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
# from sklearnex  import patch_sklearn
from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from project_functions import *

# patch_sklearn()
# matplotlib.use('Qt5Agg')

# data = read_files_parallel('data/train/tdcsfog/')
data = pd.read_csv('engineered_data.csv')

print(data.dtypes)

target_cols = ['starth', 'turn', 'walk', 'normal']

normal = data[data['normal'] == 1].sample(frac=0.1).index
turn = data[data['turn'] == 1].sample(frac=0.25).index
starth = data[data['starth'] == 1].sample(frac=1).index
walk = data[data['walk'] == 1].sample(frac=1).index
print([len(x) for x in [normal, turn, starth, walk]])

filter_index = pd.concat([normal.to_series(), turn.to_series(), starth.to_series(), walk.to_series()], axis=0)
filter_index = pd.Index(filter_index)

filtered_data = data[data.index.isin(filter_index)].copy()
filtered_data.dropna(inplace=True)

x = filtered_data.drop(columns=['time', 'starth', 'turn', 'walk', 'session_id', 'normal']).copy()
y = filtered_data[target_cols].idxmax(axis=1)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lgb = LGBMClassifier(n_estimators=1000, max_depth=6)
lgb.fit(x_train, y_train)
preds = lgb.predict(x_test)
np.unique(preds, return_counts=True)
print('test score:', lgb.score(x_test, y_test))

joblib.dump(lgb, 'lgb_model.joblib')
joblib.dump(encoder, 'encoder.joblib')

# for num, col in zip(range(4), ['starth', 'turn', 'walk']):
#     plt.plot(sub[:, col])
#     plt.title(col)
#     plt.show()
