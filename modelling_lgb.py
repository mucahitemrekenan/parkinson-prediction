from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from project_functions import *
from data2_model_adapter import prepare_data4_booster


x, y = prepare_data4_booster('data/engineered_data2.csv')
print(x.dtypes)

splitter = StratifiedShuffleSplit(n_splits=10, random_state=42)
for num, (train_index, _) in tqdm(enumerate(splitter.split(x, y))):
    lgb = LGBMClassifier(n_estimators=500, max_depth=6, n_jobs=16)
    lgb.fit(x.loc[train_index], y[train_index])
    joblib.dump(lgb, f'models/lgb_model{num}.joblib')
