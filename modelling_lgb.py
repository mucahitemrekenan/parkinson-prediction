from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from project_functions import *


data = pd.read_csv('data/engineered_data2.csv')
print(data.dtypes)

target_cols = ['StartHesitation', 'Turn', 'Walking', 'Normal']

x = data.drop(columns=['Time', 'StartHesitation', 'Turn', 'Walking', 'Normal', 'session_id']).copy()
y = data[target_cols].idxmax(axis=1)

splitter = StratifiedShuffleSplit(n_splits=10, random_state=42)
for num, (train_index, _) in tqdm(enumerate(splitter.split(x, y))):
    lgb = LGBMClassifier(n_estimators=1000, max_depth=6)
    lgb.fit(x.loc[train_index], y[train_index])
    joblib.dump(lgb, f'models/lgb_model{num}.joblib')

# preds = lgb.predict_proba(x)