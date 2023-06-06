from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from base import *


main_config = {
    abs_energy: {'AccV': [25],
                 'AccML': [32],
                 'AccAP': [50]},

    Training.linear_trend_func: {'AccV': [5, 20],
                                 'AccML': [10, 25],
                                 'AccAP': [15, 50]},

    Training.autocorrelation_func: {'AccV': [5, 25],
                                    'AccML': [10, 50],
                                    'AccAP': [15, 100]},

    Training.mean_diff_func: {'AccV': [2],
                              'AccML': [5],
                              'AccAP': [10]}
}

data = pd.read_csv('data/engineered_data.csv', low_memory=False)
trainer = Training(data_path='data/', converters_path='converters/', models_path='models/', main_config=main_config)
trainer.data = data
trainer.prepare_data()
trainer.prepare_data4_booster()
Training.inspect_data(trainer.data)

# lgb = LGBMClassifier(n_estimators=5000, max_depth=6, n_jobs=-1)
# lgb.fit(trainer.x, trainer.y)
# joblib.dump(lgb, 'models/lgb_model.joblib')

splitter = StratifiedShuffleSplit(n_splits=50, random_state=42)
for num, (train_index, _) in tqdm(enumerate(splitter.split(trainer.x, trainer.y))):
    lgb = LGBMClassifier(n_estimators=200, max_depth=6, n_jobs=-1)
    lgb.fit(trainer.x.loc[train_index], trainer.y[train_index])
    joblib.dump(lgb, f'models/lgb_multi{num}.joblib')
