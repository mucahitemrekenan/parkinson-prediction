import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from tqdm import tqdm
from sklearnex  import patch_sklearn
from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
patch_sklearn()
matplotlib.use('Qt5Agg')


# setting train / test paths for tfog defog datasets
# ---train---
tfog_train_path = 'data/train/tdcsfog/'
tfog_train_files = os.listdir(tfog_train_path)
tfog_train_data = pd.DataFrame()

defog_train_path = 'data/train/defog/'
defog_train_files = os.listdir(defog_train_path)
defog_train_data = pd.DataFrame()
# ---test---
tfog_test_path = 'data/test/tdcsfog/'
tfog_test_files = os.listdir(tfog_test_path)
tfog_test_data = pd.DataFrame()

defog_test_path = 'data/test/defog/'
defog_test_files = os.listdir(defog_test_path)
defog_test_data = pd.DataFrame()

# file reading and concatenation for training files
for file in tqdm(tfog_train_files):
    patient_tfog_train_data = pd.read_csv(tfog_train_path+file)
    patient_tfog_train_data['session_id'] = file.replace('.csv', '')
    tfog_train_data = pd.concat([tfog_train_data, patient_tfog_train_data], ignore_index=True)


conditions = [tfog_train_data['StartHesitation'] == 1,
              tfog_train_data['Turn'] == 1,
              tfog_train_data['Walking'] == 1]

choises = ['StartHesitation', 'Turn', 'Walking']
tfog_train_data['event'] = np.select(conditions, choises, default='normal')

event_encoder = LabelEncoder()
tfog_train_data['target'] = event_encoder.fit_transform(tfog_train_data['event'])

x = tfog_train_data[['AccV', 'AccML', 'AccAP']].copy()
y = tfog_train_data['target'].copy()

lgb = LGBMClassifier(objective='multiclass', n_estimators=10, max_depth=7,
                     learning_rate=0.1, num_class=4)

lgb.fit(x, y, eval_metric='multi_logloss')

# file reading and concatenation for test files
for tfog_file, defog_file in tqdm(zip(tfog_test_files, defog_test_files)):
    patient_tfog_data = pd.read_csv(tfog_test_path+tfog_file)
    patient_tfog_data['session_id'] = tfog_file.replace('.csv', '')
    tfog_test_data = pd.concat([tfog_test_data, patient_tfog_data], ignore_index=True)

    patient_defog_data = pd.read_csv(defog_test_path+defog_file)
    patient_defog_data['session_id'] = defog_file.replace('.csv', '')
    defog_test_data = pd.concat([defog_test_data, patient_defog_data], ignore_index=True)

tfog_test_data['Id'] = tfog_test_data['session_id'] + '_' + tfog_test_data['Time'].astype(str)
defog_test_data['Id'] = defog_test_data['session_id'] + '_' + defog_test_data['Time'].astype(str)

test_data = pd.concat([tfog_test_data, defog_test_data], axis=0, ignore_index=True)
test_data = test_data[['Id', 'AccV', 'AccML', 'AccAP']].copy()

test_data[['StartHesitation', 'Turn', 'Walking', 'Normal']] = lgb.predict_proba(test_data.drop(columns=['Id']))
test_data[['StartHesitation', 'Turn', 'Walking', 'Normal']] = test_data[['StartHesitation', 'Turn', 'Walking', 'Normal']].round(3)

sample_submission = pd.read_csv('data/sample_submission.csv')
submission = pd.merge(sample_submission[['Id']], test_data[['Id', 'StartHesitation', 'Turn', 'Walking']], how='left', on='Id').fillna(0.0)
submission.to_csv('submission.csv', index=False)

sample_sub = pd.read_csv('data/sample_submission.csv')

print('index:',np.unique(sample_sub.index == submission.index, return_counts=True))
print('id:', np.unique(sample_sub.Id == submission.Id, return_counts=True))
