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
unlabeled_path = 'data/unlabeled/'
unlabeled_files = os.listdir(unlabeled_path)
unlabeled_data = pd.DataFrame()

# file reading and concatenation for training files
# for file in tqdm(unlabeled_files[:3]):
#     patient_unlabeled_data = pd.read_parquet(unlabeled_path+file)
#     patient_unlabeled_data['session_id'] = file.replace('.csv', '')
#     unlabeled_data = pd.concat([unlabeled_data, patient_unlabeled_data])

file = unlabeled_files[0]
patient_unlabeled_data = pd.read_parquet(unlabeled_path+file)
patient_unlabeled_data['session_id'] = file.replace('.parquet', '')

i = 1_000_000
patient_unlabeled_data.loc[i:i+1_000_000,:].plot(x='Time', y=['AccV', 'AccML', 'AccAP'])
plt.show()


session_time_infos = pd.DataFrame(columns=['session_id', 'min'])
for file in tqdm(unlabeled_files):
    patient_unlabeled_data = pd.read_parquet(unlabeled_path+file, columns=['Time'])
    session_id = file.replace('.parquet', '')
    min = patient_unlabeled_data['Time'].max() / (100 * 60)
    patient_data = pd.DataFrame({'session_id': session_id, 'min':min}, index=[0])
    session_time_infos = pd.concat([session_time_infos, patient_data], ignore_index=True)

session_time_infos['hour'] = session_time_infos['min'] / 60
session_time_infos['day'] = session_time_infos['hour'] / 24