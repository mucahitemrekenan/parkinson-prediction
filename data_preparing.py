import numpy as np
import pandas as pd


data = pd.read_csv('data/engineered_data.csv')
daily = pd.read_csv('data/daily_metadata.csv')
events = pd.read_csv('data/events.csv')
subjects = pd.read_csv('data/subjects.csv')
tasks = pd.read_csv('data/tasks.csv')
tfog_meta = pd.read_csv('data/tdcsfog_metadata.csv')
defog_meta = pd.read_csv('data/defog_metadata.csv')

subjects.drop(columns=['Visit'], inplace=True)
subjects.drop_duplicates(subset='Subject', inplace=True)

tfog_meta = pd.merge(tfog_meta, subjects, on=['Subject'], how='left')
defog_meta = pd.merge(defog_meta, subjects, on=['Subject'], how='left')

meta = pd.concat([tfog_meta, defog_meta], axis=0)

data = pd.merge(data, meta, left_on=['session_id'], right_on=['Id'], how='left')

data.drop(columns=['Id', 'Subject', 'Medication', 'YearsSinceDx', 'UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ'], inplace=True)
data['Sex'] = data['Sex'].map({'F': 0, 'M': 1}).fillna(1)

data.to_csv('data/engineered_data2.csv', index=False)
