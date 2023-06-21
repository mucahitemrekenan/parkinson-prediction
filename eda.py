import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from base import Utils

matplotlib.use('Qt5Agg')

"""
This is a simple eda. More detailed analysis were in the competition page. We benefited from the analyzes on the page.
Because we wanted to save time.
"""


def get_target_col_distribution(data: pd.DataFrame, cols: list, msg: str):
    print(msg)
    for col in cols:
        print(data.value_counts(subset=col, normalize=True))


def plot_sessions(sensor_data: pd.DataFrame, sessions: list, target_cols: list, num_graphs: int):
    # there is so much session data, we pick some of them randomly to make analysis
    assert num_graphs <= len(sessions), 'number of graphs can not be bigger than the session size'
    session_subset = np.random.choice(sessions, num_graphs, replace=False)
    for session in tqdm(session_subset):
        sensor_data.loc[sensor_data['session_id'] == session].plot(x='Time', y=target_cols + ['AccV', 'AccML', 'AccAP'],
                                                                   figsize=(19.1, 9.6), legend=True)
    plt.show()
    plt.clf()


# variables to target dataset
source_type = 'tdcsfog'
path = f'data/train/{source_type}/'
target_cols = ['StartHesitation', 'Turn', 'Walking']

data = Utils.read_files_parallel(path)

# data = pd.read_csv(f'data/{source_type}_train_data.csv')
data['session_id'] = data['session_id'].astype(str)

# we yield length of data through all patients as minute
(data['session_id'].value_counts() / (100 * 60)).plot(figsize=(19.1, 9.6), legend=True)
plt.xticks(rotation=90)
plt.show()
plt.clf()

get_target_col_distribution(data, target_cols, 'distribution of main data')

# we inspect target variables by session_id which sesssion_id does which movement and how much
move_counts = data.value_counts(subset=target_cols + ['session_id']).reset_index()
move_counts = move_counts[(move_counts[target_cols].sum(axis=1) == 1)]

for col in target_cols:
    plt.plot('session_id', 'count', '.', data=move_counts.loc[move_counts[col] == 1])
plt.xticks(rotation=90)
plt.legend(target_cols)
plt.grid(visible=True, axis='both', which='major')
plt.show()
plt.clf()

# We get rows which Valid and Task features are True. They have more reliable target variables, competition holders say.
if 'Valid' in data.columns and 'Task' in data.columns:
    sensor_data = data[data['Valid'] & data['Task']].copy()
else:
    sensor_data = data.copy()

get_target_col_distribution(sensor_data, target_cols, 'distribution after valid & task filtering')

plot_sessions(sensor_data, sensor_data['session_id'].unique(), target_cols, 10)

metadata = pd.read_csv(f'data/{source_type}_metadata.csv')
events = pd.read_csv('data/events.csv')
tasks = pd.read_csv('data/tasks.csv')
subjects = pd.read_csv('data/subjects.csv')

# Time data is 100 hz sampled. We convert it to second.
data['time_sec'] = data['Time'] / 100

df = data.copy()
# We merge events data 2 times. First for the beginning of the event, second for the finish of the event.
df = df.merge(events, left_on=['session_id', 'time_sec'], right_on=['Id', 'Init'], how='left')
df = df.merge(events, left_on=['session_id', 'time_sec'], right_on=['Id', 'Completion'], how='left', copy=False)

df['Type_x'] = df['Type_x'].fillna(df['Type_y'])
df['Kinetic_x'] = df['Kinetic_x'].fillna(df['Kinetic_y'])

df.drop(columns=['Id_x', 'Init_x', 'Completion_x', 'Id_y', 'Init_y', 'Completion_y', 'Type_y', 'Kinetic_y'], inplace=True)
df.rename(columns={'Type_x': 'type', 'Kinetic_x': 'kinetic'}, inplace=True)

event_filter = events['Id'].isin(df.session_id.unique())

encoder = LabelEncoder()
df['type_num'] = encoder.fit_transform(df['type'])

# We fill NaN values forward.Thus values between beginning and finish are filled.
# Forward fill operation performed while filtering data by session id.
# Because we don't want interactions between sessions.
for _, row in tqdm(events[event_filter].iterrows()):
    filter1 = df['session_id'] == row['Id']
    filter2 = df['time_sec'].between(row['Init'], row['Completion'])
    df.loc[filter1 & filter2, ['type', 'kinetic']] = df.loc[filter1 & filter2, ['type', 'kinetic']].fillna(method='ffill')

plot_sessions(df, df.session_id.unique(), target_cols + ['type_num'], 90)

# ==============================
# analysis for unlabeled data
# ==============================

unlabeled_path = 'data/unlabeled/'
unlabeled_files = os.listdir(unlabeled_path)

# There is so much data, so we randomly select unlabeled files to plot.
for file in np.random.choice(unlabeled_files, 25, replace=False):
    patient_unlabeled_data = pd.read_parquet(unlabeled_path+file)
    patient_unlabeled_data['session_id'] = file.replace('.parquet', '')
    patient_unlabeled_data.plot(x='Time', y=['AccV', 'AccML', 'AccAP'])
    plt.show()

# We get how many minutes each patient has recorded.
session_time_infos = pd.DataFrame(columns=['session_id', 'minute'])
for file in tqdm(unlabeled_files):
    patient_unlabeled_data = pd.read_parquet(unlabeled_path+file, columns=['Time'])
    session_id = file.replace('.parquet', '')
    minute = patient_unlabeled_data['Time'].max() / (100 * 60)
    patient_data = pd.DataFrame({'session_id': session_id, 'minute': minute}, index=[0])
    session_time_infos = pd.concat([session_time_infos, patient_data], ignore_index=True)

session_time_infos['hour'] = session_time_infos['minute'] / 60
session_time_infos['day'] = session_time_infos['hour'] / 24
