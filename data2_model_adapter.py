import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import joblib


def prepare_data(engineered_file_path):
    data = pd.read_csv(engineered_file_path, low_memory=False)

    data.drop(columns=['Time', 'Test'], inplace=True)
    data.dropna(inplace=True)

    sessions = data['session_id'].unique()
    session_labels = {value: key for key, value in dict(enumerate(sessions)).items()}
    data['session_id'] = data['session_id'].map(session_labels).fillna(999)
    data['session_id'] = np.sin(data['session_id'])

    joblib.dump(session_labels, 'sessions_dict.joblib')
    return data


def prepare_data4_booster(path):
    data = prepare_data(path)

    target_cols = ['StartHesitation', 'Turn', 'Walking', 'Normal']

    x = data.drop(columns=target_cols).copy()
    y = data[target_cols].idxmax(axis=1)
    return x, y


def prepare_data4_nn(path):
    data = prepare_data(path)

    target_cols = ['StartHesitation', 'Turn', 'Walking', 'Normal']

    x = data.drop(columns=target_cols).copy()
    y = data[target_cols].to_numpy()

    x = StandardScaler().fit_transform(x)
    x = MinMaxScaler((0, 1)).fit_transform(x)
    return x, y


def prepare_data4_rnn(path, sequence_length):
    data = prepare_data(path)

    target_cols = ['StartHesitation', 'Turn', 'Walking', 'Normal']

    x = data.drop(columns=target_cols).copy()
    y = data[target_cols].to_numpy()

    x = StandardScaler().fit_transform(x)
    x = MinMaxScaler((0, 1)).fit_transform(x)

    num_rows = x.shape[0]

    # Calculate the required padding
    trim_size = num_rows % sequence_length

    if trim_size != 0:
        x = x[:-trim_size]
        y = y[:-trim_size]

    x = np.reshape(x, (-1, sequence_length, x.shape[1]))
    y = np.reshape(y, (-1, sequence_length, y.shape[1]))
    return x, y

