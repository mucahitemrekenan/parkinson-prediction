import os
import pandas as pd
from tqdm import tqdm
from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
from tsflex.features.integrations import seglearn_feature_dict_wrapper
from seglearn.feature_functions import *
from multiprocessing import Pool, cpu_count


def read_file(path: str, file: str) -> pd.DataFrame:
    patient_data = pd.read_csv(path + file)
    patient_data['session_id'] = file.replace('.csv', '')
    return patient_data


def read_files_parallel(path: str) -> pd.DataFrame:
    files = os.listdir(path)
    # Create a Pool object with as many cores as your machine has
    with Pool(cpu_count()) as pool:
        # Apply the read_file function to all files in parallel
        data_list = list(tqdm(pool.starmap(read_file, [(path, file) for file in files]), total=len(files)))
    # Concatenate all DataFrames in the list into one DataFrame
    data = pd.concat(data_list)
    data.reset_index(drop=True, inplace=True)
    return data


def generate_normal_col(data: pd.DataFrame):
    data['normal'] = ((data['starth'] == 0) & (data['turn'] == 0) & (data['walk'] == 0)).astype(int)


def rename_cols(data: pd.DataFrame):
    data.rename(columns={'Time': 'time', 'AccV': 'accv', 'AccML': 'accml',
                         'AccAP': 'accap', 'StartHesitation': 'starth',
                         'Turn': 'turn', 'Walking': 'walk', 'Valid': 'valid',
                         'Task': 'task', 'Id': 'id'}, inplace=True)


def calculate_features(cols: list, window: int, data: pd.DataFrame, function: callable) -> pd.DataFrame:
    # Copy the data for this process
    data_copy = data.copy()

    # Create feature descriptors for this window size
    feat_descriptor = MultipleFeatureDescriptors(
        functions=seglearn_feature_dict_wrapper(function),
        series_names=cols,
        windows=[window],
        strides=[1]
    )

    # Calculate features
    feature_collection = FeatureCollection(feat_descriptor)
    engineered_data = feature_collection.calculate(
        data_copy,
        window_idx='begin',
        return_df=True,
        show_progress=True,
        include_final_window=True,
        n_jobs=os.cpu_count()
    )
    return engineered_data


def abs_energy_func(x):
    return abs_energy(x.reshape(-1, 1))[0]


def std_func(x):
    return std(x.reshape(-1, 1))[0]


def var_func(x):
    return var(x.reshape(-1, 1))[0]


def min_func(x):
    return minimum(x.reshape(-1, 1))[0]


def max_func(x):
    return maximum(x.reshape(-1, 1))[0]


def skew_func(x):
    return skew(x.reshape(-1, 1))[0]


def kurt_func(x):
    return kurt(x.reshape(-1, 1))[0]


def mse_func(x):
    return mse(x.reshape(-1, 1))[0]


def mnx_func(x):
    return mean_crossings(x.reshape(-1, 1))[0]


def rolling_operation(data, column, window, func):
    return data[column].rolling(window).apply(func, raw=True)


def generate_rollingw_columns(data, rollingw_config, func):
    with Pool(cpu_count()) as pool:
        data_list = list(tqdm(pool.starmap(rolling_operation,
                                           [(data, column, value, func) for column, values in rollingw_config.items()
                                            for value in values])))
    engineered_data = pd.concat(data_list, axis=1)
    engineered_data.index = data.index
    return pd.concat([data, engineered_data], axis=1)
