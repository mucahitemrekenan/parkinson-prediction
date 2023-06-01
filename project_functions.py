import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from tsfresh.feature_extraction.feature_calculators import *
import seglearn.feature_functions as seg
from multiprocessing import Pool, cpu_count


def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage became: ", mem_usg, " MB")

    return df

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
    data['Normal'] = ((data['StartHesitation'] == 0) & (data['Turn'] == 0) & (data['Walking'] == 0)).astype(int)


def rename_cols(data: pd.DataFrame):
    data.rename(columns={'Time': 'time', 'AccV': 'accv', 'AccML': 'accml',
                         'AccAP': 'accap', 'StartHesitation': 'starth',
                         'Turn': 'turn', 'Walking': 'walk', 'Valid': 'valid',
                         'Task': 'task', 'Id': 'id'}, inplace=True)


def autocorrelation_func(x):
    return np.nan_to_num(autocorrelation(x, 1))


def autocorrelation_edge_func(x):
    return np.nan_to_num(autocorrelation(x, len(x) - 1))


def linear_trend_func(x):
    return linear_trend(x, [{"attr": "slope"}])[0][1]


def mean_diff_func(x):
    return seg.mean_diff([x])[0]


def tras_func(x):
    return time_reversal_asymmetry_statistic(x, 1)


def fft_aggregated_skew_func(x):
    return list(fft_aggregated(x, [{'aggtype': 'skew'}]))[0][1]


def partial_autocorrelation_func(x):
    return list(partial_autocorrelation(x, [{'lag': 1}]))[0][1]


def c3_func(x):
    return c3(x, 1)


def cid_ce_func(x):
    return cid_ce(x, normalize=False)


def std_func(x):
    return np.std(x)


def function_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {round(end - start, 2)} seconds to run.")
        return result
    return wrapper


def rolling_operation(grouped_data, column, window, func):
    return grouped_data[column].rolling(window).apply(func, raw=True) \
        .rename(f'{column}_{window}_{func.__name__}')


@function_timer
def generate_rollingw_columns(data, main_config):
    grouped_data = data.groupby('session_id')
    starmap_args = []
    for func, rollingw_config in main_config.items():
        for column, windows in rollingw_config.items():
            for window in windows:
                starmap_args.append((grouped_data, column, window, func))
    with Pool(cpu_count()) as pool:
        manipulated_data = list(pool.starmap(rolling_operation, starmap_args))
        manipulated_data = pd.concat(manipulated_data, axis=1)
        manipulated_data.index = data.index
    return pd.concat([data, manipulated_data], axis=1)



