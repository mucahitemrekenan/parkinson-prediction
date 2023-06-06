import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import time
from tsfresh.feature_extraction.feature_calculators import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seglearn.feature_functions as seg
from multiprocessing import Pool, cpu_count
# import tensorflow as tf


class Training:

    def __init__(self, data_path, converters_path, models_path, main_config):

        self.data_path = data_path
        self.converters_path = converters_path
        self.models_path = models_path
        self.main_config = main_config
        self.tfog_data = None
        self.defog_data = None
        self.data = None
        self.x = None
        self.y = None
        self.target_cols = ['StartHesitation', 'Turn', 'Walking', 'Normal']
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler((0, 1))

    def run(self):
        self.tfog_data = Training.read_files_parallel(self.data_path + 'train/tdcsfog/')
        self.defog_data = Training.read_files_parallel(self.data_path + 'train/defog/')
        self.defog_data.drop(columns=['Valid', 'Task'], inplace=True)
        self.prepare_meta()
        self.generate_normal_col()
        self.generate_rollingw_columns()
        self.data.to_csv(self.data_path + 'engineered_data.csv', index=False)

    @staticmethod
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

    @staticmethod
    def read_file(path: str, file: str) -> pd.DataFrame:
        patient_data = pd.read_csv(path + file)
        patient_data['session_id'] = file.replace('.csv', '')
        return patient_data

    @staticmethod
    def read_files_parallel(path):
        files = os.listdir(path)
        # Create a Pool object with as many cores as your machine has
        with Pool(cpu_count()) as pool:
            # Apply the read_file function to all files in parallel
            data_list = list(tqdm(pool.starmap(Training.read_file, [(path, file) for file in files]), total=len(files)))
        # Concatenate all DataFrames in the list into one DataFrame
        data = pd.concat(data_list)
        data.reset_index(drop=True, inplace=True)
        return data

    def generate_normal_col(self):
        self.data['Normal'] = ((self.data['StartHesitation'] == 0) & (self.data['Turn'] == 0) &
                               (self.data['Walking'] == 0)).astype(int)

    @staticmethod
    def autocorrelation_func(x):
        return np.nan_to_num(autocorrelation(x, 1))

    @staticmethod
    def autocorrelation_edge_func(x):
        return np.nan_to_num(autocorrelation(x, len(x) - 1))

    @staticmethod
    def linear_trend_func(x):
        return linear_trend(x, [{"attr": "slope"}])[0][1]

    @staticmethod
    def mean_diff_func(x):
        return seg.mean_diff([x])[0]

    @staticmethod
    def tras_func(x):
        return time_reversal_asymmetry_statistic(x, 1)

    @staticmethod
    def fft_aggregated_skew_func(x):
        return list(fft_aggregated(x, [{'aggtype': 'skew'}]))[0][1]

    @staticmethod
    def partial_autocorrelation_func(x):
        return list(partial_autocorrelation(x, [{'lag': 1}]))[0][1]

    @staticmethod
    def c3_func(x):
        return c3(x, 1)

    @staticmethod
    def cid_ce_func(x):
        return cid_ce(x, normalize=False)

    @staticmethod
    def std_func(x):
        return np.std(x)

    @staticmethod
    def function_timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {round(end - start, 2)} seconds to run.")
            return result
        return wrapper

    @staticmethod
    def rolling_operation(grouped_data, column, window, func):
        return grouped_data[column].rolling(window).apply(func, raw=True).rename(f'{column}_{window}_{func.__name__}')

    @function_timer.__func__
    def generate_rollingw_columns(self):
        grouped_data = self.data.groupby('session_id')
        starmap_args = []
        for func, rollingw_config in self.main_config.items():
            for column, windows in rollingw_config.items():
                for window in windows:
                    starmap_args.append((grouped_data, column, window, func))
        with Pool(cpu_count()) as pool:
            manipulated_data = list(pool.starmap(Training.rolling_operation, starmap_args))
            manipulated_data = pd.concat(manipulated_data, axis=1)
            manipulated_data.index = self.data.index
            self.data = pd.concat([self.data, manipulated_data], axis=1)

    def prepare_meta(self):
        subjects = pd.read_csv(self.data_path+'subjects.csv')
        tfog_meta = pd.read_csv(self.data_path+'tdcsfog_metadata.csv')
        defog_meta = pd.read_csv(self.data_path+'defog_metadata.csv')

        subjects.drop(columns=['Visit'], inplace=True)
        subjects.drop_duplicates(subset='Subject', inplace=True)

        tfog_meta = pd.merge(tfog_meta, subjects, on=['Subject'], how='left')
        defog_meta = pd.merge(defog_meta, subjects, on=['Subject'], how='left')

        self.tfog_data = pd.merge(self.tfog_data, tfog_meta, left_on=['session_id'], right_on=['Id'], how='left')
        self.defog_data = pd.merge(self.defog_data, defog_meta, left_on=['session_id'], right_on=['Id'], how='left')

        self.data = pd.concat([self.tfog_data, self.defog_data], axis=0, ignore_index=True)

        del self.tfog_data, self.defog_data

        self.data.drop(columns=['Id', 'Subject', 'Medication', 'YearsSinceDx', 'UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ'],
                       inplace=True)
        self.data['Id'] = self.data['session_id'] + '_' + self.data['Time'].astype(str)
        self.data.rename(columns={'Visit_x': 'Visit'}, inplace=True)
        self.data['Sex'] = self.data['Sex'].map({'F': 0, 'M': 1}).fillna(1)

    def prepare_data(self):
        self.data.drop(columns=['Id', 'Time', 'Test'], inplace=True)
        self.data.dropna(inplace=True)
        sessions = self.data['session_id'].unique()
        session_labels = {value: key for key, value in dict(enumerate(sessions)).items()}
        self.data['session_id'] = self.data['session_id'].map(session_labels).fillna(999)
        self.data['session_id'] = np.sin(self.data['session_id'])

        joblib.dump(session_labels, self.converters_path+'sessions_dict.joblib')

        self.x = self.data.drop(columns=self.target_cols).copy()

    def prepare_data4_booster(self):
        self.y = self.data[self.target_cols].idxmax(axis=1)

    def prepare_data4_nn(self):
        self.y = self.data[self.target_cols].to_numpy()

        self.x = self.standard_scaler.fit_transform(self.x)
        self.x = self.minmax_scaler.fit_transform(self.x)

        joblib.dump({'standard_scaler': self.standard_scaler, 'minmax_scaler': self.minmax_scaler},
                    self.converters_path+'scalers.joblib')

    def prepare_data4_rnn(self, sequence_length):
        self.prepare_data4_nn()

        num_rows = self.x.shape[0]
        trim_size = num_rows % sequence_length

        if trim_size != 0:
            self.x = self.x[:-trim_size]
            self.y = self.y[:-trim_size]

        self.x = np.reshape(self.x, (-1, sequence_length, self.x.shape[1]))
        self.y = np.reshape(self.y, (-1, sequence_length, self.y.shape[1]))

    @staticmethod
    def find_variable_name(variable):
        for name, value in globals().items():
            if value is variable:
                return name

    @staticmethod
    def inspect_data(data):
        name = Training.find_variable_name(data)
        print(f'{name} shape:', data.shape)
        print('-----------------------------')
        print(f'{name} columns:', data.columns)
        print('-----------------------------')
        print(f'{name} index value counts:', data.index.value_counts())
        print('-----------------------------')
        print(f'{name} dtypes:', data.dtypes)
        print('-----------------------------')
        print(f'{name} isnull:', data.isnull().sum())
        print('-----------------------------')
        print(f'{name} head:', data.head())
        print('-----------------------------')
        print(f'{name} tail:', data.tail())
        print('-----------------------------')


class Inference(Training):
    def __init__(self, data_path, converters_path, models_path, main_config):
        super().__init__(data_path, converters_path, models_path, main_config)
        self.tfog_test_data = None
        self.defog_test_data = None
        self.ids = None
        self.session_labels = joblib.load(self.converters_path+'sessions_dict.joblib')

    def run(self):
        self.tfog_data = Training.read_files_parallel(self.data_path + 'test/tdcsfog/')
        self.defog_data = Training.read_files_parallel(self.data_path + 'test/defog/')
        self.prepare_meta()
        self.generate_rollingw_columns()


    def prepare_data(self):
        self.data.drop(columns=['Time', 'Test'], inplace=True)
        self.data.dropna(inplace=True)
        self.ids = self.data['Id'].copy()
        self.data.drop(columns=['Id'], inplace=True)
        self.data['session_id'] = self.data['session_id'].map(self.session_labels).fillna(999)
        self.data['session_id'] = np.sin(self.data['session_id'])

    def prepare_data4_nn(self):
        scalers = joblib.load(self.converters_path+'scalers.joblib')
        self.data = self.data[scalers['standard_scaler'].get_feature_names_out()]
        self.data = scalers['standard_scaler'].transform(self.data)
        self.data = scalers['minmax_scaler'].transform(self.data)

    def prepare_data4_rnn(self, sequence_length):
        self.prepare_data4_nn()

        num_rows = self.data.shape[0]
        trim_size = num_rows % sequence_length

        if trim_size != 0:
            self.data = self.data[:-trim_size]
            self.ids = self.ids[:-trim_size]

        self.data = np.reshape(self.data, (-1, sequence_length, self.data.shape[1]))


# class AveragePrecision(tf.keras.metrics.Metric):
#     def __init__(self, num_classes, thresholds=None, name='avg_precision', **kwargs):
#         super(AveragePrecision, self).__init__(name=name, **kwargs)
#         self.num_classes = num_classes
#         self.thresholds = thresholds
#         self.class_precision = [tf.keras.metrics.Precision(thresholds) for _ in range(num_classes)]
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         for i, precision in enumerate(self.class_precision):
#             precision.update_state(y_true[..., i], y_pred[..., i])
#
#     def result(self):
#         return tf.math.reduce_mean([precision.result() for precision in self.class_precision])
#
#     def reset_state(self):
#         for precision in self.class_precision:
#             precision.reset_state()
#
#     def get_config(self):
#         config = super(AveragePrecision, self).get_config()
#         config.update({"num_classes": self.num_classes, "thresholds": self.thresholds})
#         return config




