import os
from tqdm import tqdm
import joblib
import time
from tsfresh.feature_extraction.feature_calculators import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
import seglearn.feature_functions as seg
from multiprocessing import Pool, cpu_count
from tensorflow.keras.models import load_model
import scipy as sp
import pywt
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class StatisticFunctions:
    """
    In StatisticFunctions class, we define relevant functions with compatibility to our data.
    """

    # ---Seglearn functions---
    @staticmethod
    def mean_diff_func(x):
        return seg.mean_diff([x])[0]

    @staticmethod
    def mean_func(x):
        return seg.mean([x])[0]

    @staticmethod
    def median_func(x):
        return seg.median([x])[0]

    @staticmethod
    def abs_energy_func(x):
        return seg.abs_energy(np.array([list(x)]))[0]

    @staticmethod
    def std_func(x):
        return seg.std([x])[0]

    @staticmethod
    def var_func(x):
        return seg.var([x])[0]

    @staticmethod
    def min_func(x):
        return seg.minimum([x])[0]

    @staticmethod
    def max_func(x):
        return seg.maximum([x])[0]

    @staticmethod
    def skew_func(x):
        return seg.skew([x])[0]

    @staticmethod
    def kurt_func(x):
        return seg.kurt([x])[0]

    @staticmethod
    def mse_func(x):
        return seg.mse([x])[0]

    @staticmethod
    def mnx_func(x):
        return seg.mean_crossings([x])[0]

    @staticmethod
    def mean_abs_func(x):
        return seg.mean_abs([x])[0]

    @staticmethod
    def zero_cross_func(x):
        return seg.zero_crossing().__call__(np.array([list(x)]))[0]

    @staticmethod
    def slope_sign_func(x):
        return seg.slope_sign_changes().__call__(np.array([list(x)]))[0]

    @staticmethod
    def waveform_length_func(x):
        return seg.waveform_length([x])[0]

    @staticmethod
    def integrated_emg_func(x):
        return seg.abs_sum([x])[0]

    @staticmethod
    def emg_var_func(x):
        return seg.emg_var(np.array([list(x)]))[0]

    @staticmethod
    def root_mean_square_func(x):
        return seg.root_mean_square(np.array([list(x)]))[0]

    @staticmethod
    def willison_amplitude_func(x):
        return seg.willison_amplitude().__call__(np.array([list(x)]))[0]

    # ---Tsfresh functions---
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


class Utils:
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
            data_list = list(tqdm(pool.starmap(Utils.read_file, [(path, file) for file in files[:2]])))
        data = pd.concat(data_list)
        data.reset_index(drop=True, inplace=True)
        return data

    @staticmethod
    def find_variable_name(variable):
        for name, value in globals().items():
            if value is variable:
                return name

    @staticmethod
    def inspect_data(data):
        name = Utils.find_variable_name(data)
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

    @staticmethod
    def reduce_memory_usage(df):
        # I took this function from kaggle too.
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype.name
            if (col_type != 'datetime64[ns]') & (col_type != 'category'):
                if col_type != 'object':
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


class TrainingDataPreparation:
    def __init__(self, data_path, converters_path, main_config, step=1):
        self.data_path = data_path
        self.converters_path = converters_path
        self.main_config = main_config
        self.tfog_data = None
        self.defog_data = None
        self.data = None
        self.x = None
        self.y = None
        self.target_cols = ['StartHesitation', 'Turn', 'Walking', 'Normal']
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler((0, 1))
        self.step = step  # this will be used for rolling window stride size

    def run(self):
        self.tfog_data = Utils.read_files_parallel(self.data_path + 'train/tdcsfog/')
        self.defog_data = Utils.read_files_parallel(self.data_path + 'train/defog/')
        self.defog_data.drop(columns=['Valid', 'Task'], inplace=True)
        self.prepare_data()
        self.generate_normal_col()
        self.engineer_features()
        self.prepare_model_data()

    def generate_normal_col(self):
        self.data['Normal'] = ((self.data['StartHesitation'] == 0) & (self.data['Turn'] == 0) &
                               (self.data['Walking'] == 0)).astype(int)

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
    def rolling_operation(data, column, window, func, step):
        # we create rolling features and rename the columns with the name of the function
        return data[column].rolling(window, step=step).apply(func, raw=True).rename(
            f'{column}_{window}_{func.__name__}')

    @function_timer.__func__
    def generate_rolling_columns(self):
        # this function uses multiprocessing to create rolling features faster.
        # I add all combinations to a list then multiprocess them with starmap function.
        starmap_args = []
        for func, rolling_config in self.main_config.items():
            for column, windows in rolling_config.items():
                for window in windows:
                    starmap_args.append((self.data, column, window, func, self.step))
        with Pool(cpu_count()) as pool:
            manipulated_data = list(pool.starmap(TrainingDataPreparation.rolling_operation, starmap_args))
            manipulated_data = pd.concat(manipulated_data, axis=1)
            self.data = self.data.merge(manipulated_data, left_index=True, right_index=True, how='outer')

    @staticmethod
    def detect_steps(signal, frequency=128, smoothing=2, faster_please=False, min_scale_log=3.6, max_scale_log=5.4):
        """
        This function will locate and visualize individual steps and
        construct the "step rate" and "current step duration" features.
        So I took this from VRBARYSHEV's notebook in kaggle.
        """
        scales = np.exp(np.arange(min_scale_log, max_scale_log, 0.05))
        wavelet = 'morl'  # chosing the Morlet wavelet

        coeff, freq = pywt.cwt(signal, scales, wavelet)
        coeff_argmax_index = np.argmax(abs(coeff), 0)
        coeff_max = np.array([coeff[coeff_argmax_index[i], i] for i in range(coeff.shape[1])])

        for i in range(smoothing):
            coeff, freq = pywt.cwt(coeff_max, scales, wavelet)
            coeff_argmax_index = np.argmax(abs(coeff), 0).astype(int)
            coeff_argmax_index = np.round(
                pd.Series(coeff_argmax_index).rolling(128, center=True, min_periods=1).median()).astype(int)
            coeff_max = np.array([coeff[coeff_argmax_index[i], i] for i in range(coeff.shape[1])])

        if not faster_please:
            max_cwt_points = sp.signal.find_peaks(abs(coeff_max), distance=20, width=20)[0]
            max_cwt_points = np.concatenate(([0], max_cwt_points, [len(signal) - 1]))
            max_cwt_line_indexes = np.round(np.interp(range(0, len(signal)), max_cwt_points,
                                                      coeff_argmax_index[max_cwt_points])).astype(int)
            coeff_max = np.array([coeff[max_cwt_line_indexes[i], i] for i in range(coeff.shape[1])])

        zero_crossings = \
            np.where(np.diff(np.sign(pd.Series(coeff_max).rolling(10, center=True, min_periods=1).mean())))[0]
        zero_crossings = np.concatenate(([0], zero_crossings, [len(signal)]))
        step_lengths = []
        for i in range(1, len(zero_crossings)):
            step_lengths = np.concatenate(
                (step_lengths,
                 [zero_crossings[i] - zero_crossings[i - 1]] * (zero_crossings[i] - zero_crossings[i - 1])))
        step_durations = pd.Series(step_lengths).rolling(32, center=True, min_periods=1).median()
        step_rate = pd.Series(1. / step_durations) * frequency
        step_rate = step_rate.where(step_rate < 5, 0).rolling(frequency, center=True, min_periods=1).mean()

        return step_rate, step_durations, zero_crossings

    def prepare_meta(self):
        subjects = pd.read_csv(self.data_path + 'subjects.csv')
        tfog_meta = pd.read_csv(self.data_path + 'tdcsfog_metadata.csv')
        defog_meta = pd.read_csv(self.data_path + 'defog_metadata.csv')

        subjects.drop(columns=['Visit'], inplace=True)
        subjects.drop_duplicates(subset='Subject', inplace=True)

        tfog_meta = pd.merge(tfog_meta, subjects, on=['Subject'], how='left')
        defog_meta = pd.merge(defog_meta, subjects, on=['Subject'], how='left')
        return tfog_meta, defog_meta

    def prepare_data(self):
        tfog_meta, defog_meta = self.prepare_meta()
        self.tfog_data = pd.merge(self.tfog_data, tfog_meta, left_on=['session_id'], right_on=['Id'], how='left')
        self.defog_data = pd.merge(self.defog_data, defog_meta, left_on=['session_id'], right_on=['Id'], how='left')

        self.data = pd.concat([self.tfog_data, self.defog_data], axis=0, ignore_index=True)

        del self.tfog_data, self.defog_data

        self.data.drop(columns=['Id', 'Subject', 'Medication', 'YearsSinceDx', 'UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ'],
                       inplace=True)
        self.data['Id'] = self.data['session_id'] + '_' + self.data['Time'].astype(str)
        self.data.rename(columns={'Visit_x': 'Visit'}, inplace=True)
        self.data['Sex'] = self.data['Sex'].map({'F': 0, 'M': 1}).fillna(1)

    def generate_fft_cols(self):
        fft_functions = ['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'dct', 'idct', 'dst', 'idst']
        print('Creating fft columns')
        for func_name in tqdm(fft_functions):
            fft_data = pd.DataFrame()
            # Get the function from the scipy.fft module
            func = getattr(sp.fft, func_name)
            fft_data[f'{func_name}_v'] = np.abs(func(self.data['AccV'].to_numpy(), norm='ortho'))
            fft_data[f'{func_name}_ml'] = np.abs(func(self.data['AccML'].to_numpy(), norm='ortho'))
            fft_data[f'{func_name}_ap'] = np.abs(func(self.data['AccAP'].to_numpy(), norm='ortho'))
            fft_data[f'{func_name}_total'] = np.abs(func(self.data['total_acceleration'].to_numpy(), norm='ortho'))

            self.data = self.data.merge(fft_data, left_index=True, right_index=True, how='left')

    def engineer_features(self):
        self.generate_rolling_columns()
        # I didn't use zero_crossings because its shape smaller than the main data.
        # Up-scaling this data is inefficient for score.
        self.data['acc_ml_step_rate'], self.data['acc_ml_step_durations'], _ = \
            TrainingDataPreparation.detect_steps(self.data['AccML'])
        self.data['total_acceleration'] = np.sqrt(self.data['AccV'] ** 2 + self.data['AccML'] ** 2 + self.data['AccAP'] ** 2)
        self.generate_fft_cols()
        self.data.fillna(method='bfill', inplace=True)

    def prepare_model_data(self):
        self.data.drop(columns=['Id', 'Time', 'Test'], inplace=True)
        self.data.dropna(inplace=True)
        sessions = self.data['session_id'].unique()
        session_labels = {value: key for key, value in dict(enumerate(sessions)).items()}
        self.data['session_id'] = self.data['session_id'].map(session_labels).fillna(999)
        self.data['session_id'] = np.sin(self.data['session_id'])
        # model can extract session-based info, so we will use session labels in inference.
        joblib.dump(session_labels, self.converters_path + 'sessions_dict.joblib')

        self.x = self.data.drop(columns=self.target_cols).copy()

    def prepare_data4_booster(self):
        # we reduce multiple target columns into 1 column for booster models by converting them to labels
        self.y = self.data[self.target_cols].idxmax(axis=1)

    def prepare_data4_nn(self):
        self.y = self.data[self.target_cols].to_numpy()

        self.x = self.standard_scaler.fit_transform(self.x)
        self.x = self.minmax_scaler.fit_transform(self.x)
        # Scalers will be used in inference
        joblib.dump({'standard_scaler': self.standard_scaler, 'minmax_scaler': self.minmax_scaler},
                    self.converters_path + 'scalers.joblib')

    def prepare_data4_rnn(self, sequence_length):
        self.prepare_data4_nn()

        num_rows = self.x.shape[0]
        trim_size = num_rows % sequence_length
        # we divide the data into time sequences as desired
        # part longer than the last sequence is trimmed
        if trim_size != 0:
            self.x = self.x[:-trim_size]
            self.y = self.y[:-trim_size]

        self.x = np.reshape(self.x, (-1, sequence_length, self.x.shape[1]))
        self.y = np.reshape(self.y, (-1, sequence_length, self.y.shape[1]))


class InferenceDataPreparation(TrainingDataPreparation):
    def __init__(self, data_path, converters_path, main_config):
        super().__init__(data_path, converters_path, main_config)
        self.tfog_test_data = None
        self.defog_test_data = None
        self.ids = None
        self.session_labels = joblib.load(self.converters_path + 'sessions_dict.joblib')

    def run(self):
        self.tfog_data = Utils.read_files_parallel(self.data_path + 'test/tdcsfog/')
        self.defog_data = Utils.read_files_parallel(self.data_path + 'test/defog/')
        self.prepare_data()
        self.engineer_features()
        self.prepare_model_data()

    def prepare_model_data(self):
        self.data.drop(columns=['Time', 'Test'], inplace=True)
        self.data.dropna(inplace=True)
        self.ids = self.data['Id'].copy()
        self.data.drop(columns=['Id'], inplace=True)
        # I labeled the session_id as other with the number 999 which my model never saw
        self.data['session_id'] = self.data['session_id'].map(self.session_labels).fillna(999)
        self.data['session_id'] = np.sin(self.data['session_id'])

    def prepare_data4_nn(self):
        scalers = joblib.load(self.converters_path + 'scalers.joblib')
        self.data = self.data[scalers['standard_scaler'].get_feature_names_out()]
        
        self.data = scalers['standard_scaler'].transform(self.data)
        self.data = scalers['minmax_scaler'].transform(self.data)

    def prepare_data4_rnn(self, sequence_length):
        self.prepare_data4_nn()

        num_rows = self.data.shape[0]
        trim_size = num_rows % sequence_length
        # we divide the data into time sequences as desired
        # part longer than the last sequence is trimmed
        if trim_size != 0:
            self.data = self.data[:-trim_size]
            self.ids = self.ids[:-trim_size]

        self.data = np.reshape(self.data, (-1, sequence_length, self.data.shape[1]))


class Inference:
    def __init__(self, inference_creator, model_path):
        self.inference_creator = inference_creator
        self.model_path = model_path
        self.ids = inference_creator.ids.copy()
        self.predictions = None

    def run_single_booster(self):
        model = joblib.load(self.model_path + 'lgb_model.joblib')
        self.predictions = model.predict_proba(self.inference_creator.data)
        self.predictions = pd.DataFrame(self.predictions, columns=model.classes_, index=self.ids.index)
        self.predictions['Id'] = self.ids

    def run_multi_booster(self):
        multi_model_path = os.listdir(self.model_path)
        models = [joblib.load(self.model_path + f'{model}') for model in multi_model_path if model.startswith('lgb_multi')]
        self.inference_creator.data.reset_index(drop=True, inplace=True)
        splitter = KFold(n_splits=len(models), shuffle=True, random_state=42)
        self.predictions = list()
        indexes = list()
        for model, (_, train_index) in tqdm(zip(models, splitter.split(self.inference_creator.data))):
            self.predictions.append(model.predict_proba(self.inference_creator.data.loc[train_index]))
            indexes.append(train_index)
        indexes = np.concatenate(indexes, axis=0)
        self.predictions = np.concatenate(self.predictions, axis=0)
        self.predictions = pd.DataFrame(self.predictions, columns=model.classes_, index=indexes)
        self.predictions['Id'] = self.ids.reset_index(drop=True)

    def run_single_nn(self):
        self.inference_creator.prepare_data4_nn()
        model = load_model(self.model_path + 'tf_nn_model.h5')
        self.predictions = model.predict(self.inference_creator.data)
        self.predictions = pd.DataFrame(self.predictions, columns=self.inference_creator.target_cols, index=self.ids.index)
        self.predictions['Id'] = self.ids

    def run_multi_nn(self):
        self.inference_creator.prepare_data4_nn()
        multi_model_path = os.listdir(self.model_path)
        models = [load_model(self.model_path + f'{model}') for model in multi_model_path if model.startswith('tf_nn_multi')]
        splitter = KFold(n_splits=len(models), shuffle=True, random_state=42)
        self.predictions = list()
        indexes = list()
        for model, (_, train_index) in tqdm(zip(models, splitter.split(self.inference_creator.data))):
            self.predictions.append(model.predict(self.inference_creator.data[train_index]))
            indexes.append(train_index)
        indexes = np.concatenate(indexes, axis=0)
        self.predictions = np.concatenate(self.predictions, axis=0)

        self.predictions = pd.DataFrame(self.predictions, columns=self.inference_creator.target_cols, index=indexes)
        self.predictions['Id'] = self.ids.reset_index(drop=True)

    def run_single_rnn(self, sequence_length):
        self.inference_creator.prepare_data4_rnn(sequence_length)
        model = load_model(self.model_path + 'tf_rnn_model.h5')
        self.predictions = model.predict(self.inference_creator.data)
        self.predictions = np.reshape(self.predictions, (-1, self.predictions.shape[2]))
        self.predictions = pd.DataFrame(self.predictions, columns=self.inference_creator.target_cols, index=self.ids.index)
        self.predictions['Id'] = self.ids

    def run_multi_rnn(self, sequence_length):
        self.inference_creator.prepare_data4_rnn(sequence_length)
        multi_model_path = os.listdir(self.model_path)
        models = [load_model(self.model_path + f'{model}') for model in multi_model_path if model.startswith('tf_rnn_multi')]
        splitter = KFold(n_splits=len(models), shuffle=False)
        self.predictions = list()
        for model, (_, train_index) in tqdm(zip(models, splitter.split(self.inference_creator.data))):
            preds = model.predict(self.inference_creator.data[train_index])
            self.predictions.append(np.reshape(preds, (-1, preds.shape[2])))

        self.predictions = np.concatenate(self.predictions, axis=0)

        self.predictions = pd.DataFrame(self.predictions, columns=self.inference_creator.target_cols)
        self.predictions['Id'] = self.ids.reset_index(drop=True)

