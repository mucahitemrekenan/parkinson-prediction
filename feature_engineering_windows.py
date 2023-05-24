from seglearn.feature_functions import base_features, emg_features
from multiprocessing import Pool, cpu_count
from project_functions import *


if __name__ == '__main__':
    target_cols = ['starth', 'turn', 'walk', 'normal']
    x_cols = ['accv', 'accml', 'accap']

    # setting train / test paths for tfog defog datasets
    data = read_files('data/train/tdcsfog/')
    rename_cols(data)
    generate_normal_col(data)

    with Pool(cpu_count()) as pool:
        # Calculate features for each window size in parallel
        results = pool.starmap(calculate_features,
                               [(window, data, base_features) for window in [2, 4, 8, 16, 32, 64, 128, 256, 512]])
        base_features_data = pd.concat(results, axis=1)

    with Pool(cpu_count()) as pool:
        # Calculate features for each window size in parallel
        results = pool.starmap(calculate_features,
                               [(window, data, emg_features) for window in [2, 4, 8, 16, 32, 64, 128, 256, 512]])
        emg_features_data = pd.concat(results, axis=1)

    engineered_data = pd.concat([base_features_data, emg_features_data], axis=1)
    engineered_data.to_csv('data/emg_base_features_tdcsfog_data.csv', index=False)

    print('base shape:', base_features_data.shape)
    print('emg shape:', emg_features_data.shape)
    print('engineered data shape', engineered_data.shape)
