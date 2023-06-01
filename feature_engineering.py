from multiprocessing.spawn import freeze_support
from project_functions import *

# if __name__ == '__main__':
#     freeze_support()

tfog_data = read_files_parallel('data/train/tdcsfog/')
defog_data = read_files_parallel('data/train/defog/')
# notype_data = read_files_parallel('data/train/notype/')

# defog_data = defog_data[(defog_data['Valid'] == 1) & (defog_data['Task'] == 1)]
# defog_data.drop(columns=['Valid', 'Task'], inplace=True)

data = pd.concat([tfog_data, defog_data], axis=0)
del tfog_data, defog_data

if ('Valid' in data.columns) & ('Task' in data.columns):
    data.drop(columns=['Valid', 'Task'], inplace=True)

generate_normal_col(data)

main_config = {
    abs_energy: {'AccV': [25],
                 'AccML': [32],
                 'AccAP': [50]},

    linear_trend_func: {'AccV': [5, 20],
                        'AccML': [10, 25],
                        'AccAP': [15, 50]},

    autocorrelation_func: {'AccV': [5, 25],
                           'AccML': [10, 50],
                           'AccAP': [15, 100]},

    mean_diff_func: {'AccV': [2],
                     'AccML': [5],
                     'AccAP': [10]}
}

engineered_data = generate_rollingw_columns(data, main_config)
engineered_data.dropna(inplace=True)
engineered_data.to_csv('data/engineered_data.csv', index=False)
