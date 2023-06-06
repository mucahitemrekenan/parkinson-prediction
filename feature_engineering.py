from multiprocessing.spawn import freeze_support
from base import *


# if __name__ == '__main__':
#     freeze_support()

# if ('Valid' in data.columns) & ('Task' in data.columns):
#     data.drop(columns=['Valid', 'Task'], inplace=True)


main_config = {
    abs_energy: {'AccV': [25],
                 'AccML': [32],
                 'AccAP': [50]},

    Training.linear_trend_func: {'AccV': [5, 20],
                                 'AccML': [10, 25],
                                 'AccAP': [15, 50]},

    Training.autocorrelation_func: {'AccV': [5, 25],
                                    'AccML': [10, 50],
                                    'AccAP': [15, 100]},

    Training.mean_diff_func: {'AccV': [2],
                              'AccML': [5],
                              'AccAP': [10]}
}

trainer = Training(data_path='data/', converters_path='converters/', models_path='models/', main_config=main_config)
trainer.run()