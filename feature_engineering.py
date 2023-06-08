from multiprocessing.spawn import freeze_support
from base import *

# if __name__ == '__main__':
#     freeze_support()

# if ('Valid' in data.columns) & ('Task' in data.columns):
#     data.drop(columns=['Valid', 'Task'], inplace=True)
window = 100
main_config = {

    Training.mean_func: {'AccV': [window]},

    Training.median_func: {'AccAP': [window]},

    Training.abs_energy_func: {
                               'AccML': [window]
                               },

    Training.std_func: {
                        'AccAP': [window]},

    Training.var_func: {'AccV': [window]},

    Training.min_func: {

                        'AccAP': [window]},

    Training.max_func: {'AccV': [window]

                        },

    Training.skew_func: {
                         'AccML': [window]
                         },

    Training.kurt_func: {
                         'AccAP': [window]},

    Training.mse_func: {'AccV': [window]

                        },

    Training.mnx_func: {
                        'AccML': [window]
                        },

    Training.mean_abs_func: {
                             'AccAP': [window]},

    Training.zero_cross_func: {'AccV': [window],
                               },

    Training.slope_sign_func: {
                               'AccML': [window]
                               },

    Training.waveform_length_func: {
                                    'AccAP': [window]},

    Training.integrated_emg_func: {'AccV': [window]
                                   },

    Training.emg_var_func: {
                            'AccML': [window],
                            },

    Training.root_mean_square_func: {
                                     'AccML': [window],
                                     },

    Training.willison_amplitude_func: {
                                       'AccAP': [window]}
}

trainer = Training(data_path='data/', converters_path='converters/', models_path='models/', main_config=main_config)
trainer.run()

