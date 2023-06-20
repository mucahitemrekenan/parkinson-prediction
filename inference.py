from base import InferenceDataPreparation, Utils, Inference
from config import Config
import pandas as pd
from multiprocessing.spawn import freeze_support


# multiprocessing may throw exceptions thus we use freeze_support
if __name__ == '__main__':
    freeze_support()

    inference_config = Config()
    inference_config.create_main_config()

    pre_inference = InferenceDataPreparation(data_path='data/', converters_path='converters/',
                                             main_config=inference_config.main_config)
    pre_inference.run()
    Utils.inspect_data(pre_inference.data)

    inference = Inference(pre_inference, model_path='models/')

    model_type = 'nn'
    single = True

    if model_type == 'booster':
        if single:
            inference.run_single_booster()
        else:
            inference.run_multi_booster()

    elif model_type == 'nn':
        if single:
            inference.run_single_nn()
        else:
            inference.run_multi_nn()

    elif model_type == 'rnn':
        if single:
            inference.run_single_rnn(sequence_length=512)
        else:
            inference.run_multi_rnn(sequence_length=512)

    sample_submission = pd.read_csv('data/sample_submission.csv')
    submission = pd.merge(sample_submission[['Id']], inference.predictions[['Id', 'StartHesitation', 'Turn', 'Walking']],
                          how='left', on='Id')
    print('null counts of submission before fillna:', submission.isnull().sum())
    submission = submission.fillna(0.0).round(4)
    submission.to_csv('submission.csv', index=False)

    Utils.inspect_data(submission)
