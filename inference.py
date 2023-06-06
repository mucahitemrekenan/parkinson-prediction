from base import *
from tensorflow.keras.models import load_model
from multiprocessing.spawn import freeze_support


# if __name__ == '__main__':
#     freeze_support()

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

inference = Inference(data_path='data/', converters_path='converters/', models_path='models/', main_config=main_config)
inference.run()
inference.prepare_data()

model_type = 'rnn'
single = False
path = 'models/'

if model_type == 'booster':
    if single:
        model = joblib.load(path+'lgb_model.joblib')
        predictions = model.predict_proba(inference.data)
    else:
        model_path = os.listdir(path)
        models = [joblib.load(path+f'{model}') for model in model_path if model.startswith('lgb_multi')]
        predictions = list()
        for model in models:
            predictions.append(model.predict_proba(inference.data))
        predictions = np.mean(predictions, axis=0)

elif model_type == 'nn':
    inference.prepare_data4_nn()
    if single:
        model = load_model(path+'tf_nn_model.h5')
        predictions = model.predict(inference.data)
    else:
        model_path = os.listdir(path)
        models = [load_model(path+f'{model}') for model in model_path if model.startswith('tf_nn_multi')]
        predictions = list()
        for model in models:
            predictions.append(model.predict(inference.data))
        predictions = np.mean(predictions, axis=0)

elif model_type == 'rnn':
    inference.prepare_data4_rnn(128)
    if single:
        model = load_model(path+'tf_rnn_model.h5')
        predictions = model.predict(inference.data)
        predictions = np.reshape(predictions, (-1, predictions.shape[2]))
    else:
        model_path = os.listdir(path)
        models = [load_model(path+f'{model}') for model in model_path if model.startswith('tf_rnn_multi')]
        predictions = list()
        for model in models:
            predictions.append(model.predict(inference.data))
        predictions = np.mean(predictions, axis=0)
        predictions = np.reshape(predictions, (-1, predictions.shape[2]))

predictions = pd.DataFrame(predictions, columns=inference.target_cols, index=inference.ids.index)
predictions['Id'] = inference.ids

sample_submission = pd.read_csv('data/sample_submission.csv')
submission = pd.merge(sample_submission[['Id']], predictions[['Id', 'StartHesitation', 'Turn', 'Walking']], how='left', on='Id')
print('null counts of submission before fillna:', submission.isnull().sum())
submission = submission.fillna(0.0).round(4)
submission.to_csv('submission.csv', index=False)
Training.inspect_data(submission)
