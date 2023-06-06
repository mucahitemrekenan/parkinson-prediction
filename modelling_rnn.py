import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import GlorotUniform, RandomNormal
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from base import *



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

data = pd.read_csv('data/engineered_data.csv', low_memory=False)
trainer = Training(data_path='data/', converters_path='converters/', models_path='models/', main_config=main_config)
trainer.data = data
trainer.prepare_data()
trainer.prepare_data4_rnn(sequence_length=512)
Training.inspect_data(trainer.data)

# class_weights = np.array([(20_000_000 - 305290) / 20_000_000,
#                           (20_000_000 - 2247047) / 20_000_000,
#                           (20_000_000 - 300239) / 20_000_000,
#                           (20_000_000 - 17735798) / 20_000_000])
# sample_weights = np.zeros((y.shape[0], y.shape[1]))
#
# for i in range(y.shape[0]):
#     for j in range(y.shape[1]):
#         sample_weights[i, j] = class_weights[np.argmax(y[i, j])]


with tf.device('/GPU:0'):
    model = Sequential()
    model.add(LSTM(64, kernel_initializer=RandomNormal(), input_shape=trainer.x.shape[1:], activation='relu', return_sequences=True))
    model.add(LSTM(64, kernel_initializer=RandomNormal(), activation='relu', return_sequences=True))
    # model.add(LSTM(64, kernel_initializer=RandomNormal(), activation='relu', return_sequences=True))
    # model.add(LSTM(64, kernel_initializer=RandomNormal(), activation='relu', return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(trainer.y.shape[2], kernel_initializer=RandomNormal(), activation='sigmoid')))
    model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy',
                  metrics='categorical_accuracy')
    model.summary()
    model.fit(trainer.x, trainer.y, epochs=1, batch_size=1024)#, sample_weight=sample_weights

# 'categorical_crossentropy'
# 'categorical_accuracy'
# AveragePrecision(y.shape[1])
# tf.keras.losses.BinaryCrossentropy()
weights = model.layers[0].get_weights()
weights2 = model.layers[1].get_weights()

# scores = model.evaluate(trainer.x, trainer.y, verbose=1)
# predictions = model.predict(x)
# sub = np.reshape(predictions, (-1, y.shape[1]))

model.save('models/tf_rnn_model.h5')

