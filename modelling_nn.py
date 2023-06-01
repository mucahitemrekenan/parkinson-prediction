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
from data2_model_adapter import prepare_data4_rnn


class AveragePrecision(tf.keras.metrics.Metric):

    def __init__(self, num_classes, thresholds=None, name='avg_precision', **kwargs):
        super(AveragePrecision, self).__init__(name=name, **kwargs)
        self.class_precision = [tf.keras.metrics.Precision(thresholds) for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i, precision in enumerate(self.class_precision):
            precision.update_state(y_true[..., i], y_pred[..., i])

    def result(self):
        return tf.math.reduce_mean([precision.result() for precision in self.class_precision])

    def reset_state(self):
        for precision in self.class_precision:
            precision.reset_state()


x, y = prepare_data4_rnn('data/engineered_data2.csv', sequence_length=512)
# x = pd.DataFrame(x)
# x.isnull().sum()
# np.isinf(x).any()

with tf.device('/GPU:0'):
    model = Sequential()
    model.add(LSTM(64, kernel_initializer=RandomNormal(), input_shape=x.shape[1:], activation='relu', return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, kernel_initializer=RandomNormal(), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, kernel_initializer=RandomNormal(), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, kernel_initializer=RandomNormal(), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(y.shape[1], kernel_initializer=RandomNormal(), activation='sigmoid')))
    model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy',
                  metrics='categorical_accuracy')
    model.summary()
    model.fit(x, y, epochs=1, batch_size=1024)

# 'categorical_crossentropy'
# 'categorical_accuracy'
# AveragePrecision(y.shape[1])
# tf.keras.losses.BinaryCrossentropy()
weights = model.layers[0].get_weights()
weights2 = model.layers[1].get_weights()

scores = model.evaluate(x, y, verbose=1)
predictions = model.predict(x)
sub = np.reshape(predictions, (-1, y.shape[1]))

model.save('tf_model.h5')

