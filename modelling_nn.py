import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from data2_model_adapter import prepare_data4_nn


x, y = prepare_data4_nn('engineered_data2.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create LSTM model with Sequential API
with tf.device('/GPU:0'):
    model = Sequential()
    model.add(Dense(64, kernel_initializer=GlorotUniform(), input_shape=x_train.shape[1:], activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer=GlorotUniform(), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer=GlorotUniform(), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer=GlorotUniform(), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], kernel_initializer=GlorotUniform(), activation='softmax'))
    model.compile(optimizer=SGD(learning_rate=0.01), loss="categorical_crossentropy", metrics=['AUC'])
    model.summary()
    model.fit(x_train, y_train, epochs=3, batch_size=512, validation_split=0.2)

weights = model.layers[0].get_weights()
weights2 = model.layers[1].get_weights()

scores = model.evaluate(x_test, y_test, verbose=1)
predictions = model.predict(x_test)
sub = np.reshape(predictions, (-1, y_train.shape[1]))

model.save('tf_model.h5')

