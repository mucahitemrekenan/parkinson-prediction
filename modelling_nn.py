import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
# from sklearnex  import patch_sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import average_precision_score
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from project_functions import *


# patch_sklearn()
# matplotlib.use('Qt5Agg')

# data = read_files_parallel('data/train/tdcsfog/')
data = pd.read_csv('engineered_data.csv')

print(data.dtypes)

target_cols = ['starth', 'turn', 'walk', 'normal']

normal = data[data['normal'] == 1].sample(frac=0.1).index
turn = data[data['turn'] == 1].sample(frac=0.25).index
starth = data[data['starth'] == 1].sample(frac=1).index
walk = data[data['walk'] == 1].sample(frac=1).index
print([len(x) for x in [normal, turn, starth, walk]])

filter_index = pd.concat([normal.to_series(), turn.to_series(), starth.to_series(), walk.to_series()], axis=0)
filter_index = pd.Index(filter_index)

filtered_data = data[data.index.isin(filter_index)].copy()
filtered_data.dropna(inplace=True)

x = filtered_data.drop(columns=['time', 'starth', 'turn', 'walk', 'session_id', 'normal']).copy()
y = filtered_data[target_cols].copy().to_numpy()

x_cols = x.columns
print(x.isnull().sum())

x = StandardScaler().fit_transform(x)
x = MinMaxScaler((0, 1)).fit_transform(x)

# for col in range(x.shape[1]):
#     plt.plot(x[:,col])
#     plt.show()

x = x[:, :3]

num_rows = x.shape[0]
sequence_length = 512

# Calculate the required padding
trim_size = num_rows % sequence_length

# if trim_size != 0:
#     x = x[:-trim_size]
#     y = y[:-trim_size]
#
# x = np.reshape(x, (-1, sequence_length, x.shape[1]))
# y = np.reshape(y, (-1, sequence_length, y.shape[1]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

tensorboard_callback = TensorBoard(log_dir="./logs")
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
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, epochs=3, batch_size=512, validation_split=0.2, callbacks=[tensorboard_callback])

weights = model.layers[0].get_weights()
weights2 = model.layers[1].get_weights()

scores = model.evaluate(x_test, y_test, verbose=1)
predictions = model.predict(x_test)
sub = np.reshape(predictions, (-1, y_train.shape[1]))

for num, col in zip(range(5), ['starth', 'turn', 'walk', 'normal']):
    plt.plot(sub[:, num])
    plt.title(col)
    plt.show()

model.save('tf_model.h5')

# tensorboard --logdir ./logs --purge_orphaned_data true
# rm -r ./logs/*
