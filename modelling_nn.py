from sklearn.metrics import average_precision_score
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import GlorotUniform, RandomNormal
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import tensorflow as tf
from base import TrainingDataPreparation, Utils
from config import Config
from multiprocessing.spawn import freeze_support


# multiprocessing may throw exceptions thus we use freeze_support
if __name__ == '__main__':
    freeze_support()

    training_config = Config()
    training_config.create_main_config()

    trainer = TrainingDataPreparation(data_path='data/', converters_path='converters/',
                                      main_config=training_config.main_config, step=1)
    trainer.run()
    Utils.inspect_data(trainer.data)
    trainer.prepare_data4_nn()


    with tf.device('/GPU:0'):
        model = Sequential()
        model.add(Dense(64, kernel_initializer=RandomNormal(), input_shape=trainer.x.shape[1:], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16, kernel_initializer=RandomNormal(), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, kernel_initializer=RandomNormal(), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(trainer.y.shape[1], kernel_initializer=RandomNormal(), activation='sigmoid'))
        model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics='categorical_accuracy')
        model.summary()
        model.fit(trainer.x, trainer.y, epochs=3, batch_size=1024)


    predictions = model.predict(trainer.x)
    print(average_precision_score(trainer.y, predictions))

    model.save('models/tf_nn_model.h5')



