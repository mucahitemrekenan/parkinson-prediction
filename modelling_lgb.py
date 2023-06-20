from lightgbm.sklearn import LGBMClassifier
import joblib
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
    trainer.prepare_data4_booster()
    Utils.inspect_data(trainer.data)

    lgb = LGBMClassifier(n_estimators=5000, max_depth=6, n_jobs=-1)
    lgb.fit(trainer.x, trainer.y)
    joblib.dump(lgb, 'models/lgb_model.joblib')

    my_dict = {key: value for key, value in zip(lgb.feature_name_, lgb.feature_importances_)}
    sorted_d = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
    print(sorted_d)

