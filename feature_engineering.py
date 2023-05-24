import numpy as np
from seglearn.feature_functions import *
from tsflex.features import FeatureDescriptor, FeatureCollection, MultipleFeatureDescriptors
from tsflex.features.integrations import seglearn_feature_dict_wrapper
from project_functions import *
import dask.dataframe as dd
from time import time


tfog_data = read_files_parallel('data/train/tdcsfog/')
defog_data = read_files_parallel('data/train/defog/')



rename_cols(data)
generate_normal_col(data)

rollingw_config = {'accv': [4, 8, 24, 48, 96, 128, 256, 512, 1024]}
start = time()
new_data = generate_rollingw_columns(data, rollingw_config, np.mean)
end = time()
print(end - start)


rollingw_config = {'accv': [4, 8, 24, 48, 96, 128, 256, 512, 1024],
                   'accml': [4, 8, 24, 48, 96, 128, 256, 512, 1024],
                   'accap': [4, 8, 24, 48, 96, 128, 256, 512, 1024]}
start = time()
new_data = generate_rollingw_columns(data, rollingw_config, np.mean)
end = time()
print(end - start)
