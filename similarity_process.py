#### when we concat the similarity scores, we made a mistake.
#### we should concat the similarity scores according to the groupby('cve') of data.

import os
import pandas as pd
from utils import reduce_mem_usage
import gc

DATA_DIR = '/data/kaixuan/ramdisk/data'
DATA_TMP_DIR = '/data/kaixuan/data_tmp'

# Load data

similarity_df = pd.read_csv(os.path.join(DATA_TMP_DIR, 'similarity_data.csv'))
reduce_mem_usage(similarity_df)
similarity_scores = similarity_df['similarity']

commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_info.csv'))
reduce_mem_usage(commit_data)
desc_data = pd.read_csv(os.path.join(DATA_DIR, 'cve_desc.csv'))

# Merge commit_data and desc_data on 'cve' column
data = pd.merge(commit_data, desc_data, on='cve', how='left')

# Reduce memory usage
reduce_mem_usage(data)

data = data.drop(columns=['cve_desc', 'msg', 'diff'])

del commit_data, desc_data
gc.collect()

data_group = data.groupby('cve')

print("shape of data: ", data.shape)
print("shape of similarity_scores: ", similarity_scores.shape)

# Reset the index of data (to ensure it starts from 0)
data.reset_index(drop=True, inplace=True)

# Add similarity scores to data
data['similarity'] = similarity_scores.values

# Save the data to csv
data.to_csv(os.path.join(DATA_TMP_DIR, 'data_group.csv'), index=False)
