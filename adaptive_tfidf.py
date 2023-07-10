import os
from tqdm import tqdm
import pandas as pd

DATA_DIR = '/data/kaixuan/ramdisk/data'
DATA_TMP_DIR = '/data/kaixuan/data_tmp'

### read tfidf similarity data
tfidf_data = pd.read_csv(os.path.join(DATA_DIR, 'similarity_data_1.csv'))
tfidf_cve = tfidf_data.groupby('cve')

### normalize the similarity score
for cve, group in tqdm(tfidf_cve):
    tfidf_data.loc[group.index, 'similarity'] = (group['similarity'] - group['similarity'].min()) / \
    (group['similarity'].max() - group['similarity'].min())


### 
