import nltk
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tokenize_cpp_code import tokenize_cpp_code
from utils import reduce_mem_usage
from tqdm import tqdm

#### deprecated, since we split the process to first generate the tokenized data, and then load them
#### to tokenize the cve descriptions, commit messages, we still use this script.


nltk.download('punkt')

DATA_DIR = '/data/kaixuan/ramdisk/data'

# scp -P 32908 kaixuan_cuda11@XXXXXX:/mnt/local/Baselines_Bugs/CodeBert/data/cve_desc.csv ./data/

# sudo mkdir -p /data/kaixuan/ramdisk
# sudo mount -t tmpfs -o size=110G tmpfs /data/kaixuan/ramdisk
# mv ./data /data/kaixuan/ramdisk/

# sync
# sudo umount /data/kaixuan/ramdisk

# Load data
# commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_sample.csv')) ### for test
commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_info.csv'))
reduce_mem_usage(commit_data)
desc_data = pd.read_csv(os.path.join(DATA_DIR, 'cve_desc.csv'))

# Merge commit_data and desc_data on 'cve' column
data = pd.merge(commit_data, desc_data, on='cve', how='left')

# Reduce memory usage
reduce_mem_usage(data)

# Tokenization
print("Tokenizing...")
data['desc_token'] = data['cve_desc'].apply(nltk.word_tokenize).apply(' '.join)
data['desc_token'].to_csv(os.path.join(DATA_DIR, 'desc_token.csv'), index=False)
print("Tokenized CVE descriptions")
data = data.drop(columns=['cve_desc'])

### we need to process the cases there msg is null
data['msg'] = data['msg'].fillna(' ')
data['msg_token'] = data['msg'].apply(nltk.word_tokenize).apply(' '.join)
data['msg_token'].to_csv(os.path.join(DATA_DIR, 'msg_token.csv'), index=False)
print("Tokenized commit messages")

data = data.drop(columns=['msg'])


data['diff'] = data['diff'].fillna(' ')
data['diff_token'] = data['diff'].apply(tokenize_cpp_code).apply(' '.join)
data['diff_token'].to_csv(os.path.join(DATA_DIR, 'diff_token.csv'), index=False)
print("Tokenized diffs")
data = data.drop(columns=['diff'])

# Combine tokenized commit messages and diffs
print("Combining tokenized commit messages and diffs...")
data['combined'] = data['msg_token'] + " " + data['diff_token']

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer()

print("Computing TF-IDF vectors...")
similarity_scores = []

### 
data_cve = data.groupby('cve')

for cve, group in tqdm(data_cve, total=len(data_cve)):
    vectorizer.fit(group['combined'])
    for _, row in group.iterrows():
        desc_tfidf = vectorizer.transform([row['desc_token']])
        combined_tfidf = vectorizer.transform([row['combined']])
        similarity_scores.append(cosine_similarity(desc_tfidf, combined_tfidf).diagonal()[0])

similarity_scores = np.array(similarity_scores)

# Save the required information in a CSV file
similarity_data = pd.DataFrame()
similarity_data['cve'] = data['cve']
similarity_data['owner'] = data['owner']
similarity_data['repo'] = data['repo']
similarity_data['commit_id'] = data['commit_id']
similarity_data['similarity'] = similarity_scores
similarity_data['label'] = data['label']

# Sort by similarity score in descending order
similarity_data = similarity_data.sort_values(by=['similarity'], ascending=False)

# Save to CSV
similarity_data.to_csv(os.path.join(DATA_DIR, 'similarity_data.csv'), index=False)
print("Saved similarity_data.csv to {}".format(os.path.join(DATA_DIR, 'similarity_data.csv')))

### for test
# similarity_data.to_csv(os.path.join(DATA_DIR, 'similarity_test.csv'), index=False)
# print("Saved similarity_data.csv to {}".format(os.path.join(DATA_DIR, 'similarity_test.csv')))
