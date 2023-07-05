import nltk
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tokenize_cpp_code import tokenize_cpp_code
from utils import reduce_mem_usage
from tqdm import tqdm

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
data['desc_token'] = data['cve_desc'].apply(nltk.word_tokenize).apply(' '.join)
data['msg_token'] = data['msg'].apply(nltk.word_tokenize).apply(' '.join)
data['diff_token'] = data['diff'].apply(tokenize_cpp_code).apply(' '.join)

# Combine tokenized commit messages and diffs
data['combined'] = data['msg_token'] + " " + data['diff_token']

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer()

similarity_scores = []
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    vectorizer.fit([row['desc_token'], row['combined']])
    desc_tfidf = vectorizer.transform([row['desc_token']])
    combined_tfidf = vectorizer.transform([row['combined']])
    similarity_scores.append(cosine_similarity(desc_tfidf, combined_tfidf).diagonal()[0])

similarity_scores = np.array(similarity_scores)

# vectorizer.fit(data['desc_token'])  # Fit on description text data
# desc_tfidf = vectorizer.transform(data['desc_token'])
# combined_tfidf = vectorizer.transform(data['combined'])
# # Calculate the similarity for each row
# similarity_scores = cosine_similarity(desc_tfidf, combined_tfidf).diagonal()

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
