import nltk
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tokenize_cpp_code import tokenize_cpp_code
from utils import reduce_mem_usage

DATA_DIR = '/mnt/local/Baselines_Bugs/CodeBert/data/'

# Load data
commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_info.csv')).head(100)
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

# Compute TF-IDF vectors for tokenized cve_desc and combined info
vectorizer = TfidfVectorizer()
desc_tfidf = vectorizer.fit_transform(data['desc_token'])
combined_tfidf = vectorizer.transform(data['combined'])

# Calculate the similarity for each row
similarity_scores = cosine_similarity(desc_tfidf, combined_tfidf).diagonal()

# Save the required information in a CSV file
similarity_data = pd.DataFrame()
similarity_data['cve'] = data['cve']
similarity_data['owner'] = data['owner']
similarity_data['repo'] = data['repo']
similarity_data['commit_id'] = data['commit_id']
similarity_data['similarity'] = similarity_scores
similarity_data['label'] = data['label']

# Sort by CVE
similarity_data = similarity_data.sort_values(by=['cve'])

# Save to CSV
similarity_data.to_csv(os.path.join(DATA_DIR, 'similarity_data.csv'), index=False)
