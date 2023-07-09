from tokenize_cpp_code_re import tokenize_cpp_code
from utils import reduce_mem_usage
from tqdm import tqdm
import os
import pandas as pd
import multiprocessing as mp
import ast
import csv ## fix the tokenization error



##### 2023.07.07
##### Since the diff is too large, we need to parallelize the tokenization process.
##### This script is used to tokenize the diff of each commit in parallel.

DATA_DIR = '/data/kaixuan/ramdisk/data'
DATA_TMP_DIR = '/data/kaixuan/data_tmp' ### since the ramdisk is 110G, we should save the tokenized data to disk to avoid OOM.

commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_info.csv'))
reduce_mem_usage(commit_data)
desc_data = pd.read_csv(os.path.join(DATA_DIR, 'cve_desc.csv'))

# Merge commit_data and desc_data on 'cve' column
data = pd.merge(commit_data, desc_data, on='cve', how='left')

data = data.drop(columns=['cve_desc','msg'])

print("shape of data: ", data.shape)
# Reduce memory usage
reduce_mem_usage(data)
data['diff'] = data['diff'].fillna(' ')

### we want to multiprocessing here
def tokenize_diff(diff):
    return tokenize_cpp_code(diff)

### we want to multiprocess tokenization
def multiprocess_tokenization(data, col_name):
    pool = mp.Pool(processes=mp.cpu_count())
    ### add tqdm here
    tokens = list(tqdm(pool.imap(tokenize_diff, data[col_name]), total=len(data[col_name])))
    pool.close()
    return tokens

## deprecated since will cause tokenization error
# ############ multiprocess the tokenized diff (list-like string) to a string

# def list2string(diff_token):
#     processed_diff = ' '.join(ast.literal_eval(diff_token)) 
#     return processed_diff

# def multiprocess_list2string(data, col_name):
#     pool = mp.Pool(processes=mp.cpu_count())
#     ### add tqdm here
#     tokens = list(tqdm(pool.imap(list2string, data[col_name]), total=len(data[col_name])))
#     pool.close()
#     return tokens


if __name__ == '__main__':
    
    # # Load already tokenized data
    # tokenized_file_path = os.path.join(DATA_TMP_DIR, 'diff_token_1.csv')
    # tokenized_file_path_read = os.path.join(DATA_TMP_DIR, 'diff_token.csv')

    # # data['diff_token'] = multiprocess_tokenization(data, 'diff')
    
    # data = pd.read_csv(tokenized_file_path_read)
    # reduce_mem_usage(data)
    
    # data['diff_token'] = data['diff_token'].fillna('[]')
    # data['diff_token'] = multiprocess_list2string(data, 'diff_token')
    # # Save the final data to the CSV file
    # # data['diff_token'].to_csv(tokenized_file_path, index=False)
    # data['diff_token'].to_csv(tokenized_file_path, index=False)
    
    # print("Tokenized diffs")
    
    
    # Load already tokenized data
    tokenized_file_path = os.path.join(DATA_TMP_DIR, 'diff_token_1.csv')

    data['diff_token'] = multiprocess_tokenization(data, 'diff')
    
    # Save the final data to the CSV file
    data['diff_token'].to_csv(tokenized_file_path, index=False)
    
    print("Tokenized diffs")
    
