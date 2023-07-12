import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = '/data/kaixuan/ramdisk/data'
DATA_TMP_DIR = '/data/kaixuan/data_tmp'

### read tfidf similarity data
tfidf_data = pd.read_csv(os.path.join(DATA_DIR, 'similarity_data.csv'))
tfidf_cve = tfidf_data.groupby('cve')

def normalize_similarity(run=False):
    ''' normalize the similarity score and save to csv '''
    if run:
        
        for _, group in tqdm(tfidf_cve):
            tfidf_data.loc[group.index, 'similarity'] = (group['similarity'] - group['similarity'].min()) / \
                (group['similarity'].max() - group['similarity'].min())
        tfidf_data.to_csv(os.path.join(DATA_TMP_DIR, 'similarity_data_normalized.csv'), index=False)
        print("normalize_similarity() is done")
        print("saved to {}".format(os.path.join(DATA_TMP_DIR, 'similarity_data_normalized.csv')))
    else:
        print('normalize_similarity() is not run')


'''Split the data into train, validate, and test'''
def split_data(run=False):
    if not run:
        print('split_data() is not run')
        return False
    # read the normalized similarity data
    normalized_data = pd.read_csv(os.path.join(DATA_TMP_DIR, 'similarity_data_normalized.csv'))
    grouped_data = normalized_data.groupby('cve')

    # List of unique cve
    cve_list = normalized_data['cve'].unique()
    print("There are {} unique CVEs".format(len(cve_list)))
    cve_list.sort()
    
    # Split the cve into train and rest (80% train, 20% rest)
    train_cve, rest_cve = train_test_split(cve_list, test_size=0.2, random_state=3407)
    
    # Split the rest cve into validate and test (50% validate, 50% test)
    validate_cve, test_cve = train_test_split(rest_cve, test_size=0.5, random_state=3407)

    # Get the corresponding data
    train_data = normalized_data[normalized_data['cve'].isin(train_cve)]
    validate_data = normalized_data[normalized_data['cve'].isin(validate_cve)]
    test_data = normalized_data[normalized_data['cve'].isin(test_cve)]
    
    # Save the sets as CSV files
    data_split_dir = os.path.join(DATA_TMP_DIR, 'data_split')
    os.makedirs(data_split_dir, exist_ok=True)
    train_data.to_csv(os.path.join(data_split_dir, 'train.csv'), index=False)
    validate_data.to_csv(os.path.join(data_split_dir, 'validate.csv'), index=False)
    test_data.to_csv(os.path.join(data_split_dir, 'test.csv'), index=False)

# if __name__ == '__main__':
#     normalize_similarity()
#     split_data()
    


# """Evaluate the accuracy of the DrQA retriever module."""

# import regex as re
# import logging
# import argparse
# import json
# import time
# import os

# from multiprocessing import Pool as ProcessPool
# from multiprocessing.util import Finalize
# from functools import partial
# from drqa import retriever, tokenizers
# from drqa.retriever import utils

# # ------------------------------------------------------------------------------
# # Multiprocessing target functions.
# # ------------------------------------------------------------------------------

# PROCESS_TOK = None
# PROCESS_DB = None


# def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
#     global PROCESS_TOK, PROCESS_DB
#     PROCESS_TOK = tokenizer_class(**tokenizer_opts)
#     Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
#     PROCESS_DB = db_class(**db_opts)
#     Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


# def regex_match(text, pattern):
#     """Test if a regex pattern is contained within a text."""
#     try:
#         pattern = re.compile(
#             pattern,
#             flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
#         )
#     except BaseException:
#         return False
#     return pattern.search(text) is not None


# def has_answer(answer, doc_id, match):
#     """Check if a document contains an answer string.

#     If `match` is string, token matching is done between the text and answer.
#     If `match` is regex, we search the whole text with the regex.
#     """
#     global PROCESS_DB, PROCESS_TOK
#     text = PROCESS_DB.get_doc_text(doc_id)
#     text = utils.normalize(text)
#     if match == 'string':
#         # Answer is a list of possible strings
#         text = PROCESS_TOK.tokenize(text).words(uncased=True)
#         for single_answer in answer:
#             single_answer = utils.normalize(single_answer)
#             single_answer = PROCESS_TOK.tokenize(single_answer)
#             single_answer = single_answer.words(uncased=True)
#             for i in range(0, len(text) - len(single_answer) + 1):
#                 if single_answer == text[i: i + len(single_answer)]:
#                     return True
#     elif match == 'regex':
#         # Answer is a regex
#         single_answer = utils.normalize(answer[0])
#         if regex_match(text, single_answer):
#             return True
#     return False


# def get_score(answer_doc, match):
#     """Search through all the top docs to see if they have the answer."""
#     answer, (doc_ids, doc_scores) = answer_doc
#     for doc_id in doc_ids:
#         if has_answer(answer, doc_id, match):
#             return 1
#     return 0


# # ------------------------------------------------------------------------------
# # Main
# # ------------------------------------------------------------------------------


# if __name__ == '__main__':
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
#                             '%m/%d/%Y %I:%M:%S %p')
#     console = logging.StreamHandler()
#     console.setFormatter(fmt)
#     logger.addHandler(console)

#     parser = argparse.ArgumentParser()
#     parser.add_argument('dataset', type=str, default=None)
#     parser.add_argument('--model', type=str, default=None)
#     parser.add_argument('--doc-db', type=str, default=None,
#                         help='Path to Document DB')
#     parser.add_argument('--tokenizer', type=str, default='regexp')
#     parser.add_argument('--n-docs', type=int, default=5)
#     parser.add_argument('--num-workers', type=int, default=None)
#     parser.add_argument('--match', type=str, default='string',
#                         choices=['regex', 'string'])
#     args = parser.parse_args()

#     # start time
#     start = time.time()

#     # read all the data and store it
#     logger.info('Reading data ...')
#     questions = []
#     answers = []
#     for line in open(args.dataset):
#         data = json.loads(line)
#         question = data['question']
#         answer = data['answer']
#         questions.append(question)
#         answers.append(answer)

#     # get the closest docs for each question.
#     logger.info('Initializing ranker...')
#     ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

#     logger.info('Ranking...')
#     closest_docs = ranker.batch_closest_docs(
#         questions, k=args.n_docs, num_workers=args.num_workers
#     )
#     answers_docs = zip(answers, closest_docs)

#     # define processes
#     tok_class = tokenizers.get_class(args.tokenizer)
#     tok_opts = {}
#     db_class = retriever.DocDB
#     db_opts = {'db_path': args.doc_db}
#     processes = ProcessPool(
#         processes=args.num_workers,
#         initializer=init,
#         initargs=(tok_class, tok_opts, db_class, db_opts)
#     )

#     # compute the scores for each pair, and print the statistics
#     logger.info('Retrieving and computing scores...')
#     get_score_partial = partial(get_score, match=args.match)
#     scores = processes.map(get_score_partial, answers_docs)

#     filename = os.path.basename(args.dataset)
#     stats = (
#         "\n" + "-" * 50 + "\n" +
#         "{filename}\n" +
#         "Examples:\t\t\t{total}\n" +
#         "Matches in top {k}:\t\t{m}\n" +
#         "Match % in top {k}:\t\t{p:2.2f}\n" +
#         "Total time:\t\t\t{t:2.4f} (s)\n"
#     ).format(
#         filename=filename,
#         total=len(scores),
#         k=args.n_docs,
#         m=sum(scores),
#         p=(sum(scores) / len(scores) * 100),
#         t=time.time() - start,
#     )

#     print(stats)
