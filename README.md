# Adaptive Document Retrieval for Deep Question Answering
 This is the GitHub Repository complementing the EMNLP 2018 paper [Adaptive Document Retrieval for Deep Question Answering](https://www.google.com)
 
## Abstract
State-of-the-art systems in deep question answering proceed as follows: (1) an initial document retrieval selects relevant documents, which (2) are then processed by a neural network in order to extract the final answer. Yet the exact interplay between both components is poorly understood, especially concerning the number of candidate documents that should be retrieved. We show that choosing a static number of documents -- as used in prior research -- suffers from a noise-information trade-off and yields suboptimal results. As a remedy, we propose an adaptive document retrieval model. This learns the optimal candidate number for document retrieval, conditional on the size of the corpus and the query. We report extensive experimental results showing that our adaptive approach outperforms state-of-the-art methods on multiple benchmark datasets, as well as in the context of corpora with variable sizes.
 
## Demo
If you want to implement adaptive document retrieval for DrQA or your custom QA system please refer to the notebook in this repository.

## Contact
For questions, please contact bkratzwald ( at ) ethz (dot) ch
 
## Citation
Please consider citing us if you find this helpful for your work:

```
@inprocidings{kratzwald2018adaptive, 
  title={Adaptive Document Retrieval for Deep Question Answering},
  author={Kratzwald, Bernhard and Feuerriegel, Stefan},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2018}
}
```
