# A Systematic Study of TextGCN: On Improving Effectiveness and Efficiency

## UCLA CS 247 W22 Course Project (Group 8)

| Name       | UID |
|---------------|----------|
| Da Yin |   305450915  |
| Rakesh Bal   | 605526216     |
| Rahul Kapur   | 405530587 |

This repository is a PyTorch and Python of the course project for UCLA CS 247 (Advanced Data Mining) W22 Course.

Tested on the 20NG/R8/R52/Ohsumed/MR data set, the code on this repository can achieve the effect of the original TextGCN paper and all of our approaches mentioned in the report.

## Requirements
* fastai
* PyTorch
* scipy
* pandas
* spacy
* nltk
* prettytable
* numpy
* networkx
* tqdm
* scikit_learn

## Usage
1. First clone the original pytorch TextGCN repository 
`git clone https://github.com/chengsen/PyTorch_TextGCN.git`
2. Then clone this repository `git clone https://github.com/WadeYin9712/CS247_proj.git`
3. Replace the `build_graph.py` and `trainer.py` files in the original pytorch TextGCN repository with our `build_graph.py` and `trainer.py`.
4. Then in the original repo process the data first by running `data_processor.py` (Already done)
2. Then run `build_graph.py` to generate all the graphs
3. Then run `trainer.py` to train the model and also see the test accuracies for all the datasets and our proposed approaches. 

## References
[1] [Yao, L. , Mao, C. , & Luo, Y. . (2018). Graph convolutional networks for text classification.](https://arxiv.org/abs/1809.05679)
