# text_classification-of-biomedicine
## Introduction
* The title of the paper is Biomedical document triage using hierarchical attention-based capsule network.
* It is my graduation project, mainly about text classification.
* It uses 3 data sets，including PM,IAS and ACT.More detailed information can be viewed in the paper.
* It's main innovation are hierarchical attention and capsule network.
* word2vec_PM is about datasets PM,word2vec_ISA is about ISA,word2vec_BC3_ACT is about ACT
## Requirement
* python 3.5
* keras
* tensorflow 1.13.1
* sklearn
* gensim
* numpy
## Usage
* 1. download word2vec vector.
word2vec vector available:链接：https://pan.baidu.com/s/1Z3qsclRmcPNijubAkcLcWw  提取码：kvno 
* 2. copy the folder vector to folder word2vec_PM,word2vec_ISA,word2vec_BC3_ACT.
* 3. run python word2vec_cnn_hierarchy_att.py
* Capsule.py:it is model of Capsule Network.
* attention.py:it is self-attention and hierarchical self-attention.
* data_helepers.py and data_helpers_att.py:Data preprocess
* w2v.py:word2vec vector
* word2vec_cnn_att.py and word2vec_cnn_hierarchy_att.py:train and test
* visualization.py:visualization code
