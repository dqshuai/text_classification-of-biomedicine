import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
import json
"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    #string = re.sub(r"[^A-Za-z0-9(),!?\-\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
#    string = re.sub(r",", " , ", string)
#    string = re.sub(r"!", " ! ", string)
#    string = re.sub(r"\(", " \( ", string)
#    string = re.sub(r"\)", " \) ", string)
#    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    df=pd.read_csv('./BC3_ACT_Training/bc3_act_gold_standard.tsv', sep='\t',header=None,names=['PMID','label'])
    id=[]
    label=[]
    for i in range(df.shape[0]):
        id.append(df.iloc[i][0])
        y_t=int(df.iloc[i][1])
        if(y_t==1):
            label.append([0,1])
        elif(y_t==0):
            label.append([1,0])
    #print(len(id))
    #print(len(label))
    dict={}
    for i in range(len(id)):
        dict[id[i]]=label[i]
    train=pd.read_csv('./BC3_ACT_Training/bc3_act_all_records.tsv', sep='\t',header=None,names=['PMID','Journal','NLMID','Year','Title','Abstract'])
    x_total=[]
    x=[]
    y=[]
    #print(train.shape[0])
    for i in range(train.shape[0]):
        x_total.append(train.iloc[i][5])
        str_t=train.iloc[i][0]
        y.append(dict[str_t])   
    x=[s.strip() for s in x_total]
    x_text=[clean_str(sent) for sent in x]
    x_text=[s.split(" ") for s in x_text]
    #y=np.array(y)
    #print(count_1,count_2)
    return [x_text,y]
def load_data_and_labels_dev():
    df=pd.read_csv('./BC3_ACT_Development/bc3_act_gold_standard_development.tsv', sep='\t',header=None,names=['PMID','label'])
    id=[]
    label=[]
    for i in range(df.shape[0]):
        id.append(df.iloc[i][0])
        y_t=int(df.iloc[i][1])
        if(y_t==1):
            label.append([0,1])
        elif(y_t==0):
            label.append([1,0])
    #print(len(id))
    #print(len(label))
    dict={}
    for i in range(len(id)):
        dict[id[i]]=label[i]
    train=pd.read_csv('./BC3_ACT_Development/bc3_act_all_records_development.tsv', sep='\t',header=None,names=['PMID','Journal','NLMID','Year','Title','Abstract'])
    x_total=[]
    x=[]
    y=[]
    #print(train.shape[0])
    for i in range(train.shape[0]):
        x_total.append(train.iloc[i][5])
        str_t=train.iloc[i][0]
        y.append(dict[str_t])   
    x=[s.strip() for s in x_total]
    x_text=[clean_str(sent) for sent in x]
    x_text=[s.split(" ") for s in x_text]
    #y=np.array(y)
    #print(count_1,count_2)
    return [x_text,y]
def load_data_and_labels_test():
    df=pd.read_csv('./BC3_ACT_Test/bc3_act_pmids_test.tsv', sep='\t',header=None,names=['PMID','label'])
    id=[]
    label=[]
    for i in range(df.shape[0]):
        id.append(df.iloc[i][0])
        label.append([1,0])
        """
        y_t=int(df.iloc[i][1])
        if(y_t==1):
            label.append([0,1])
        elif(y_t==0):
            label.append([1,0])
        """
    #print(len(id))
    #print(len(label))
    dict={}
    for i in range(len(id)):
        dict[id[i]]=label[i]
    train=pd.read_csv('./BC3_ACT_Test/bc3_act_all_records_test.tsv', sep='\t',header=None,names=['PMID','Journal','NLMID','Year','Title','Abstract'])
    x_total=[]
    x=[]
    y=[]
    #print(train.shape[0])
    for i in range(train.shape[0]):
        x_total.append(train.iloc[i][5])
        str_t=train.iloc[i][0]
        y.append(dict[str_t])   
    x=[s.strip() for s in x_total]
    x_text=[clean_str(sent) for sent in x]
    x_text=[s.split(" ") for s in x_text]
    #y=np.array(y)
    #print(count_1,count_2)
    return [x_text,y]


def pad_sentences(sentences,sequence_length,padding_word="<PAD/>"):
    #sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        new_sentence = new_sentence[0:sequence_length]
        padded_sentences.append(new_sentence)
    #return padded_sentences,sequence_length
    return padded_sentences


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(sequence_length):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_dev, labels_dev = load_data_and_labels_dev()
    sentences=sentences+sentences_dev
    labels=labels+labels_dev
    sentences_padded = pad_sentences(sentences,sequence_length)
    sentences_test, labels_test = load_data_and_labels_test()
    sentences_padded_test = pad_sentences(sentences_test,sequence_length)
    sentences_padded_total=sentences_padded+sentences_padded_test
    vocabulary, vocabulary_inv = build_vocab(sentences_padded_total)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_t, y_t = build_input_data(sentences_padded_test, labels_test, vocabulary)
    return [x, y, vocabulary, vocabulary_inv,x_t,y_t]


def load_data_test(sequence_length):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_test()
    sentences_padded = pad_sentences(sentences,sequence_length)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
