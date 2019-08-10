import numpy as np
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
    file_train_pos=open('./IAS/TRAIN/bc2_ppi_ias_abstract_pos.txt','r')
    file_train_neg=open('./IAS/TRAIN/bc2_ppi_ias_abstract_neg.txt','r')
    lines_pos = file_train_pos.readlines()
    lines_neg = file_train_neg.readlines()
    #print(len(lines))
    x_total=[]
    y=[]
    x_pos=[]
    y_pos=[]
    x_neg=[]
    y_neg=[]
    for i in range(len(lines_pos)):
        if '<ABSTRACT>' in lines_pos[i]:
            x_pos.append(lines_pos[i+1])
        if '<CURATION_RELEVANCE>' in lines_pos[i]:
            str=int(lines_pos[i+1])
            if(str==1):
                y_pos.append([0,1])
    #print(len(x_pos))
    #print(len(y_pos))
    for i in range(len(lines_neg)):
        if '<ABSTRACT>' in lines_neg[i]:
            x_neg.append(lines_neg[i+1])
        if '<CURATION_RELEVANCE>' in lines_neg[i]:
            str=int(lines_neg[i+1])
            if(str==0):
                y_neg.append([1,0])
    #print(len(x_neg))
    #print(len(y_neg))
    x_total=x_pos+x_neg
    y=y_pos+y_neg
    #print(len(x))
    #print(len(y))
    file_train_pos.close()
    file_train_neg.close()
  
    x=[s.strip() for s in x_total]
    x_text=[clean_str(sent) for sent in x]
    x_text=[s.split(" ") for s in x_text]
    y=np.array(y)
    #print(count_1,count_2)
    return [x_text,y]
def load_data_and_labels_test():
    file_test_label=open('./IAS/TEST/ias_test_pmid2label.txt','r')
    dict={}
    id=[]
    label=[]
    for line in file_test_label:
        str=line
        list_str=str.split()
        id.append(list_str[0])
        if (list_str[1]=='P'):
            label.append([0,1])
        elif (list_str[1]=='N'):
            label.append([1,0])
    #print(len(id))
    #print(len(label))
    for i in range(len(id)):
        dict[id[i]]=label[i]
    file_test_label.close()
    file_test=open('./IAS/TEST/ias_test_abs.txt','r')
    lines = file_test.readlines()
    #print(len(lines))
    x_total=[]
    y=[]
    for i in range(len(lines)):
        if '<ABSTRACT>' in lines[i]:
            x_total.append(lines[i+1])
        if '<PMID>' in lines[i]:
            str=lines[i+1].replace('\n','')
            y.append(dict[str])
    #print(len(x_total))
    #print(len(y))
    file_test.close()
    #print(load_dict["documents"][0]["passages"][1]["text"])
    x=[]
    
    x=[s.strip() for s in x_total]
    x_text=[clean_str(sent) for sent in x]
    x_text=[s.split(" ") for s in x_text]
    y=np.array(y)
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
