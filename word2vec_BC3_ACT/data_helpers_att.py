# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:03:31 2019

@author: dqs
"""

import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
import json

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


def load_data_and_labels(num_sentence,length_sentence):
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
    for i in range(len(x_total)):
        str=x_total[i].split(". ")
        str=str[0:num_sentence]
        str=[s.strip() for s in str]
        str=[clean_str(sent) for sent in str]
        str=[s.split(" ") for s in str]
        sentence=[]
        for j in range(len(str)):
            if(len(str[j])>5):
                s=str[j][0:length_sentence]
                sentence.append(s)
        x.append(sentence)
    return [x,y]
def load_data_and_labels_dev(num_sentence,length_sentence):
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
    for i in range(len(x_total)):
        str=x_total[i].split(". ")
        str=str[0:num_sentence]
        str=[s.strip() for s in str]
        str=[clean_str(sent) for sent in str]
        str=[s.split(" ") for s in str]
        sentence=[]
        for j in range(len(str)):
            if(len(str[j])>3):
                s=str[j][0:length_sentence]
                sentence.append(s)
        x.append(sentence)
    return [x,y]


def load_data_and_labels_test(num_sentence,length_sentence):
    df=pd.read_csv('./BC3_ACT_Test/label.txt', sep='\t',header=None,names=['PMID','label'])
    id=[]
    label=[]
    count_1=0
    count_0=0
    for i in range(df.shape[0]):
        id.append(df.iloc[i][0])
        y_t=int(df.iloc[i][1])
        if(y_t==1):
            count_1=count_1+1
            label.append([0,1])
        elif(y_t==0):
            count_0=count_0+1
            label.append([1,0])
    #print(label[0:10])
    #print(count_1)
    #print(count_0)
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
    for i in range(len(x_total)):
        str=x_total[i].split(". ")
        str=str[0:num_sentence]
        str=[s.strip() for s in str]
        str=[clean_str(sent) for sent in str]
        str=[s.split(" ") for s in str]
        sentence=[]
        for j in range(len(str)):
            if(len(str[j])>3):
                s=str[j][0:length_sentence]
                #print(len(str[j]))
                sentence.append(s)
        x.append(sentence)
    return [x,y]
    


def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(max(len(y) for y in x) for x in sentences)
    document_length = max(len(x) for x in sentences)
    #print(sequence_length)
    #print(document_length)
    padding_sentence=[padding_word]*sequence_length
    #print(padding_sentence)
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            num_padding = sequence_length - len(sentences[i][j])
            sentences[i][j]=sentences[i][j]+[padding_word] * num_padding
        num_padding_sen=document_length-len(sentences[i])  
        sentences[i]=sentences[i]+[padding_sentence]*num_padding_sen
    #print(sentences[1])
    #print(len(sentences[0]))
    #print(len(sentences[0][14]))
    return sentences
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        new_sentence = new_sentence[0:400]
        padded_sentences.append(new_sentence)
    #return padded_sentences,sequence_length
    return padded_sentences,400
    """


def build_vocab(sentences):
    # Build vocabulary
    sentences_temp=[]
    for i in range(len(sentences)):
        x=[]
        for j in range(len(sentences[i])):
            for k in range(len(sentences[i][j])):
                x.append(sentences[i][j][k])
        sentences_temp.append(x)
    #print(sentences_temp)
    word_counts = Counter(itertools.chain(*sentences_temp))
    
    #print(word_counts)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[[vocabulary[word] for word in s] for s in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(num_sentence=12,length_sentence=45):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(num_sentence,length_sentence)
    sentences_dev, labels_dev = load_data_and_labels_dev(num_sentence,length_sentence)
    sentences=sentences+sentences_dev
    labels=labels+labels_dev
    sentences_padded = pad_sentences(sentences)
    sentences_test, labels_test = load_data_and_labels_test(num_sentence,length_sentence)
    sentences_padded_test = pad_sentences(sentences_test)
    #print(len(sentences_padded))
    #print(len(sentences_padded_test))
    sentences_padded_total=sentences_padded+sentences_padded_test
    #print(len(sentences_padded_total))
    vocabulary, vocabulary_inv = build_vocab(sentences_padded_total)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_t,y_t=build_input_data(sentences_padded_test, labels_test, vocabulary)
    return [x, y, vocabulary, vocabulary_inv,x_t,y_t]

def load_data_test(num_sentence=12,length_sentence=45):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_test(num_sentence,length_sentence)
    """
    for i in range(10):
        print(len(sentences[i]))
        print("句子")
        for j in range(len(sentences[i])):
            print(len(sentences[i][j]))
    """
    sentences_padded = pad_sentences(sentences)
    #vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

if __name__=='__main__':
    x,y,_,_=load_data(12,45)
    #x,y,_,_=load_data_test(12,45)
    print(x[0])
    