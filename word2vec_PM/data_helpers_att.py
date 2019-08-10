
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:10:03 2019

@author: dqs
"""

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
    with open("./data/PMtask_Triage_TrainingSet.json",'r',encoding='utf-8') as load_f:
        load_dict=json.load(load_f)
    x=[]
    y=[]
    for i in range(len(load_dict["documents"])):
        if(len(load_dict["documents"][i]["passages"])==2):
            str=load_dict["documents"][i]["passages"][1]["text"].split(". ")
            str=str[0:12]
            str=[s.strip() for s in str]
            str=[clean_str(sent) for sent in str]
            str=[s.split(" ") for s in str]
            sentence=[]
            for j in range(len(str)):
                if(len(str[j])>5):
                    s=str[j][0:45]
                    sentence.append(s)
            x.append(sentence)
            if(load_dict["documents"][i]["infons"]["relevant"]=="yes"):
                y.append([1,0])
            elif(load_dict["documents"][i]["infons"]["relevant"]=="no"):
                y.append([0,1])
    #print(max(len(x) for x in x))
    
    #print(count_1,count_2)
    return [x,y]

def load_data_and_labels_test():
    with open("./data/PMtask_Triage_TestSet.json",'r',encoding='utf-8') as load_f:
        load_dict=json.load(load_f)
    x=[]
    y=[]
    #print(load_dict["documents"][0]["passages"][1]["text"])
    for i in range(len(load_dict["documents"])):
        if(len(load_dict["documents"][i]["passages"])==2):
            str=load_dict["documents"][i]["passages"][1]["text"].split(". ")
            str=str[0:12]
            str=[s.strip() for s in str]
            str=[clean_str(sent) for sent in str]
            str=[s.split(" ") for s in str]
            """
            for j in range(len(str)):
                str[j]=str[j][0:40]
            x.append(str)
            """
            sentence=[]
            for j in range(len(str)):
                if(len(str[j])>5):
                    s=str[j][0:45]
                    sentence.append(s)
            x.append(sentence)
            if(load_dict["documents"][i]["infons"]["relevant"]=="yes"):
                y.append([1,0])
            elif(load_dict["documents"][i]["infons"]["relevant"]=="no"):
                y.append([0,1])
    #print(max(len(x) for x in x))
    
    #print(count_1,count_2)
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


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    sentences_test, labels_test = load_data_and_labels_test()
    sentences_padded_test = pad_sentences(sentences_test)
    sentences_padded_total=sentences_padded+sentences_padded_test
    vocabulary, vocabulary_inv = build_vocab(sentences_padded_total)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_t,y_t=build_input_data(sentences_padded_test, labels_test, vocabulary)
    return [x, y, vocabulary, vocabulary_inv,x_t,y_t]

def load_data_test():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_test()
    """
    for i in range(10):
        print(len(sentences[i]))
        print("句子")
        for j in range(len(sentences[i])):
            print(len(sentences[i][j]))
    """
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

if __name__=='__main__':
    load_data_test()
    