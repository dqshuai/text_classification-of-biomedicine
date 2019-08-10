# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:03:31 2019

@author: dqs
"""

import numpy as np
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
    x=[]
    for i in range(len(x_total)):
        str=x_total[i].split(". ")
        """
        if(len(str)<num_sentence):
            padding_sentence="<PAD/> "*length_sentence
            num=num_sentence-len(str)
            print(num)
            str=str+[padding_sentence]*num
        #print(str)
        """
        str=str[0:num_sentence]
        str=[s.strip() for s in str]
        str=[clean_str(sent) for sent in str]
        str=[s.split(" ") for s in str]
        sentence=[]
        for j in range(len(str)):
            if(len(str[j])>5):
                """
                if(len(str[j])<length_sentence):
                    str[j]=str[j]+["<PAD/>"]*(length_sentence-len(str[j]))
                """
                s=str[j][0:length_sentence]
                sentence.append(s)
        x.append(sentence)
    return [x,y]

def load_data_and_labels_test(num_sentence,length_sentence):
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
    for i in range(len(x_total)):
        str=x_total[i].split(". ")
        str=str[0:num_sentence]
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
                s=str[j][0:length_sentence]
                sentence.append(s)
        x.append(sentence)
    #print(max(len(x) for x in x))
    
    #print(count_1,count_2)
    return [x,y]


def pad_sentences(sentences,num_sentence,length_sentence,padding_word="<PAD/>"):
    sequence_length=length_sentence
    document_length=num_sentence
    #sequence_length = max(max(len(y) for y in x) for x in sentences)
    #document_length = max(len(x) for x in sentences)
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
    sentences_padded = pad_sentences(sentences,num_sentence,length_sentence)
    sentences_test, labels_test = load_data_and_labels_test(num_sentence,length_sentence)
    sentences_padded_test = pad_sentences(sentences_test,num_sentence,length_sentence)
    sentences_padded_total=sentences_padded+sentences_padded_test
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
    sentences_padded = pad_sentences(sentences,num_sentence,length_sentence)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

if __name__=='__main__':
    x,y=load_data_and_labels(30,45)
    #print(x)
    