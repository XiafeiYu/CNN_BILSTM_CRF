# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:48:33 2018

@author: Sofie
"""
import numpy as np


#--------------read sentences and labels from dataset--------------------
def read_sentences(path, encoding="utf8"):
    with open(path) as fp:
        sentence = []
        sentences = []
        label = []
        labels = []
        for line in fp.readlines()[2:]:
            if line == '\n':
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
            else:
                sentence.append(line.split()[0])
                label.append(line.split()[-1])        
    return (sentences, labels)

#--------randomly initial char embedding-------------
def char_embedding_matrix(sentences, d):
    chars = []
    embedding_matrix = []
    char_id = {}
    for sentence in sentences:
        for word in sentence:
            for char in word:
                chars.append(char)
    chars = set(chars)
#   add a null symbol for char padding to CNN to represent character-level--------------- 
    char_id['NUL'] = 0
#   add a unknown symbol in case the test dataset appear the chars which are not appeared in training dataset
    char_id['Un_known'] = len(chars) + 1
    embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (d, 1)))
    for i, char in enumerate(chars):
        char_id[char] = i+1
        embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (d, 1)))
    embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (d, 1))) 
    embedding_matrix = np.reshape(embedding_matrix, [-1, d])
    embedding_matrix = embedding_matrix.astype(np.float32)
    return (char_id, chars, embedding_matrix)
    
#-----------------read pretraining word embedding matrix, widipedia2014 GloVe 100 dimension-----------------  
def word_embedding_matrix(path, sentences, d):
    word_id = {}   
    vocab = []
    with open(path, encoding="utf8") as fp:
        for sentence in sentences:
            for word in sentence:
                vocab.append(word.lower())
        vocab = set(vocab)
        vocabulary = []
#the first vector is embedding of null word
        embedding_matrix = [np.random.uniform(-1.0, 1.0, (d, 1))]
        for line in fp.readlines():
            if line.split()[0] in vocab:
                vocabulary.append(line.split()[0])
                embedding_matrix.append(np.reshape(np.array(line.split()[1:]), (d, 1)))
#   mapping words and ids, add a null and a unknown symbol
    word_id['NUL'] = 0
    word_id['Un_known'] = len(vocabulary) + 1
    embedding_matrix.append(np.random.uniform(-1.0, 1.0, (d, 1)))
    embedding_matrix = np.reshape(embedding_matrix, [-1, d])
    embedding_matrix = embedding_matrix.astype(np.float32)
#    embedding_matrix.dtype = 'float32' 
    for i, word in enumerate(vocabulary):
        word_id[word] = i+1
    
    word_max = 50
    for sentence in sentences:
        word_max = max(word_max, len(sentence))
        
    return (word_id, vocabulary, embedding_matrix)

#----------mapping labels to ids------------
def build_label_ids(labels):
    label = []
    label_id = {}
    for sentence in labels:
        for word in sentence:
            label.append(word)
    label = set(label)
    label_num = len(label)
    label_id['NUL'] = 0
    for i, label in enumerate(label):
        label_id[label] = i + 1
    id_label = {v : k for k, v in label_id.items()}
           
    return (label_id, id_label, label_num)

#------------convert words in batch sentences to ids-------------------- 
def word2id(word_id, vocabulary, X, max_lenth):    
    X2id = []
    sequence_lengths = []
    for sentence in X:
        index = []
        for word in sentence:
            if word.lower() in vocabulary:
                index.append(word_id[word.lower()])
            else:
                index.append(word_id['Un_known'])
#pad each sentence ids with 0
        sequence_lengths.append(len(index))
        index = np.pad(index, (0, max_lenth-len(index)), 'constant')
        X2id.append(index)
    return (X2id, sequence_lengths)

#------------convert chars in batch sentences to ids--------------------        
def char2id(char_id, vocabulary, X, max_sentence, max_word):
    X2id = []
    for sentence in X:
        word_index = []
        for word in sentence:
            char_index = []
            for char in word:
                if char in vocabulary:
                    char_index.append(char_id[char])
                else:
                    char_index.append(char_id['Un_known'])                
#pad each word ids with 0, pad each sentence ids with 0
            char_index = np.pad(char_index, (0, max_word-len(char_index)), 'constant')
            word_index.append(char_index)
        word_index = np.pad(word_index, ((0, max_sentence-len(word_index)), (0, 0)), 'constant')
        X2id.append(word_index)
    return X2id        
               
#-------------convert labels in batch sentences to ids------------------
def label2id(label_id, X, max_sentence):
    X2id = []
    for sentence in X:
        index = []
        for label in sentence:
            index.append(label_id[label])
        index = np.pad(index, (0, max_sentence-len(index)), 'constant') 
        X2id.append(index)    
    return X2id
       