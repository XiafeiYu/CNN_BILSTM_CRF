# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:59:02 2018

@author: Sofie
"""
import tensorflow as tf
import numpy as np
import os
from sklearn import metrics

class cnn_bilstm_crf(object):
    def __init__(self, clip_grad, batch_size, char_embedding_size, word_embedding_size, 
                 filter_num, hidden_dim, dropout_rate, max_sentence, max_word, word_embed, 
                 char_embed, train_words2ids, train_chars2ids, train_labels2ids, 
                 train_sequence_lengths, test_words2ids, test_chars2ids, 
                 test_labels2ids, test_sequence_lengths, label_num):
#--------------------------------------------------------------------------
#        clip_grad: gradient clipping number
#        char_embedding_size: char embedding dimension
#        word_embedding_size: word embedding dimension
#        filter_num: the number of convolution kernels
#        hidden_dim: number of hidden states of bilstm
#        max_sentence: the max length of sentence in dataset
#        max_word: the max length of word in dataset
#        word_embed: the embedding matrix of words
#        char_embed: the embedding matrix of chars
#        train_words2ids: words to ids matrix of training dataset
#        train_chars2ids: chars to ids matrix of training dataset
#        train_labels2ids: labels to ids matrix of training dataset
#        train_sequence_lengths: the real length(before padding) of each sentence in training dataset
#        test_words2ids: words to ids matrix of test dataset
#        test_chars2ids: chars to ids matrix of test dataset
#        test_labels2ids: labels to ids matrix of test dataset
#        test_sequence_lengths: the real length(before padding) of each sentence in test dataset
#        label_num: the total number of labels
#-------------------------------------------------------------------------------        
        self.clip_grad = clip_grad
        self.batch_size = batch_size
        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size
        self.filter_num = filter_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_sentence = max_sentence
        self.max_word = max_word
        self.train_words2ids = train_words2ids
        self.train_chars2ids = train_chars2ids
        self.train_labels2ids = train_labels2ids
        self.train_sequence_lengths = train_sequence_lengths
        self.test_words2ids = test_words2ids
        self.test_chars2ids = test_chars2ids
        self.test_labels2ids = test_labels2ids
        self.test_sequence_lengths = test_sequence_lengths
        self.label_num = label_num
        
        self.input_words_ids = tf.placeholder(tf.int32, shape = [self.batch_size, max_sentence])
        self.input_chars_ids = tf.placeholder(tf.int32, shape = [self.batch_size, max_sentence, max_word])
        self.input_labels = tf.placeholder(tf.int32, shape = [self.batch_size, max_sentence])
        self.input_sequence_lengths = tf.placeholder(tf.int32, shape = [self.batch_size])
        self.word_embedding = tf.Variable(word_embed, name = 'word_embedding')
        self.char_embedding = tf.Variable(char_embed, name = 'char_embedding')
#--------------------using ids to find corresponding embedding vectors-------------------    
    def look_up(self, words_ids, chars_ids):
        with tf.variable_scope('look_up'):
            words_vectors = tf.nn.embedding_lookup(self.word_embedding, words_ids)
            chars_vectors = tf.nn.embedding_lookup(self.char_embedding, chars_ids)
    #        words_vectors = tf.nn.dropout(words_vectors, keep_prob = (1.0 - dropout_rate))
        return (words_vectors, chars_vectors)
#---------------------cnn layer to extract the char-level character----------------------    
    def CNN(self, x):
        with tf.variable_scope('convolution'):
#            dropout before input to convolution
            char_embedding = tf.reshape(x, [-1, self.max_word, self.char_embedding_size, 1])
            char_embedding = tf.nn.dropout(char_embedding, keep_prob = (1.0 - self.dropout_rate))
            w = tf.get_variable('w', shape = [3, self.char_embedding_size,  1, self.filter_num], 
                                initializer = tf.truncated_normal_initializer(stddev = 0.01))
            b = tf.get_variable('b', shape = [self.filter_num], initializer = tf.zeros_initializer())
            conv = tf.nn.relu(tf.nn.conv2d(char_embedding, w, strides = [1, 1, 1, 1], padding = 'VALID') + b)
            char_represent = tf.nn.max_pool(conv, ksize = [1, self.max_word-2, 1, 1], 
                                            strides = [1, 1, 1, 1], padding = 'VALID')
        return tf.reshape(char_represent, [-1, self.max_sentence, self.filter_num])
#-------------------------bilstm layer -----------------------------    
    def bilstm(self, embedding, sequence_lengths):
        with tf.variable_scope('bilstm'):
#           dropout before input to bilstm and dropout after output from bilstm
            embedding = tf.nn.dropout(embedding, keep_prob = (1.0 - self.dropout_rate))
#            forward unit
            lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
#            backward unit
            lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw, cell_bw = lstm_bw, 
                                                         inputs = embedding, dtype=tf.float32,
                                                         sequence_length = sequence_lengths)
            outputs = tf.concat([outputs[0], outputs[1]], 2)
            outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])
            outputs = tf.nn.dropout(outputs, keep_prob = (1.0 - self.dropout_rate))
        return outputs
#------------------------ fully contacted beteewn output from bilstm and labels--------------------
    def linear(self, x, label_num):
        with tf.variable_scope('linear'):
            w = tf.get_variable('w', shape = [2*self.hidden_dim, label_num],
                                initializer = tf.truncated_normal_initializer(stddev = 0.01))
            b = tf.get_variable('b', shape = [label_num], initializer = tf.zeros_initializer())
            output = tf.matmul(x, w) + b
        return tf.reshape(output, [-1, self.max_sentence, label_num])
#--------------------calculate the loss, take out each sentence paddings----------------------------    
    def cal_loss(self, x, y, sequence_lengths):
        with tf.variable_scope('loss'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(x, y, sequence_lengths)
            loss = -tf.reduce_mean(log_likelihood)
        return (loss, transition_params)
#-----------------optimize loss using adam optimizer-----------------------
    def optimize(self, lr, loss, momentum):
        optim = tf.train.AdamOptimizer(learning_rate = lr, beta1 = momentum)
        grads_and_vars = optim.compute_gradients(loss)
        #gradient clipping
        grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
        optimizer = optim.apply_gradients(grads_and_vars_clip)
        return optimizer
#-----------------viterbi decode, find the highest score labels---------------------
    def viterbi_decode(self, num, predication, sequence_lengths, trans_matrix):
        labels = []
        scores = []
        for i in range(num):
            sentence_len = sequence_lengths[i]
            unary_scores = predication[i][:sentence_len]
            label_seqence, label_score = tf.contrib.crf.viterbi_decode(unary_scores, trans_matrix)
            labels.append(label_seqence)
            scores.append(label_score)
        return (labels, scores)
#--------------------calculate the precison of training batch----------------------    
    def evaluate(self, num, predict_labels, labels, sequence_lengths):
        reals = []
        predicts = []
        for i in range(num):
            sentence_len = sequence_lengths[i]
            reals.extend(labels[i][:sentence_len])
            predicts.extend(predict_labels[i])
        p = metrics.precision_score(reals, predicts, average='micro')
        return p
#-------------------train--------------------        
    def train(self, config, sess):        
                
        batch_word_embedding,  batch_char_embedding = self.look_up(self.input_words_ids, self.input_chars_ids)
        batch_word_embedding = tf.reshape(batch_word_embedding, 
                                          shape = [-1, self.max_sentence, self.word_embedding_size])
        chars_represent = self.CNN(batch_char_embedding)        
        final_embedding = tf.concat([chars_represent, batch_word_embedding], 2)            
        outputs = self.bilstm(final_embedding, self.input_sequence_lengths)
        self.emit_matrix = self.linear(outputs, self.label_num)
        loss, self.transition_params = self.cal_loss(self.emit_matrix, self.input_labels, self.input_sequence_lengths)
        
        optimizer = self.optimize(config.learning_rate, loss, config.momentum)
        
        if os.path.exists('Model/model.ckpt.meta'): 
            saver = tf.train.Saver()
            saver.restore(sess, './Model/model.ckpt')
        else:
            tf.global_variables_initializer().run()
        num_batch = len(self.train_words2ids) // config.batch_size
        for epo in range(config.epoch):
            for i in range(num_batch):
                batch_words_ids = np.array(self.train_words2ids[i*config.batch_size : (i+1)*config.batch_size])
                batch_chars_ids = np.array(self.train_chars2ids[i*config.batch_size : (i+1)*config.batch_size]) 
                batch_labels = np.array(self.train_labels2ids[i*config.batch_size : (i+1)*config.batch_size])
                batch_sequence_lengths = np.array(self.train_sequence_lengths[i*config.batch_size : (i+1)*config.batch_size])
                _, predication, show_loss, trans_matrix = sess.run([optimizer, self.emit_matrix, loss, self.transition_params], 
                                                     feed_dict = {self.input_words_ids: batch_words_ids, 
                                                                  self.input_chars_ids: batch_chars_ids,
                                                                  self.input_labels: batch_labels,
                                                                  self.input_sequence_lengths: batch_sequence_lengths})
                predict_labels, _ = self.viterbi_decode(
                        config.batch_size, predication, batch_sequence_lengths, trans_matrix)
                p = self.evaluate(
                        config.batch_size, predict_labels, batch_labels, batch_sequence_lengths)
                if i % 10 == 0:    
#                    show_loss = loss.eval(feed_dict = {self.input_words_ids: batch_words_ids, 
#                                                 self.input_chars_ids: batch_chars_ids,
#                                                 self.input_labels: batch_labels,
#                                                 self.input_sequence_lengths: batch_sequence_lengths})
#                    print("Epoch: [%2d] [%4d/%4d], loss: %.8f"% (epo, i, num_batch, show_loss))
                    print("Epoch: [%2d] [%4d/%4d], loss: %.8f, accurate = %.4f"% (epo, i, num_batch, show_loss, p))
        saver = tf.train.Saver()
        saver.save(sess, 'Model/model.ckpt')        
#        return (emit_matrix, transition_params)
#-------------------------test---------------------
    def test(self, sess, config, ids2labels, label_num):        
        predict_label = []
        test_label = []         
        batch_num = len(self.test_words2ids) // config.batch_size
        for i in range(batch_num):
            batch_words_ids = np.array(self.test_words2ids[i*config.batch_size : (i+1)*config.batch_size])
            batch_chars_ids = np.array(self.test_chars2ids[i*config.batch_size : (i+1)*config.batch_size]) 
            batch_labels = np.array(self.test_labels2ids[i*config.batch_size : (i+1)*config.batch_size])
            batch_sequence_lengths = np.array(self.test_sequence_lengths[i*config.batch_size : (i+1)*config.batch_size])
            predication, trans_matrix = sess.run([self.emit_matrix, self.transition_params], 
                                                 feed_dict = {self.input_words_ids: batch_words_ids, 
                                                              self.input_chars_ids: batch_chars_ids,
                                                              self.input_labels: batch_labels,
                                                              self.input_sequence_lengths: batch_sequence_lengths})
            predicts, _ = self.viterbi_decode(config.batch_size, predication, batch_sequence_lengths, trans_matrix)
            for j in range(config.batch_size):
                sentence_len = batch_sequence_lengths[j]
                test_label.extend(batch_labels[j][:sentence_len])
                predict_label.extend(predicts[j])
            if i % 50 == 0:
                print('process completed %d %%, please be patient' %int(i/batch_num*100))
        target_names = []
#       remove the padding labels we add
        for i, label in enumerate(test_label):
            test_label[i] = label - 1 
        for i, label in enumerate(predict_label):
            predict_label[i] = label - 1         
        for i in range(label_num):
            if i == 0:
                continue
            target_names.append(ids2labels[i])
#        f = open('result.txt', 'wb')
#        f.write(metrics.classification_report(test_label, predict_label, target_names=target_names).encode(encoding = 'utf-8'))
#        f.close()
        print(metrics.classification_report(test_label, predict_label, target_names=target_names))


           



    