# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:38:26 2018

@author: Sofie
"""
import tensorflow as tf
import util
from model import cnn_bilstm_crf
import os

flags = tf.app.flags
flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_integer('batch_size', 10, 'mini batch size')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('dropout_rate', 0.5, 'dropout rate')
flags.DEFINE_float('momentum', 0.9, 'momentum of optimizer')
flags.DEFINE_integer('clip_grad', 5, 'gardient clipping')
flags.DEFINE_integer('char_embedding_size', 30, 'char embedding size')
flags.DEFINE_integer('word_embedding_size', 100, 'word embedding size')
flags.DEFINE_integer('filter_num', 30, 'number of convolution kenerls')
flags.DEFINE_integer('hidden_dim', 200, 'number of hidden states of bilstm')
flags.DEFINE_string('dataset_path', './CoNLL2003', 'raw dataset, we use CoNLL2003')
FLAGS = flags.FLAGS

def main(_):
    max_sentence = 150
    max_word = 70
        
    train_sentences, train_labels = util.read_sentences(os.path.join(FLAGS.dataset_path, 'train.txt')) 
    valid_sentences, valid_labels = util.read_sentences(os.path.join(FLAGS.dataset_path, 'valid.txt'))
    test_sentences, test_labels = util.read_sentences(os.path.join(FLAGS.dataset_path, 'test.txt'))
    
    chars2ids, char_vocabulary, char_embedding= util.char_embedding_matrix(train_sentences + valid_sentences, 
                                                                          FLAGS.char_embedding_size)
    train_chars2ids  = util.char2id(chars2ids, char_vocabulary, train_sentences, max_sentence, max_word)
    test_chars2ids = util.char2id(chars2ids, char_vocabulary, test_sentences, max_sentence, max_word)     
    words2ids, word_vocabulary, word_embedding = util.word_embedding_matrix('./glove.6B.100d.txt', 
                                                                            train_sentences + valid_sentences, 
                                                                            FLAGS.word_embedding_size)
    train_words2ids, train_sequence_lengths = util.word2id(words2ids, word_vocabulary, train_sentences, max_sentence)
    test_words2ids, test_sequence_lengths = util.word2id(words2ids, word_vocabulary, test_sentences, max_sentence)
    labels2ids, ids2labels, label_num = util.build_label_ids(train_labels + valid_labels)
    train_labels2ids = util.label2id(labels2ids, train_labels, max_sentence)
    test_labels2ids = util.label2id(labels2ids, test_labels, max_sentence)
    
#    run_config = tf.ConfigProto()
#    run_config.gpu_options.allow_growth=True
#    with tf.Session(config=run_config) as sess:
    with tf.Session() as sess:
        model = cnn_bilstm_crf(FLAGS.clip_grad, FLAGS.batch_size, FLAGS.char_embedding_size, 
                               FLAGS.word_embedding_size, FLAGS.filter_num, FLAGS.hidden_dim, 
                               FLAGS.dropout_rate, max_sentence, max_word, word_embedding, 
                               char_embedding, train_words2ids, train_chars2ids, train_labels2ids, 
                               train_sequence_lengths, test_words2ids, test_chars2ids, 
                               test_labels2ids, test_sequence_lengths, label_num)
        emit_matrix, transition_params = model.train(FLAGS, sess)
#save training model        
        saver = tf.train.Saver()
        saver.save(sess, 'Model/model.ckpt')
        
        print('start running test')
        if os.path.exists('Model/model.ckpt.meta'): 
            saver = tf.train.Saver()
            saver.restore(sess, './Model/model.ckpt')
            model.test(sess, FLAGS, emit_matrix, transition_params, ids2labels, label_num)
        else:
            raise Exception("[!] Train a model first")

if __name__ == '__main__':
  tf.app.run()
        
        
        
        
        