# -*- coding: utf-8 -*-
'''Contain simple neural network models for semantic role labeling.

The models in this file are inspired by the work of Diego Marcheggiani, Anton Frolov and Ivan Titov
"A Simple and Accurate Syntax-Agnostic Neural Model for Dependency-based Semantic Role Labeling"
'''

import tensorflow as tf
import srl_models.model

CELL_SIZE = 150
BILSTM_SIZE = CELL_SIZE*2
PRELOGIT_SIZE = BILSTM_SIZE*2


class SimpleModel(srl_models.model.Model):
    '''Simple neural network for semantic role labeling.'''

    def _get_feed_dict(self, batch):
        # prepare dictionary
        feed_dict = {self.ph_words: batch.words,
                     self.ph_pos:   batch.pos,
                     self.ph_pred:  batch.predicates,
                     self.ph_spred: batch.predicate_positions,
                     self.ph_dep:   batch.dependencies,
                     self.ph_roles: batch.roles,
                     self.ph_lens:  batch.sequence_lengths}
        return feed_dict

    # function used in order to create the tensorflow graph
    def _create_graph(self, embedding_data, roles):
        (wordembeddings,
         posembeddings,
         depembeddings,
         predembeddings) = embedding_data
        role_count = len(roles)

        # initialize graph
        graph = tf.Graph()
        self.graph = graph
        with graph.as_default():
            #-------------------------- EMBEDDINGS ----------------------------
            with tf.name_scope('embeddings'):
                # pre-trained word embeddings
                wordembs = tf.Variable(wordembeddings.embeddings,
                                       dtype=tf.float32,
                                       trainable=False,
                                       name='wordEmbeddings')

                # part-of-speech embeddings
                posembs = tf.Variable(
                    posembeddings.embeddings,
                    dtype=tf.float32,
                    trainable=True,
                    name='posEmbeddings')

                # predicate embeddings
                predembs = tf.Variable(
                    predembeddings.embeddings,
                    dtype=tf.float32,
                    trainable=True,
                    name='predEmbeddings')

                # dependency  embeddings
                depembs = tf.Variable(
                    depembeddings.embeddings,
                    dtype=tf.float32,
                    trainable=True,
                    name='depEmbeddings')

    #------------------------------------ INPUTS -------------------------------
            with tf.name_scope('inputs'):
                batch = self.batch_size

                # input data and labels placeholders
                self.ph_words = tf.placeholder(tf.int32, [batch, None], 'ph_words')
                self.ph_pos = tf.placeholder(tf.int32, [batch, None], 'ph_pos')
                self.ph_pred = tf.placeholder(tf.int32, [batch, None], 'ph_predicates')
                self.ph_spred = tf.placeholder(tf.int32, [batch, 2], 'ph_pred_index')
                self.ph_lens = tf.placeholder(tf.int32, [batch], 'ph_dep')
                self.ph_roles = tf.placeholder(tf.int32, [batch, None], 'ph_roles')
                self.ph_dep = tf.placeholder(tf.int32, [batch, None], 'ph_dependencies')

                timestep = tf.shape(self.ph_words)[1]

                # get sentences lengths and embeddings
                batch_x_word = tf.nn.embedding_lookup(wordembs, self.ph_words, name='words')
                batch_x_pos = tf.nn.embedding_lookup(posembs, self.ph_pos, name='pos')
                batch_x_pred = tf.nn.embedding_lookup(predembs, self.ph_pred, name='pred')
                batch_x_dep = tf.nn.embedding_lookup(depembs, self.ph_dep, name='dep')
                batch_x_flag = tf.expand_dims(
                    tf.scatter_nd(
                        indices=self.ph_spred,
                        updates=tf.fill([batch], 1.0),
                        shape=[batch, timestep]),
                    axis=-1)

                # final concatenated input
                batch_x = tf.concat([batch_x_word,
                                     batch_x_pos,
                                     batch_x_pred,
                                     batch_x_dep,
                                     batch_x_flag],
                                    axis=-1,
                                    name='input')


    #------------------------------- LSTM LAYERS ------------------------------
            with tf.name_scope('LSTM_layers'):
                # forward pass memory cells
                cellsf = [tf.contrib.rnn.BasicLSTMCell(CELL_SIZE)]
                # backward pass memory cells
                cellsb = [tf.contrib.rnn.BasicLSTMCell(CELL_SIZE)]

                # multi-layered dynamic BiLSTM
                (batch_lstm, _, _) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cellsf,
                    cellsb,
                    inputs=batch_x,
                    sequence_length=self.ph_lens,
                    dtype=tf.float32,
                    scope='lstm_subgraph')

                # get predicate cell
                batch_lstm_predicate = tf.gather_nd(batch_lstm, self.ph_spred)

                # repeat predicate timestep times
                batch_lstm_pred_repeat = tf.tile(
                    tf.expand_dims(batch_lstm_predicate, axis=1),
                    [1, timestep, 1])

                # append predicate info to LSTM output
                # shape: [batch_size,timestep,hidden_lstm]
                batch_lstm_final = tf.concat([batch_lstm, batch_lstm_pred_repeat], axis=-1)


    #-------------------------------- optimization -----------------------------
            # create prediction logits shape:
            with tf.name_scope('logits'):
                weights = tf.Variable(
                    tf.random_normal(
                        [PRELOGIT_SIZE, role_count],
                        dtype=tf.float32),
                    trainable=True,
                    name='W')

                bias = tf.Variable(
                    tf.random_normal(
                        [role_count],
                        dtype=tf.float32),
                    trainable=True,
                    name='W')

                # logits shape: [batch_size, timestep, role]
                batch_lstm_final_flat = tf.reshape(batch_lstm_final, [-1, PRELOGIT_SIZE])
                logits_flat = tf.matmul(batch_lstm_final_flat, weights) + bias
                self.logits = tf.reshape(logits_flat, [-1, timestep, role_count])

            # loss function
            with tf.name_scope('loss_function'):
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.ph_roles,
                        logits=self.logits,
                        name='softmax'),
                    name='loss')


            # optimizer used for training
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

    #--------------------------------------------------------------------------
            with tf.name_scope('meta'):
                # summary info
                if self.profile_data:
                    tf.summary.scalar('loss', loss)
                    self.summary = tf.summary.merge_all()

                # variable initializer (must be executed in order to initialize variables)
                self.init = tf.global_variables_initializer()

                # saver (store and restore model on disk)
                self.saver = tf.train.Saver()

class SimpleModel_V2(SimpleModel):
    '''Simple neural network for semantic role labeling.

    This version uses randomily initialized trainable word embeddings alongside GloVe embeddings.
    '''

    def _create_graph(self, embedding_data, roles):
        (wordembeddings,
         posembeddings,
         depembeddings,
         predembeddings) = embedding_data
        role_count = len(roles)

        # initialize graph
        graph = tf.Graph()
        self.graph = graph
        with graph.as_default():
            #-------------------------- EMBEDDINGS ----------------------------
            with tf.name_scope('embeddings'):
                # pre-trained word embeddings
                wordembs = tf.Variable(
                    wordembeddings.embeddings,
                    dtype=tf.float32,
                    trainable=False,
                    name='wordEmbeddings')

                randwordembs = tf.Variable(
                    tf.random_normal(
                        wordembeddings.embeddings.shape,
                        dtype=tf.float32),
                    trainable=True,
                    name='randomWordEmbeddings')

                # part-of-speech embeddings
                posembs = tf.Variable(
                    posembeddings.embeddings,
                    dtype=tf.float32,
                    trainable=True,
                    name='posEmbeddings')

                # predicate embeddings
                predembs = tf.Variable(
                    predembeddings.embeddings,
                    dtype=tf.float32,
                    trainable=True,
                    name='predEmbeddings')

                # dependency  embeddings
                depembs = tf.Variable(
                    depembeddings.embeddings,
                    dtype=tf.float32,
                    trainable=True,
                    name='depEmbeddings')

    #------------------------------------ INPUTS -------------------------------
            with tf.name_scope('inputs'):
                batch = self.batch_size

                # input data and labels placeholders
                self.ph_words = tf.placeholder(tf.int32, [batch, None], 'ph_words')
                self.ph_pos = tf.placeholder(tf.int32, [batch, None], 'ph_pos')
                self.ph_pred = tf.placeholder(tf.int32, [batch, None], 'ph_predicates')
                self.ph_spred = tf.placeholder(tf.int32, [batch, 2], 'ph_pred_posi')
                self.ph_lens = tf.placeholder(tf.int32, [batch], 'ph_dep')
                self.ph_roles = tf.placeholder(tf.int32, [batch, None], 'ph_roles')
                self.ph_par = tf.placeholder(tf.int32, [batch, None], 'ph_par_posi')
                self.ph_dep = tf.placeholder(tf.int32, [batch, None], 'ph_dependencies')

                timestep = tf.shape(self.ph_words)[1]

                # get sentences lengths and embeddings
                batch_x_word = tf.nn.embedding_lookup(wordembs, self.ph_words, name='words')
                batch_x_word_rand = tf.nn.embedding_lookup(randwordembs, self.ph_words, name='rand_words')
                batch_x_pos = tf.nn.embedding_lookup(posembs, self.ph_pos, name='pos')
                batch_x_pred = tf.nn.embedding_lookup(predembs, self.ph_pred, name='pred')
                batch_x_dep = tf.nn.embedding_lookup(depembs, self.ph_dep, name='dep')
                batch_x_flag = tf.expand_dims(
                    tf.scatter_nd(
                        indices=self.ph_spred,
                        updates=tf.fill([batch], 1.0),
                        shape=[batch, timestep]),
                    axis=-1)

                # final concatenated input
                batch_x = tf.concat([batch_x_word,
                                     batch_x_word_rand,
                                     batch_x_pos,
                                     batch_x_pred,
                                     batch_x_dep,
                                     batch_x_flag],
                                    axis=-1,
                                    name='input')


    #------------------------------- LSTM LAYERS ------------------------------
            with tf.name_scope('LSTM_layers'):
                # forward pass memory cells
                cellsf = [tf.contrib.rnn.BasicLSTMCell(CELL_SIZE)]
                # backward pass memory cells
                cellsb = [tf.contrib.rnn.BasicLSTMCell(CELL_SIZE)]

                # multi-layered dynamic BiLSTM
                (batch_lstm, _, _) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cellsf,
                    cellsb,
                    inputs=batch_x,
                    sequence_length=self.ph_lens,
                    dtype=tf.float32,
                    scope='lstm_subgraph')

                # get predicate cell
                batch_lstm_predicate = tf.gather_nd(batch_lstm, self.ph_spred)

                # repeat predicate timestep times
                batch_lstm_pred_repeat = tf.tile(
                    tf.expand_dims(
                        batch_lstm_predicate,
                        axis=1),
                    [1, timestep, 1])

                # append predicate info to LSTM output
                # shape: [batch_size,timestep,hidden_lstm]
                batch_lstm_final = tf.concat([batch_lstm, batch_lstm_pred_repeat], axis=-1)


    #-------------------------------- optimization -----------------------------
            # create prediction logits shape:
            with tf.name_scope('logits'):
                weights = tf.Variable(
                    tf.random_normal(
                        [PRELOGIT_SIZE, role_count],
                        dtype=tf.float32),
                    trainable=True,
                    name='W')

                bias = tf.Variable(
                    tf.random_normal(
                        [role_count],
                        dtype=tf.float32),
                    trainable=True,
                    name='W')

                # logits shape: [batch_size, timestep, role]
                batch_lstm_final_flat = tf.reshape(batch_lstm_final, [-1, PRELOGIT_SIZE])
                logits_flat = tf.matmul(batch_lstm_final_flat, weights) + bias
                self.logits = tf.reshape(logits_flat, [-1, timestep, role_count])

            # loss function
            with tf.name_scope('loss_function'):
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.ph_roles,
                        logits=self.logits,
                        name='softmax'),
                    name='loss')


            # optimizer used for training
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

    #--------------------------------------------------------------------------
            with tf.name_scope('meta'):
                # summary info
                if self.profile_data:
                    tf.summary.scalar('loss', loss)
                    self.summary = tf.summary.merge_all()

                # variable initializer (must be executed in order to initialize variables)
                self.init = tf.global_variables_initializer()

                # saver (store and restore model on disk)
                self.saver = tf.train.Saver()