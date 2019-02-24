# -*- coding: utf-8 -*-
'''Contain neural network models using GCN layers for semantic role labeling.

The models in this file are inspired by the work of Diego Marcheggiani and Ivan Titov in
"Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling"
'''
import tensorflow as tf

import srl_models.model

CELL_SIZE = 150
BILSTM_SIZE = CELL_SIZE*2+32
GCN_SIZE = 300



################################################################################
class OriginalModel(srl_models.model.Model):
    '''Neural network using a BiLSTM and GCN stacked together for semantic role labeling.

    This model is a scaled-down implementation of the model described in
    "Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling"
    '''

    def _get_feed_dict(self, batch):
        # prepare dictionary
        feed_dict = {self.ph_words: batch.words,
                     self.ph_pos:   batch.pos,
                     self.ph_pred:  batch.predicates,
                     self.ph_spred: batch.predicate_positions,
                     self.ph_dep:   batch.dependencies,
                     self.ph_roles: batch.roles,
                     self.ph_par:   batch.parent_positions,
                     self.ph_lens:  batch.sequence_lengths}
        return feed_dict

    # function used in order to create the tensorflow graph
    def _create_graph(self, embedding_data, roles):
        (wordembeddings,
         posembeddings,
         depembeddings,
         predembeddings) = embedding_data
        role_count = len(roles)
        dep_count = len(depembeddings)


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

                wordembstrain = _create_random_var(wordembeddings.embeddings.shape)

                # part-of-speech embeddings
                posembs = tf.Variable(posembeddings.embeddings,
                                      dtype=tf.float32,
                                      trainable=True,
                                      name='posEmbeddings')

                # predicate embeddings
                predembs = tf.Variable(predembeddings.embeddings,
                                       dtype=tf.float32,
                                       trainable=True,
                                       name='predEmbeddings')

                # dependency  embeddings
                depembs = tf.Variable(depembeddings.embeddings,
                                      dtype=tf.float32,
                                      trainable=True,
                                      name='depEmbeddings')

            #---------------------------- INPUTS -------------------------------
            with tf.name_scope('inputs'):
                # input data and labels placeholders
                self.ph_words = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_words')
                self.ph_pos = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_pos')
                self.ph_pred = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_predicates')
                self.ph_spred = tf.placeholder(tf.int32, [self.batch_size, 2], 'ph_pred_index')
                self.ph_lens = tf.placeholder(tf.int32, [self.batch_size], 'ph_dep')
                self.ph_roles = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_roles')
                self.ph_par = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_par_positions')
                self.ph_dep = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_dependencies')

                batch_size = self.batch_size
                timestep = tf.shape(self.ph_words)[1]

                # get sentences lengths and embeddings
                batch_x_word = tf.nn.embedding_lookup(wordembs, self.ph_words, name='words')
                batch_x_word_train = tf.nn.embedding_lookup(wordembstrain, self.ph_words, name='words_train')
                batch_x_pos = tf.nn.embedding_lookup(posembs, self.ph_pos, name='pos')
                batch_x_pred = tf.nn.embedding_lookup(predembs, self.ph_pred, name='pred')
                batch_x_dep = tf.nn.embedding_lookup(depembs, self.ph_dep, name='dep')
                batch_x_flag = tf.expand_dims(
                    tf.scatter_nd(
                        indices=self.ph_spred,
                        updates=tf.fill([batch_size], 1.0),
                        shape=[batch_size, timestep]),
                    axis=-1)

                # final concatenated input
                batch_x = tf.concat(
                    [batch_x_word,
                     batch_x_word_train,
                     batch_x_pos,
                     batch_x_pred,
                     batch_x_dep,
                     batch_x_flag],
                    axis=-1,
                    name='input')

            #------------------------ LSTM LAYERS ------------------------------
            with tf.name_scope('lstm'):
                batch_lstm = bilstm_layer(batch_x, self.ph_lens)

            #--------------------- graph conv. network -------------------------
            with tf.name_scope('gcn'):
                batch_gcn = gcn_layer_v2(
                    batch_lstm, self.ph_par, self.ph_dep, dep_count)

            #--------------------- predicate encoding --------------------------
            with tf.name_scope('predicate_encoding'):
                # get predicate cell (shape [batch, gcn])
                batch_gcn_predicate = tf.gather_nd(batch_gcn, self.ph_spred)

                # repeat predicate timestep times (shape [batch, timestep, gcn])
                batch_gcn_pred_repeat = tf.tile(
                    tf.expand_dims(batch_gcn_predicate, axis=1),
                    [1, timestep, 1])

                # append predicate info to GCN output
                # shape: [batch_size, timestep, gcn*2]
                batch_gcn_final = tf.concat([batch_gcn, batch_gcn_pred_repeat], axis=-1)

            #------------------------ optimization -----------------------------
            # create prediction logits shape:
            with tf.name_scope('logits'):
                self.logits = tf.layers.dense(batch_gcn_final, role_count)

            # loss function
            with tf.name_scope('loss_function'):
                loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.ph_roles,
                    logits=self.logits,
                    name='softmax')
                losses = tf.boolean_mask(loss_raw, tf.sequence_mask(self.ph_lens))
                loss = tf.reduce_mean(losses)

            # optimizer used for training
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

            #-------------------------------------------------------------------
            with tf.name_scope('meta'):
                # summary info
                if self.profile_data:
                    tf.summary.scalar('loss', loss)
                    self.summary = tf.summary.merge_all()

                # variable initializer (must be executed in order to initialize variables)
                self.init = tf.global_variables_initializer()

                # saver (store and restore model on disk)
                self.saver = tf.train.Saver()


################################################################################
class SequentialModel(OriginalModel):
    '''Neural network using a BiLSTM and GCN stacked together for semantic role labeling.'''

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
                posembs = tf.Variable(posembeddings.embeddings,
                                      dtype=tf.float32,
                                      trainable=True,
                                      name='posEmbeddings')

                # predicate embeddings
                predembs = tf.Variable(predembeddings.embeddings,
                                       dtype=tf.float32,
                                       trainable=True,
                                       name='predEmbeddings')

                # dependency  embeddings
                depembs = tf.Variable(depembeddings.embeddings,
                                      dtype=tf.float32,
                                      trainable=True,
                                      name='depEmbeddings')

            #------------------------------------ INPUTS -------------------------------
            with tf.name_scope('inputs'):
                # input data and labels placeholders
                self.ph_words = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_words')
                self.ph_pos = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_pos')
                self.ph_pred = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_predicates')
                self.ph_spred = tf.placeholder(tf.int32, [self.batch_size, 2], 'ph_pred_index')
                self.ph_lens = tf.placeholder(tf.int32, [self.batch_size], 'ph_dep')
                self.ph_roles = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_roles')
                self.ph_par = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_par_positions')
                self.ph_dep = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_dependencies')

                batch_size = self.batch_size
                timestep = tf.shape(self.ph_words)[1]

                # get sentences lengths and embeddings
                batch_x_word = tf.nn.embedding_lookup(wordembs, self.ph_words, name='words')
                batch_x_pos = tf.nn.embedding_lookup(posembs, self.ph_pos, name='pos')
                batch_x_pred = tf.nn.embedding_lookup(predembs, self.ph_pred, name='pred')
                batch_x_dep = tf.nn.embedding_lookup(depembs, self.ph_dep, name='dep')
                batch_x_flag = tf.expand_dims(
                    tf.scatter_nd(
                        indices=self.ph_spred,
                        updates=tf.fill([batch_size], 1.0),
                        shape=[batch_size, timestep]),
                    axis=-1)

                # final concatenated input
                batch_x = tf.concat(
                    [batch_x_word,
                     batch_x_pos,
                     batch_x_pred,
                     batch_x_dep,
                     batch_x_flag],
                    axis=-1,
                    name='input')

            #-------------------------- LSTM LAYERS ----------------------------
            with tf.name_scope('lstm'): #shape [batch, timestep, cellsize*2]
                batch_lstm = bilstm_layer(batch_x, self.ph_lens)

            #---------------------- graph conv. network ------------------------
            with tf.name_scope('gcn'): #shape [batch, timestep, gcn]
                batch_gcn = gcn_layer(batch_lstm, batch_size, timestep, self.ph_par)

            #---------------------- predicate encoding -------------------------
            with tf.name_scope('predicate_encoding'):
                # get predicate cell (shape [batch, gcn])
                batch_gcn_predicate = tf.gather_nd(batch_gcn, self.ph_spred)
                batch_lstm_predicate = tf.gather_nd(batch_lstm, self.ph_spred)

                # repeat predicate timestep times (shape [batch, timestep, gcn])
                batch_gcn_pred_repeat = tf.tile(
                    tf.expand_dims(batch_gcn_predicate, axis=1),
                    [1, timestep, 1])

                batch_lstm_pred_repeat = tf.tile(
                    tf.expand_dims(batch_lstm_predicate, axis=1),
                    [1, timestep, 1])

                # append predicate info to GCN output
                # shape: [batch_size, timestep, gcn*2]
                batch_hidden_final = tf.concat(
                    [batch_lstm,
                     batch_gcn,
                     batch_lstm_pred_repeat,
                     batch_gcn_pred_repeat],
                    axis=-1)

            #-------------------------- optimization ---------------------------
            # create prediction logits shape:
            with tf.name_scope('logits'):
                self.logits = tf.layers.dense(batch_hidden_final, role_count)

            # loss function
            with tf.name_scope('loss_function'):
                loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.ph_roles,
                    logits=self.logits,
                    name='softmax')
                losses = tf.boolean_mask(loss_raw, tf.sequence_mask(self.ph_lens))
                loss = tf.reduce_mean(losses)

            # optimizer used for training
            with tf.name_scope('optimizer'):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(
                        loss,
                        global_step=global_step)

            #-------------------------------------------------------------------
            with tf.name_scope('meta'):
                # summary info
                if self.profile_data:
                    tf.summary.scalar('loss', loss)
                    self.summary = tf.summary.merge_all()

                # variable initializer (must be executed in order to initialize variables)
                self.init = tf.global_variables_initializer()

                # saver (store and restore model on disk)
                self.saver = tf.train.Saver()


################################################################################
class ParallelGatedModel(OriginalModel):
    '''Neural network using a BiLSTM and GCN parallelly together for semantic role labeling.
    
    This model uses a BiLSTM and GCN layer parallelly in order to compute the 
    probabilities of semantic roles.
    The two layers are joined together through the classifier layer.
    '''

    # function used in order to create the tensorflow graph
    def _create_graph(self, embedding_data, roles):
        (wordembeddings,
         posembeddings,
         depembeddings,
         predembeddings) = embedding_data
        role_count = len(roles)
        dep_count = len(depembeddings)


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

                wordembstrain = _create_random_var(wordembeddings.embeddings.shape)

                # part-of-speech embeddings
                posembs = tf.Variable(posembeddings.embeddings,
                                      dtype=tf.float32,
                                      trainable=True,
                                      name='posEmbeddings')

                # predicate embeddings
                predembs = tf.Variable(predembeddings.embeddings,
                                       dtype=tf.float32,
                                       trainable=True,
                                       name='predEmbeddings')

                # dependency  embeddings
                depembs = tf.Variable(depembeddings.embeddings,
                                      dtype=tf.float32,
                                      trainable=True,
                                      name='depEmbeddings')

            #---------------------------- INPUTS -------------------------------
            with tf.name_scope('inputs'):
                # input data and labels placeholders
                self.ph_words = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_words')
                self.ph_pos = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_pos')
                self.ph_pred = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_predicates')
                self.ph_spred = tf.placeholder(tf.int32, [self.batch_size, 2], 'ph_pred_index')
                self.ph_lens = tf.placeholder(tf.int32, [self.batch_size], 'ph_dep')
                self.ph_roles = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_roles')
                self.ph_par = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_par_positions')
                self.ph_dep = tf.placeholder(tf.int32, [self.batch_size, None], 'ph_dependencies')

                batch_size = self.batch_size
                timestep = tf.shape(self.ph_words)[1]

                # get sentences lengths and embeddings
                batch_x_word = tf.nn.embedding_lookup(wordembs, self.ph_words, name='words')
                batch_x_word_train = tf.nn.embedding_lookup(wordembstrain, self.ph_words, name='words_train')
                batch_x_pos = tf.nn.embedding_lookup(posembs, self.ph_pos, name='pos')
                batch_x_pred = tf.nn.embedding_lookup(predembs, self.ph_pred, name='pred')
                batch_x_dep = tf.nn.embedding_lookup(depembs, self.ph_dep, name='dep')
                batch_x_flag = tf.expand_dims(
                    tf.scatter_nd(
                        indices=self.ph_spred,
                        updates=tf.fill([batch_size], 1.0),
                        shape=[batch_size, timestep]),
                    axis=-1)

                # final concatenated input
                batch_x = tf.concat(
                    [batch_x_word,
                     batch_x_word_train,
                     batch_x_pos,
                     batch_x_pred,
                     batch_x_dep,
                     batch_x_flag],
                    axis=-1,
                    name='input')

    #------------------------------- LSTM LAYERS ------------------------------
            with tf.name_scope('lstm'):
                batch_lstm = bilstm_layer(batch_x, self.ph_lens)

    #--------------------------- graph conv. network ---------------------------
            with tf.name_scope('gcn'):
                batch_gcn = gcn_layer(
                    batch_x, batch_size, timestep, self.ph_par)

            #----------------------- predicate encoding ------------------------
            with tf.name_scope('predicate_encoding'):
                # repeat predicate timestep times (shape [batch, timestep, gcn])
                batch_lstm_predicate = tf.tile(
                    tf.expand_dims(tf.gather_nd(batch_lstm, self.ph_spred), axis=1),
                    [1, timestep, 1])

                batch_gcn_predicate = tf.tile(
                    tf.expand_dims(tf.gather_nd(batch_gcn, self.ph_spred), axis=1),
                    [1, timestep, 1])

                # append predicate info to GCN output
                # shape: [batch_size, timestep, gcn*2]
                batch_lstm_final = tf.concat([batch_lstm, batch_lstm_predicate], axis=-1)
                batch_gcn_final = tf.concat([batch_gcn, batch_gcn_predicate], axis=-1)

            #-------------------------- optimization ---------------------------
            # create prediction logits shape:
            with tf.name_scope('logits'):
                gate_bilstm = tf.layers.dense(batch_lstm_final, 1, tf.sigmoid)
                gate_gcn = tf.layers.dense(batch_gcn_final, 1, tf.sigmoid)
                logits_lstm = tf.layers.dense(batch_lstm_final, role_count)*gate_bilstm
                logits_gcn = tf.layers.dense(batch_gcn_final, role_count)*gate_gcn
                self.logits = logits_lstm + logits_gcn

            # loss function
            with tf.name_scope('loss_function'):
                loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.ph_roles,
                    logits=self.logits,
                    name='softmax')
                losses = tf.boolean_mask(loss_raw, tf.sequence_mask(self.ph_lens))
                loss = tf.reduce_mean(losses)

            # optimizer used for training
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

            #-------------------------------------------------------------------
            with tf.name_scope('meta'):
                # summary info
                if self.profile_data:
                    tf.summary.scalar('loss', loss)
                    self.summary = tf.summary.merge_all()

                # variable initializer (must be executed in order to initialize variables)
                self.init = tf.global_variables_initializer()

                # saver (store and restore model on disk)
                self.saver = tf.train.Saver()


################################################################################

# create a randomly initialized variable with given shape
def _create_random_var(shape):
    return  tf.Variable(tf.random_normal(shape, dtype=tf.float32), trainable=True)

# return tensor of shape [batch, timestep, gcn] representing the gate for each node in timestep
def _create_gate(input_tensor):
    return tf.tile(tf.layers.dense(input_tensor, 1, tf.sigmoid), [1, 1, GCN_SIZE])

#-------------------------------------------------------------------------------
# return tensor of shape [batch, timestep, gcn] representing the gate for each node in timestep
def _create_gate_v2(input_tensor, dep_sentence, dep_count):
    dimensions = int(input_tensor.get_shape()[-1])
    weights = _create_random_var([dimensions, 1])
    dep_embeddings = _create_random_var([dep_count, 1])
    biases = tf.nn.embedding_lookup(dep_embeddings, dep_sentence)

    res = tf.sigmoid(tf.tensordot(input_tensor, weights, axes=1) + biases)
    return tf.tile(res, [1, 1, GCN_SIZE])

# create a special dependency base dense layer
def _create_dense_v2(input_tensor, dep_sentence, dep_count):
    dimensions = int(input_tensor.get_shape()[-1])
    weights = _create_random_var([dimensions, GCN_SIZE])
    dep_embeddings = _create_random_var([dep_count, GCN_SIZE])
    biases = tf.nn.embedding_lookup(dep_embeddings, dep_sentence)
    return tf.tensordot(input_tensor, weights, axes=1) + biases

#-------------------------------------------------------------------------------
def gcn_layer(input_gcn, batch_size, timestep, parent_positions):
    '''Create a GCN layer (slightly different from the article implementation)'''
    parent_positions_flat = tf.reshape(
        parent_positions + tf.expand_dims(
            tf.range(batch_size, dtype=tf.int32)*timestep, axis=1),
        [batch_size*timestep])

    input_size = int(input_gcn.shape[-1])
    input_gcn_flat = tf.reshape(input_gcn, [batch_size*timestep, input_size])
    par_gcn_flat = tf.gather(input_gcn_flat, parent_positions_flat)
    par_gcn = tf.reshape(par_gcn_flat, [batch_size, timestep, input_size])

    #shape [batch, timestep, gcn] (last dim. tiled)
    par_gate = _create_gate(par_gcn)
    loop_gate = _create_gate(input_gcn)
    child_gate = _create_gate(input_gcn)

    #shape [batch, timestep, gcn]
    loop_conv = tf.multiply(loop_gate, tf.layers.dense(input_gcn, GCN_SIZE))
    par_conv = tf.multiply(par_gate, tf.layers.dense(par_gcn, GCN_SIZE))
    child_matmul = tf.multiply(child_gate, tf.layers.dense(input_gcn, GCN_SIZE))

    adj_matrices = tf.one_hot(parent_positions, timestep, dtype=tf.float32)
    rec_deg_vec = tf.minimum(tf.reciprocal(tf.reduce_sum(adj_matrices, axis=1)), 1.0)
    diag_deg = tf.matrix_diag(rec_deg_vec)

    adj_mat_norm = tf.matmul(a=diag_deg, b=adj_matrices, transpose_b=True)
    children_conv = tf.matmul(a=adj_mat_norm, b=child_matmul, transpose_a=True)

    #shape [batch, timestep, gcn]
    batch_gcn = tf.nn.leaky_relu(par_conv + loop_conv + children_conv)
    input_gcn = batch_gcn
    return batch_gcn

def gcn_layer_v2(input_gcn, parent_positions, sentence_deps, dep_count):
    '''Create a GCN layer faithful to the article implementation'''
    batch_size = input_gcn.get_shape()[0]
    timestep = tf.shape(input_gcn)[1]

    parent_positions_flat = tf.reshape(
        parent_positions + tf.expand_dims(
            tf.range(batch_size, dtype=tf.int32)*timestep, axis=1),
        [batch_size*timestep])

    input_size = int(input_gcn.shape[-1])
    input_gcn_flat = tf.reshape(input_gcn, [batch_size*timestep, input_size])
    par_gcn_flat = tf.gather(input_gcn_flat, parent_positions_flat)
    par_gcn = tf.reshape(par_gcn_flat, [batch_size, timestep, input_size])

    #shape [batch, timestep, gcn] (last dim. tiled)
    loop_gate = _create_gate(input_gcn)
    par_gate = _create_gate_v2(input_gcn, sentence_deps, dep_count)
    child_gate = _create_gate_v2(par_gcn, sentence_deps, dep_count)

    #shape [batch, timestep, gcn]
    loop_conv = tf.multiply(loop_gate, tf.layers.dense(input_gcn, GCN_SIZE))
    par_conv = tf.multiply(par_gate, _create_dense_v2(par_gcn, sentence_deps, dep_count))
    child_matmul = tf.multiply(child_gate, _create_dense_v2(input_gcn, sentence_deps, dep_count))

    adj_matrices = tf.one_hot(parent_positions, timestep, dtype=tf.float32)
    children_conv = tf.matmul(a=adj_matrices, b=child_matmul, transpose_a=True)

    #shape [batch, timestep, gcn]
    batch_gcn = tf.nn.leaky_relu(par_conv + loop_conv + children_conv)
    input_gcn = batch_gcn
    return batch_gcn

def bilstm_layer(input_tensor, sequence_lengths, num_layers=1):
    '''Create K BiLSTM layers'''
    cellsf, cellsb = [], []
    for _ in range(num_layers):
        cellsf.append(tf.contrib.rnn.BasicLSTMCell(CELL_SIZE))
        cellsb.append(tf.contrib.rnn.BasicLSTMCell(CELL_SIZE))

    # multi-layered dynamic BiLSTM
    batch_lstm, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cellsf, cellsb,
        inputs=input_tensor,
        sequence_length=sequence_lengths,
        dtype=tf.float32,
        scope='lstm_subgraph')
    return batch_lstm
