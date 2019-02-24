# -*- coding: utf-8 -*-
'''
Contain the abtract class Model.

The module contains the class Model, which is a wrapper for a generic
Tensorflow graph of a model for the SRL task.
'''
import tensorflow  as tf

import numpy as np
from tqdm import tqdm as tqdm

from utils import Batch_Generator

#------------------------------------------------------------------------------

class Model:
    '''Abstract class, wrapper for a Tensorflow graph representing a SRL neural model.'''

    # function used in order to create the tensorflow graph
    def _create_graph(self, embedding_data, roles):
        '''Create the Tensorflow graph relative to this SRL model.

        Keyword arguments:
            self -- SRL model.
            embedding_data -- tuple containing different embedding
                              data.
            roles --instance of preprocessing.Classes that represents
                    the semantic roles.
        '''
        raise NotImplementedError

    def _get_feed_dict(self, batch):
        '''Return a dict obj to use as feed_dict argument for tensorflow sessions.

        This methods return a dict obj that can be used as feed_dict argument
        when working with tensorflow sessions that use the model.

        Keyword arguments:
            self -- SRL model.
            batch -- A utils.Batch object, containg the batch data
        '''
        raise NotImplementedError

#==============================================================================
    def _get_session(self):
        if self.use_GPU:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config, graph=self.graph)
        else:
            session = tf.Session(graph=self.graph)
        return session

    def  __init__(self,
                  embedding_data,
                  roles,
                  savefile=None,
                  profiledata=False,
                  batch_size=20,
                  learning_rate=0.001,
                  use_GPU=False):

        self.savefile = savefile
        self.learning_rate = learning_rate
        self.profile_data = profiledata
        self.batch_size = batch_size
        self.use_GPU = use_GPU

        # useful tensors
        self.optimizer = None
        self.logits = None

        # meta variables
        self.init = None
        self.saver = None
        self.summary = None
        self.global_step = None
        self.graph = None

        # generating model
        self._create_graph(embedding_data, roles)


#===============================================================================
    # function used to train the model
    def train_model(self, train_data, epoch_number=1):
        '''Train the underlying Tensorflow graph variables.

        Keyword arguments:
            self -- SRL model
            train_data -- training data, list of
                          preprocessing.SentenceData
                          instances.
            epoch_number -- number of epochs used
                            during training.
        '''
        # Begin training
        with self._get_session() as session:
            # initialize variables
            self.init.run()

            # generate stream of data
            batch_generator = Batch_Generator(train_data, self.batch_size)

            # if possible restore the variables' values from a previous session
            try:
                self.saver.restore(session, self.savefile)
            except Exception as exp:
                print(exp.message)
            writer = tf.summary.FileWriter(self.savefile, graph=self.graph)

            # start a new epoch -----------------------------------------------
            for epoch in range(epoch_number):
                print("Epoch number: "+str(epoch+1))

                # start a new iteration in the epoch --------------------------
                for step in tqdm(range(len(batch_generator))):
                    # get batch data
                    batch = batch_generator.generate_next_batch()
                    feed_dict = self._get_feed_dict(batch)

                    # take summary
                    if step % 50 and self.savefile is not None and self.profile_data:
                        summ = session.run(self.summary, feed_dict=feed_dict)
                        writer.add_summary(summ, step)

                    # optimize weights
                    session.run(self.optimizer, feed_dict=feed_dict)
                    # end of iteration -----------------------------------------

                # saving graph variables after epoch
                print('saving graph variables ...')
                self.saver.save(session, self.savefile)
                # end of epoch and iteration -----------------------------------

#===============================================================================
    def evaluate_model(self, eval_data):
        '''Evaluate the neural model with respect to some evaluation data.

        This methods returns the model's performance.

        Keyword arguments:
            self -- SRL model
            eval_data -- data used for evaluation (it must )

        Returns:
            Precision
            Recall
            F1 measure
        '''

        # Begin training
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        given, present = 0, 0

        with self._get_session() as session:
            # initialize variables
            self.saver.restore(session, self.savefile)
            batch_generator = Batch_Generator(eval_data, self.batch_size)

            for _ in tqdm(range(len(batch_generator))):
                batch = batch_generator.generate_next_batch()
                roles = batch.roles
                seq_lens = batch.sequence_lengths
                feed_dict = self._get_feed_dict(batch)

                # get logits
                logits = session.run(self.logits, feed_dict=feed_dict)

                # get predictions
                predictions = np.argmax(logits, axis=2)

                # for each prediction check if it was correct
                for i in range(len(seq_lens)):
                    for j in range(seq_lens[i]):
                        role_is_null = roles[i][j] == 0
                        pred_is_null = predictions[i][j] == 0

                        if predictions[i][j] == roles[i][j]:
                            if  role_is_null:
                                true_negative += 1
                            else:
                                true_positive += 1
                        else:
                            if pred_is_null:
                                false_negative += 1
                            else:
                                false_positive += 1

                        if predictions[i][j] != 0:
                            given += 1

                        if not role_is_null:
                            present += 1

            precision = true_positive/given
            recall = true_positive/present
            f_measure = precision*recall*2/(precision+recall)

            '''
            with open('../tmp/eval_2_'+self.__class__.__name__+'.txt', 'w') as file:
                file.write('\n\nTrue positives:'+str(true_positive)+'\nFalse positives:'+str(false_positive)+'\nTrue negatives:'+str(true_negative)+'\nFalse negatives:'+str(false_negative)+'\n')
                file.write('\n\nTotal Precision:'+str(precision)+'\nTotal Recall:'+str(recall)+'\nTotal f1-measure:'+str(f_measure)+'\n')
            '''
            return precision, recall, f_measure


#==============================================================================
    def predict(self, session, batch):
        # initialize variables
        feed_dict = self._get_feed_dict(batch)
        logits = session.run(self.logits, feed_dict=feed_dict)
        predictions = np.argmax(logits, axis=2)
        return predictions


###############################################################################
