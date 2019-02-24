# -*- coding: utf-8 -*-
import math
import numpy as np
import preprocessing as pr

class Batch:
    '''Maintain information ready to be used by a Tensorflow graph.'''

    def __init__(self,
                 words,
                 pos_tags,
                 predicates,
                 predicate_positions,
                 dependencies,
                 parent_positions,
                 roles,
                 sequence_lengths,
                 sentence_indices):
        self.words = words
        self.pos = pos_tags
        self.predicates = predicates
        self.predicate_positions = predicate_positions
        self.dependencies = dependencies
        self.parent_positions = parent_positions
        self.roles = roles
        self.sequence_lengths = sequence_lengths
        self.sentence_indices = sentence_indices

    def __len__(self):
        return len(self.words.shape[0])

class Batch_Generator:
    '''Generate batches by reading a sequence of input_data.

    An instance of this class generates batches (utils.Batch) of data, starting
    from a list of preprocessing.SentenceDataDigested and a batch size.
    Sentences in batches are padded wrt to longest in the batch.
    '''

    def __init__(self, data, batch_size):
        self.current_sentence_index = 0
        self.batch_size = batch_size
        self.data = data

        # compute data length
        self.data_length = 0
        for sentence_data in data:
            self.data_length += sentence_data.predicate_count()

        # define buffer data
        self._buffer_words = list()
        self._buffer_pos = list()
        self._buffer_predicates = list()
        self._buffer_predicate_positions = list()
        self._buffer_dependencies = list()
        self._buffer_parent_positions = list()
        self._buffer_roles = list()
        self._buffer_sentence_index = list()

    def __len__(self):
        return math.ceil(self.data_length/self.batch_size)

    def _buffer_length(self):
        return len(self._buffer_words)

    def _add_sentence_to_buffer(self):
        sentence_index = self.current_sentence_index
        sentence_data = self.data[sentence_index]
        self.current_sentence_index = (sentence_index + 1)%len(self.data)

        for i in range(sentence_data.predicate_count()):
            self._buffer_words.append(sentence_data.words)
            self._buffer_pos.append(sentence_data.pos_tags)
            self._buffer_predicates.append(sentence_data.predicates)
            self._buffer_predicate_positions.append(sentence_data.predicate_positions[i])
            self._buffer_dependencies.append(sentence_data.dependencies)
            self._buffer_parent_positions.append(sentence_data.parent_positions)
            self._buffer_roles.append(sentence_data.roles[i])
            self._buffer_sentence_index.append(sentence_index)

    def _get_batch(self):
        words = self._buffer_words[:self.batch_size]
        pos_tags = self._buffer_pos[:self.batch_size]
        predicates = self._buffer_predicates[:self.batch_size]
        predicate_positions = self._buffer_predicate_positions[:self.batch_size]
        dependencies = self._buffer_dependencies[:self.batch_size]
        parent_positions = self._buffer_parent_positions[:self.batch_size]
        roles = self._buffer_roles[:self.batch_size]
        sentence_indices = self._buffer_sentence_index[:self.batch_size]

        self._buffer_words[:self.batch_size] = []
        self._buffer_pos[:self.batch_size] = []
        self._buffer_predicates[:self.batch_size] = []
        self._buffer_predicate_positions[:self.batch_size] = []
        self._buffer_dependencies[:self.batch_size] = []
        self._buffer_parent_positions[:self.batch_size] = []
        self._buffer_roles[:self.batch_size] = []
        self._buffer_sentence_index[:self.batch_size] = []

        maxlen = len(max(words, key=len))
        shape = [self.batch_size, maxlen]

        words_np = np.full(shape, pr.NULL_INDEX, dtype=np.int32)
        pos_tags_np = np.full(shape, pr.NULL_INDEX, dtype=np.int32)
        predicates_np = np.full(shape, pr.NULL_INDEX, dtype=np.int32)
        dependencies_np = np.full(shape, pr.NULL_INDEX, dtype=np.int32)
        roles_np = np.full(shape, pr.NULL_INDEX, dtype=np.int32)
        parent_positions_np = np.tile(
            np.expand_dims(
                np.arange(maxlen, dtype=np.int32),
                axis=0),
            [self.batch_size, 1])

        def _initialize_np_(arraylist, nparray):
            for i in range(len(arraylist)):
                nparray[i, :len(arraylist[i])] = arraylist[i]

        _initialize_np_(words, words_np)
        _initialize_np_(pos_tags, pos_tags_np)
        _initialize_np_(predicates, predicates_np)
        _initialize_np_(dependencies, dependencies_np)
        _initialize_np_(parent_positions, parent_positions_np)
        _initialize_np_(roles, roles_np)

        predicate_positions_np = np.array([[i, p] for i, p in enumerate(predicate_positions)])
        sentence_indices_np = np.array(sentence_indices, dtype=np.int32)
        sequence_length_np = np.array([len(s) for s in words])

        batch = Batch(words=words_np,
                      pos_tags=pos_tags_np,
                      predicates=predicates_np,
                      predicate_positions=predicate_positions_np,
                      dependencies=dependencies_np,
                      parent_positions=parent_positions_np,
                      roles=roles_np,
                      sequence_lengths=sequence_length_np,
                      sentence_indices=sentence_indices_np)
        return batch

    def generate_next_batch(self):
        '''Generate next batch of data, if the input data is over then it cycles back to the first element.'''

        while self._buffer_length() < self.batch_size:
            self._add_sentence_to_buffer()
        return self._get_batch()


###############################################################################
