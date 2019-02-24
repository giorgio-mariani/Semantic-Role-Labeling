# -*- coding: utf-8 -*-
'''Contain functions and classes used during preprocessing of the input data.'''

import gensim
import numpy as np

UNK = 'unk'
UNK_INDEX = 0

NULL_ROLE = '_'
NULL_INDEX = 0


SIZE_POS_EMBS = 16
SIZE_DEP_EMBS = 32
SIZE_PRED_EMBS = 64

#------------------------------------------------------------------------------
class Embeddings:
    '''Maintain information about word embeddings.
    
    Attributes:
        embeddings -- a numpy array containing the actual embeddings
        w2i -- dictionary that maps from word to emb. indices
        i2w -- array that maps indices to words
    '''

    def __init__(self, index2word, word2index=None, embeddings=None, size=64):
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            np.random.seed(seed = 0)
            self.embeddings = np.random.normal(size=[len(index2word), size])

        if word2index == None:
            word2index = {w:i for i,w in enumerate(index2word)}

        self.w2i = word2index
        self.i2w = index2word

        self.emb_num = self.embeddings.shape[0]
        self.emb_dim = self.embeddings.shape[1]

    def __getitem__(self, word):
        if isinstance(word, str):
            return self.embeddings[self.w2i[word]]
        else:
            return self.embeddings[word]

    def __contains__(self, key):
        if not isinstance(key, str):
            raise Exception('value must be a string')
        return key in self.w2i

    def __len__(self):
        return len(self.i2w)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return  (other.w2i == self.w2i and 
                    other.i2w == self.i2w and 
                    np.array_equal(other.embeddings,self.embeddings))          
        else:
            return False

#------------------------------------------------------------------------------
class Classes:
    '''Maintain information about classification labels.

    The goal of this class is simply to maintain a map between label names and their
    numeric values.

    Attributes:
        c2i -- dictionary that maps from label to its id.
        i2c -- array that maps label ids to actual labels
    '''
    def __init__(self, classes):
        self.c2i = {}
        self.i2c = list()
        for c in classes: 
            self.addclass(c)
    
    def addclass(self, newclass):
        if not isinstance(newclass, str):
            raise Exception('newclass must be a string')
        
        if newclass not in self.c2i:
            self.c2i[newclass] = len(self.i2c)
            self.i2c.append(newclass)
    
    def __getitem__(self,cls):
        if isinstance(cls, str):
            return self.c2i[cls] 
        else:
            return self.i2c[cls]
        
    def __contains__(self, key):
        if not isinstance(key, str):
            raise Exception('value must be a string')
        return key in self.c2i
    
    def __len__(self):
        return len(self.i2c)
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return  (other.c2i == self.c2i and 
                     other.i2c == self.i2c)
        else:
            return False

#------------------------------------------------------------------------------
class SentenceData:
    '''Class containing raw sentence information.'''

    @staticmethod
    def get_sentence_data(file_obj):
        '''Create a new SentenceData object by reading the input file'''

        sentence_data = SentenceData()
        sentence_is_over, file_is_over = False, False

        while not sentence_is_over:
            row = file_obj.readline()
            if not row:
                file_is_over = True
                sentence_is_over = True
            else:
                sentence_is_over = sentence_data._add_word(row)

        # check whether the gathered data is sensible or not, if not then return None
        if not len(sentence_data):
            sentence_data = None
        return sentence_data, file_is_over

    def __init__(self):
        self.sentence_ids = []
        self.words = []
        self.lemmas = []
        self.pos_tags = []
        self.predicates = []
        self.predicate_positions = []
        self.parent_positions = []
        self.dependencies = []
        self.roles = []

        self.plemmas = []
        self.pdependencies = []
        self.pparents = []
        self.ppos_tags = []

    def __len__(self):
        return len(self.words)

    def _add_word(self, raw_line):
        #remove newline from sentence
        raw_line = raw_line[:-1]

         # check if sentence is over 
        if not raw_line:
            return True #return that the sentence is over)

        tokens = raw_line.split('\t')
        sid, word, lemma = tokens[0], tokens[1], tokens[2]
        pos_tag, parent, dep = tokens[4], tokens[8], tokens[10]
        ispred, predicate = tokens[12], tokens[13]
        roles = tokens[14:]
        
        plemma = tokens[3]
        ppos_tag = tokens[5]
        pparent = tokens[9]
        pdep = tokens[11]

        self.sentence_ids.append(sid)
        self.words.append(word)
        self.lemmas.append(lemma)
        self.pos_tags.append(pos_tag)
        self.dependencies.append(dep)
        self.predicates.append(predicate)
        self.parent_positions.append(parent)
        self.roles.append(roles)
        
        self.plemmas.append(plemma)
        self.ppos_tags.append(ppos_tag)
        self.pparents.append(pparent)
        self.pdependencies.append(pdep)

        return False

    def digest(self, classes,
               wordembeddings, posembeddings,
               depembeddings, predembeddings):
        '''Digest the raw text information into indices using embeddings and semantic role indices.

        Keyword arguments:
            classes -- instance of preprocessing.Classes
                containing semantic roles.

            wordembeddings -- instance of preprocessing.Embeddings
                containing word embeddings.

            posembeddings -- instance of preprocessing.Embeddings
                containing PoS embeddings.

            depembeddings -- instance of preprocessing.Embeddings
                containing dependency-type embeddings.

            predembeddings -- instance of preprocessing.Embeddings
                containing predicate embeddings.


        Return:
            SentenceDataDigested instance.
        '''

        if self:
            return SentenceDataDigested(self, classes,
                                        wordembeddings, posembeddings,
                                        depembeddings, predembeddings)
        else:
            return None

    def __str__(self):
        lines = []
        for i in range(len(self)):
            sid, word, lemma = self.sentence_ids[i], self.words[i], self.lemmas[i]
            pos_tag = self.pos_tags[i]
            parent = self.parent_positions[i]
            dep  = self.dependencies[i]
            predicate = self.predicates[i]
            roles = self.roles[i]

            plemma = self.plemmas[i]
            pdep = self.pdependencies[i]
            pparent = self.pparents[i]
            ppos_tag = self.ppos_tags[i]

            ispred = '_' if predicate == '_' else 'Y'
            line = [sid, word, lemma, plemma, pos_tag, ppos_tag, '_', '_',
                    parent, pparent, dep, pdep, ispred, predicate]
            for role in roles:
                line.append(role)

            line_text = '\t'.join(line)+'\n'
            lines.append(line_text)

        return ''.join(lines)

class SentenceDataDigested():
    '''Class containing digested sentence information, ready for batch creation.

    Differently than preprocessing.SentenceData instances, object of
    this class do not maintain text information, instead storing the 
    corresponding indices or digested numerical values (like for dep. heads).

    Data contained in an instance of this class can be used for generating 
    batches using the utils.BatchGenerator class
    '''

    def __len__(self):
        return len(self.words)

    def predicate_count(self):
        return len(self.predicate_positions)

    def __init__(self, sentence_data, classes,
                 wordembeddings, posembeddings,
                 depembeddings, predembeddings):

        self.words = []
        self.pos_tags = []
        self.dependencies = []
        self.predicates = []
        self.predicate_positions = []
        self.roles = []
        self.parent_positions = []

        for wordcounter in range(len(sentence_data)):
            word = sentence_data.words[wordcounter]
            lemma = sentence_data.lemmas[wordcounter]
            pos_tag = sentence_data.pos_tags[wordcounter]
            parent_position = sentence_data.parent_positions[wordcounter]
            dependency = sentence_data.dependencies[wordcounter]
            predicate = sentence_data.predicates[wordcounter]
            sentence_roles = sentence_data.roles[wordcounter]

            def _getindex_(word, emb):
                return emb.w2i[word] if word in emb else emb.w2i[UNK]

            # keep track of sentence info
            word = word.lower() if pos_tag[:2] == 'VB' else lemma
            self.words.append(_getindex_(word, wordembeddings))
            self.predicates.append(_getindex_(predicate, predembeddings))

            self.pos_tags.append(posembeddings.w2i[pos_tag])
            self.dependencies.append(depembeddings.w2i[dependency])

            parent_position = int(parent_position)-1
            parent_position = wordcounter if parent_position == -1 else parent_position
            self.parent_positions.append(parent_position)

            # if word is a sentence predicate add its position to predicate_positions list
            if predicate != '_':
                self.predicate_positions.append(wordcounter)

            self.roles.append([classes.c2i[r] for r in sentence_roles])

        # reorganize roles
        self.roles = np.array(self.roles, dtype=np.int32).transpose()

#-------------------------------------------------------------------------------

def get_glove_word2vec(embedding_file):
    '''Load the GloVe word embeddings and move UNK vector to position 0
    
    Keyword arguments:
        embedding_file -- file containing the binary GloVe embedding data.
    '''
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)  
    w2i, i2w = dict(), dict()
    for k,v in embeddings.vocab.items():
        w2i[k] = v.index
        i2w[v.index] = k

    # get embeddings vector
    embs = embeddings.syn0

    # prepare swap data
    null_word = 'unk'
    null_index = w2i[null_word]
    null_emb = embs[null_index]

    first_index = 0
    first_word = i2w[first_index]
    first_emb = embs[first_index]

    # swap embeddings
    embs[first_index] = null_emb
    embs[null_index] = first_emb

    # swap words
    i2w[first_index] = null_word
    i2w[null_index] = first_word

    # swap indices
    w2i[first_word] = null_index
    w2i[null_word] = first_index

    return Embeddings(i2w, w2i, embs)

def generate_embeddings(input_filename):
    '''Generate and return PoS, dependency and predicate embeddings and semantic roles.

    This function generates and returns (from a file with the CoNLL 2009 format)
    embeddings for PoS, dependency-types and predicates.
    Those embeddings are stored in a preprocessing.Embeddings 
    object.

    A Classes instance, containing information about semantic roles, is also
    generated and returned.

    Returns:
        embedding_data -- tuple containing preprocessing.Embedding
            instances of shape: (PoS, dep-type, predicate)

        classes -- an istance of preprocessing.Classes
            encasing information about semantic roles.
        '''
    with open(input_filename, 'r', encoding='utf-8') as file:
        pos_set = list([NULL_ROLE])
        dep_set = list([NULL_ROLE])
        pred_set = list([NULL_ROLE, UNK])
        arg_set = list([NULL_ROLE])

        def _add_(coll, obj):
            if obj not in coll:
                coll.append(obj)

        for row in file:
            #remove newline
            row = row[:-1]

            if not row:
                continue # new sentence

            tokens = row.split('\t')
            pos, dep, pred = tokens[4], tokens[10], tokens[13]

            _add_(pos_set, pos)
            _add_(dep_set, dep)
            _add_(pred_set, pred)

            for arg in tokens[14:]:
                _add_(arg_set, arg)

        posembeddings = Embeddings(pos_set, size=SIZE_POS_EMBS)
        depembeddings = Embeddings(dep_set, size=SIZE_DEP_EMBS)
        predembeddings = Embeddings(pred_set, size=SIZE_PRED_EMBS)

        embeddings = (posembeddings, depembeddings, predembeddings)
        classes = Classes(arg_set)

    return embeddings, classes

def load_data(input_file, classes,
              wordembeddings,
              posembeddings,
              depembeddings,
              predembeddings,
              num=None):
    '''Load data from input file into a sequence of preprocessing.SentenceDataDigested objects

    Keyword arguments:
        input_file -- name of file containing the 
            data to load (CoNLL 2009 format)

        classes -- instance of preprocessing.Classes 
            containing semantic roles.

        wordembeddings -- instance of preprocessing.Embeddings
            containing word embeddings.

        posembeddings -- instance of preprocessing.Embeddings
            containing PoS embeddings.

        depembeddings -- instance of preprocessing.Embeddings
            containing dependency-type embeddings.

        predembeddings -- instance of preprocessing.Embeddings
            containing predicate embeddings.

        num -- number of sentence to read.


    Return:
        data -- list of prepocessing.SentenceDataDigested
            instances encapsulating sentence information.
    '''

    # load input data
    with open(input_file, 'r', encoding='utf-8') as file:
        data, fileisover, sentence_counter = list(), False, 0
        while not fileisover and (not num or sentence_counter < num):
            (sentence_data, fileisover) = SentenceData.get_sentence_data(file)
            if sentence_data:
                digested_data = sentence_data.digest(
                    classes, wordembeddings, posembeddings,
                    depembeddings, predembeddings)

                data.append(digested_data)
                sentence_counter += 1
    return data

#-------------------------------------------------------------------------------
def create_prediction_file(input_file,
                           output_file,
                           model,
                           batch_size,
                           classes,
                           embedding_data):
    '''Create a prediction file, containing the predict semantic roles for the input sentences.

    This function creates an output file (in the CoNLL 2009 format) containing the
    predicted semantic roles for the sentences in the input file.

    Keyword arguments:
        input_file -- name of file containing the
            sentences for the SRL task.

        output_file -- name of the file that will
            contain the predicted roles.

        model -- instance of srl_models.Model used
            for the prediction.

        batch_size -- batch size used by the model
    '''

    from utils import Batch_Generator
    wordembeddings, posembeddings, depembeddings, predembeddings = embedding_data

    # load input data
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            with model._get_session() as session:
                # initialize variables
                model.saver.restore(session, model.savefile)
                fileisover = False
                while not fileisover:
                    sentence_data, fileisover = SentenceData.get_sentence_data(infile)

                    if sentence_data:
                        digested_data = sentence_data.digest(
                            classes=classes,
                            wordembeddings=wordembeddings,
                            posembeddings=posembeddings,
                            depembeddings=depembeddings,
                            predembeddings=predembeddings)

                        digested_data.roles = np.zeros([digested_data.predicate_count(),
                                                        len(digested_data)], dtype=np.int32)

                        def i2roles(array_like):
                            array, out = np.array(array_like), []
                            shape = array.shape
                            array_flat = np.reshape(array,[-1])
                            for i in array_flat: out.append(classes.i2c[i])
                            return np.reshape(out, shape)


                        if digested_data.predicate_count():
                            generator = Batch_Generator([digested_data], batch_size)
                            total_predictions = []
                            for _ in range(len(generator)):
                                batch = generator.generate_next_batch()
                                predictions = model.predict(session, batch)
                                endindex = min(generator.batch_size, digested_data.predicate_count())
                                predictions = predictions.transpose()[:, :endindex]
                                predictions_text = i2roles(predictions)
                                total_predictions.extend(predictions_text)
                            sentence_data.roles = total_predictions
                        outfile.write(str(sentence_data)+"\n")
