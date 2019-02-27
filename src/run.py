# -*- coding: utf-8 -*-
'''main script of the project.

This script contains a single function that execute the semantic role labeling task.
The function is diveded into the following phases:
 - Preprocessing of data.
 - model graph construction
 - model training
 - model evaluation
 - test data prediction
'''
import preprocessing as pr


# Constants
GLOVE_EMBEDDINGS_FILE = '../data/glove.6B.100d.bin'

# main function
'''def semantic_role_labeling(
        model_constructor,
        training_data,
        eval_data,
        test_data,
        output_file,
        save_file,
        profile_data=False,
        batch_size=20,
        epoch_number=1,
        learning_rate=0.001,
        use_gpu=False):
    """Traning, evaluation and prediction for semantic role labeling.

    This function is used to train a SRL DNN classifier using a training dataset,
    evaluate it on a certain evaluation dataset, and finally it is used for
    prediction of SRL of some given test data.

    Keyword arguments:
        model_constructor -- function called in order to obtain
                             the NN model (must be a
                             srl_models.Model sub-class)

        training_data -- name of file containing the training
                         data (must be CoNLL 2009 format compliant)

        eval_data -- name of file containing the evaluation
                     data (same format as training_data).

        test_data -- name of file containing the data to predict
                     (same format as training_data)

        output_file -- name of file that will contain the computed
                       predictions

        save_file -- name of directory that will contain the
                     Tensorflow Saver data.

        profile_data -- whether or to tensorflow should profile the
                        model data.

        batch_size -- batch size,
        epoch_number -- number of epochs,
        learning_rate -- learning rate used by the NN model
    """

#-------------------------- Loading input data --------------------------------
    print('loading training data ...')
    # load embeddings
    wordembeddings = pr.get_glove_word2vec(GLOVE_EMBEDDINGS_FILE)
    data_embeddings, classes = pr.generate_embeddings(training_data)
    (posembeddings, depembeddings, predembeddings) = data_embeddings

    data = pr.load_data(training_data,
                        classes,
                        wordembeddings,
                        posembeddings,
                        depembeddings,
                        predembeddings)

    #---------------- Creation of tensorflow dataflow graph -----------------------
    print('training data loaded\n\nstarting graph building ...')
    embedding_data = (wordembeddings, posembeddings, depembeddings, predembeddings)
    model = model_constructor(embedding_data,
                              classes,
                              savefile=save_file,
                              profiledata=profile_data,
                              batch_size=batch_size,
                              learning_rate=learning_rate,
                              use_GPU=use_gpu)

    #---------------------------- Training phase ----------------------------------
    if epoch_number != 0:
        print('graph built\n\nstarting training phase ...')
        model.train_model(data, epoch_number)
        print("training phase completed")

    #---------------------------- Evaluation phase --------------------------------
    print("starting evaluation phase ...")
    data = pr.load_data(eval_data,
                        classes,
                        wordembeddings,
                        posembeddings,
                        depembeddings,
                        predembeddings)

    #model.show_logits(data,num=1)

    precision, recall, fmeasure = model.evaluate_model(data)
    print('precision:  ', precision)
    print('recall:     ', recall)
    print('f1 measure: ', fmeasure)

    print('evaluation phase finished\nstarting prediction phase ...')
    pr.create_prediction_file(
        input_file=test_data,
        output_file=output_file,
        model=model,
        batch_size=batch_size,
        classes=classes,
        embedding_data=embedding_data)
'''

if __name__ == "__main__":
    import argparse
    import os.path
    
    import srl_models

    
    # input arguments parsing -------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Semantic Role Labeling (SRL) classifier training, evaluation and prediction.')
    parser.add_argument(
        '-g','--gated',
        action='store_true',
        help='flag indicating if the gated SRL classifier variant should be used')

    parser.add_argument(
        '-s','--simple',
        action='store_true',
        help='flag indicating if the simpler SRL calssifier should be used')
        
    parser.add_argument(
        '-e','--eval',
        action='store_true',
        help='flag indicating if evaluation of the SRL classifier should be performed')
    
    parser.add_argument(
        '--train',
        metavar='EPOCH',
        type=int,
        default=0,
        dest='epoch_number',
        help='number of epoch to use while training, defaults to 0 (i.e. no training)')

    parser.add_argument(
        '--param',
        metavar='STATE',
        default='params',
        type=str,
        dest='save_file',
        help='parmeter values directory (if not present it is stored with name "params").')

    parser.add_argument(
        '--predict',
        type=str,
        metavar='DATA',
        default=None,
        dest='target_data',
        help='data to predict')
    args = parser.parse_args()

    #--------------------------------------------------------------------------
    if args.gated:
        model_constructor = srl_models.SequentialModel
        #model_constructor = srl_models.ParallelGatedModel
    elif args.simple:
        model_constructor = srl_models.SimpleModel
    else:
        model_constructor = srl_models.OriginalModel

        
    args.save_file = os.path.join(args.save_file,'model.ckpt')

    # TODO add arguments in parser
    training_data = '../data/EN/train.txt'
    eval_data = '../data/EN/dev.txt'
    profile_data=False
    batch_size=30
    learning_rate=0.001

    print('loading training data ...')
    # load embeddings
    wordembeddings = pr.get_glove_word2vec(GLOVE_EMBEDDINGS_FILE)
    data_embeddings, classes = pr.generate_embeddings(training_data)
    (posembeddings, depembeddings, predembeddings) = data_embeddings

    data = pr.load_data(training_data,
                        classes,
                        wordembeddings,
                        posembeddings,
                        depembeddings,
                        predembeddings)

    #---------------- Creation of tensorflow dataflow graph -----------------------
    print('training data loaded\n\nstarting graph building ...')
    embedding_data = (wordembeddings, posembeddings, depembeddings, predembeddings)
    model = model_constructor(embedding_data,
                              classes,
                              savefile=args.save_file,
                              profiledata=profile_data,
                              batch_size=batch_size,
                              learning_rate=learning_rate)
    print('graph built\n')

    #---------------------------- Training phase ----------------------------------
    if args.epoch_number != 0:
        print('starting training phase ...')
        model.train_model(data, args.epoch_number)
        print("training phase completed")

    #---------------------------- Evaluation phase ----------------------------
    if args.eval:
        print("starting evaluation phase ...")
        data = pr.load_data(eval_data,
                        classes,
                        wordembeddings,
                        posembeddings,
                        depembeddings,
                        predembeddings)

        #model.show_logits(data,num=1)
        precision, recall, fmeasure = model.evaluate_model(data)
        print('precision:  ', precision)
        print('recall:     ', recall)
        print('f1 measure: ', fmeasure)
        print('evaluation phase finished\n')
    
    if args.target_data != None:
        print('starting prediction phase ...')
        pr.create_prediction_file(
            input_file=args.target_data,
            output_file=args.target_data+'.out',
            model=model,
            batch_size=batch_size,
            classes=classes,
            embedding_data=embedding_data)
