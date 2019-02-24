# Semantic-Role-Labeling
An **Semantic Role Label** classifier inspired by the article *"Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling"* by Marcheggiani and Titov.

## Dependencies
**Tensorflow** (either for cpu or gpu, version >= 1.9 and < 2.0) is required in order to run the system.

The other dependencies can be found in `requirements.txt` and installed by running the command:
`pip install -r requirements.txt`


## Usage
The system can be used to train a model, evaluate it, or predict the semantic labels for some unseen data.
To do so, the module `run.py` should be invoked, using the necessary input arguments;
A brief explenation of such arguments can be obtained by running:

   `python run.py -h`

### Training
In order to train the system on the Semantic Role Labeling task, run the command:

  `python run.py --train <epochs> --params <param_folder>`

The argument `<epochs>` is the number of epochs that will be used during training.
`<param_folder>` is the folder that will contain the trained parameters (weights) used by the classifier.

### Evaluation
It is possible to assess the performance of a trained classifier by invoking

   `python run.py --eval --params <param_folder>`

The argument `<param_folder>` should contain the trained parameters (weights) used by the SRL classifier. A good classifier should have **Precision**, **Recall** and **F1**around 

|**Precision**|**Recall**|**F1**|
|-------------|----------|------|
|    0.85     |    0.82  | 0.83 |

### Prediction
By invoking the command

   `python run.py --predict <data-file> --params <param_folder>`,

it is possible to predict the classifier output with respect to the data stored in 
`<data-file>` (file that must follow the **CoNLL 2009** data format). The predicted labels will be stored in the file `<data-file>.out`.