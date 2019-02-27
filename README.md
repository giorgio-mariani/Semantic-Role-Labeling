# Semantic-Role-Labeling
*Semantic Role Labeling* is a **Natural Language Processing** problem that consists in the assignment of semantic roles to words in a sentence. Specifically, given the main predicate of a sentence, the task requires the identification (and correct labeling) of the predicate's semantic arguments. A simple example is the sentence *"the cat eats a fish"*, with *cat* and *fish* rispectively the **agent** and the **patient** of the main predicate *eats*.

## Project Description
The project consists in the implementation of a **Semantic Role Label** classifier inspired by the article *"Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling"* by Marcheggiani and Titov. An in detail [report](docs/report.pdf) about the project and the assignment's [specification](docs/assignment.pdf) can be found in the `docs` folder.


## Dependencies
**Tensorflow** (either for cpu or gpu, version >= 1.9 and < 2.0) is required in order to run the system.
The other software dependencies can be found in `requirements.txt` and installed by running the command:

`pip install -r requirements.txt`

## Usage
The system can be used to train a model, evaluate it, or predict the semantic labels for some unseen data.
To do so, the module `run.py` should be invoked, using the necessary input arguments;
A brief explenation of the software's options can be obtained by running

   `python run.py -h`

***IMPORTANT:*** In order to work properly, the system requires the download of this [data](https://drive.google.com/open?id=1gBtnChRt5BXbq5mfCHpk-J8KCl_XWJ7Q). After downloading the content, place it into the `data` directory.

### Training
In order to train the system on the Semantic Role Labeling task, run the command:

  `python run.py --train <epochs> --params <param_folder>`

The argument `<epochs>` is the number of epochs that will be used during training.
`<param_folder>` is the folder that will contain the trained parameters (weights) used by the classifier.

### Pre-Trained Models
Pre-trained models are available in this [link](https://drive.google.com/open?id=14K_U-xQMzMlqpr4jMpuH6zw6UinIsbkz). After download, place these models in the `models` directory. You can then use these through the commands

  `python run.py --params ../models/original <...> `

or

   `python run.py --gated --params ../models/gated <...> `,
  
depending on the model you are using.

### Evaluation
It is possible to assess the performance of a trained classifier by invoking

   `python run.py --eval --params <param_folder>`

The argument `<param_folder>` should contain the trained parameters (weights) used by the SRL classifier. A good classifier should have **Precision**, **Recall** and **F1** around 

|**Precision**|**Recall**|**F1**|
|-------------|----------|------|
|    0.83     |    0.81  | 0.82 |

### Prediction
By invoking the command

   `python run.py --predict <data-file> --params <param_folder>`,

it is possible to predict the classifier output with respect to the data stored in 
`<data-file>` (file that must follow the **CoNLL 2009** data format). The predicted labels will be stored in the file `<data-file>.out`.

