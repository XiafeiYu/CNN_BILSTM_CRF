# CNN_BILSTM_CRF - tensorflow

the program implement this paper: https://arxiv.org/pdf/1603.01354.pdf <br> 

the pretraining word embedding can download at https://nlp.stanford.edu/projects/glove/, the pretrainng embedding matrix is 100 dimension vectors in glove.6B.zip <br>

the dataset is CoNLL2003 and the dataset path in this project is ./CoNLL2003

## Prerequisites
python 3.5 <br> 
tensorflow 0.12.1 <br> 
gpu or cpu <br> 

## Usage
To train a model <br> 
```
$ python main.py --train True <br> 
```
To test a model <br> 
```
$ python main.py --train False <br> 
```
