# Natural Language Processing - Sentiment Analysis
 IMDB dataset having 50,000 movie reviews for natural language processing or Text analytics. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.


<img algin="center" src="https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg"/>

## Sentiment Analysis for reviews using IMDB Dataset with CNN and LSTM
___

## Run

- get data: download [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) and [Glove embeddings](http://nlp.stanford.edu/data/glove.6B.zip).
- Clone and unzip this repo.
- modify variables `dataset_dir ` and `glove_dir`  in `saverData.py` with your absolute path to dataset and embedding directory.
- Run `saverData.py`, this will load your dataset and embedding, saving them in numpy format.
- Run one of the following: `mainBIDIRLSTM.py`, `mainCNNnonstatic.py`, `mainCNNrand.py`, `mainCNNstatic.py`, `mainDOBLE.py`, `mainLSTM.py`

## Getting the Dataset 

The "Large Movie Review Dataset"(*) shall be used for this project. The dataset is compiled from a collection of 50,000 reviews from IMDB on the condition there are no more than 30 reviews per movie. The numbers of positive and negative reviews are equal. Negative reviews have scores less or equal than 4 out of 10 while a positive review have score greater or equal than 7 out of 10. Neutral reviews are not included. The 50,000 reviews are divided evenly into the training and test set. 

The Training Dataset used is stored in the zipped folder: aclImbdb.tar file. This can also be downloaded from: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. 

The Test Dataset is stored in the folder named 'test'


## Data Preprocessing 

The training dataset in aclImdb folder has two sub-directories pos/ for positive texts and neg/ for negative ones. Use only these two directories. The first task is to combine both of them to a single csv file, “imdb_tr.csv”. The csv file has three columns,"row_number" and “text” and “polarity”. The column “text” contains review texts from the aclImdb database and the column “polarity” consists of sentiment labels, 1 for positive and 0 for negative. The file imdb_tr.csv is an output of this preprocessing. In addition, common English stopwords should be removed. An English stopwords reference ('stopwords.en') is given in the code for reference.


## Data Representations Used  

Unigram , Bigram , TfIdf 


## Implementation

This project was implemented using Keras framework with Tensorflow backend.
After loading text data, and embedding file, I created an embedding_matrix with as many entries as unique words in training data (111525 unique tokens), where each row is the equivalent embedding representation. If the word is not present in the embedding file, it's representation would be simply a vector of zeros.

The mean number of word per review is 230 with a variance of 171. Using `MAX_SEQUENCE_LENGTH = 500` you can cover the majority of reviews and remove the outliers with too many words.

Essentially three different architectures were used:

- Only CNN (non-static/static/random)
- Only LSTM (and BiDirectional LSTM)
- Both CNN and LSTM

### Only CNN

#### non-static
Using trainable word embedding.

```Python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix], 
                                               input_length=MAX_SEQUENCE_LENGTH, trainable=True)
x = embedding_layer(sequence_input)
x = Dropout(0.5)(x)
x = Conv1D(200, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.5)(x)
x = Conv1D(200, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(180,activation='sigmoid', kernel_regularizer=regularizers.l2(0.05))(x)
x = Dropout(0.5)(x)
prob = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, prob)

optimizer = optimizers.Adam(lr=0.00035)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy', 'mae'])
```


#### static
Using non trainable word embedding.
```Python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix],
                                               input_length=MAX_SEQUENCE_LENGTH, trainable=False)
x = embedding_layer(sequence_input)
x = Dropout(0.3)(x)
x = Conv1D(25, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = Conv1D(20, 5, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(120,activation='sigmoid')(x)
x = Dropout(0.3)(x)
prob = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, prob)

optimizer = optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy', 'mae'])
```

```
Total params: 34,023,686
Trainable params: 565,886
Non-trainable params: 33,457,800
```
### Only LSTM

```Python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix],
                                               input_length=MAX_SEQUENCE_LENGTH, trainable=False)

x = embedding_layer(sequence_input)
x = Dropout(0.3)(x)
x = LSTM(100)(x)
prob = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, prob)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
```


### Both CNN and LSTM

```Python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix],
                                               input_length=MAX_SEQUENCE_LENGTH, trainable=False)

x = embedding_layer(sequence_input)
x = Dropout(0.3)(x)
x = Conv1D(200, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = LSTM(100)(x)
x = Dropout(0.3)(x)
prob = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, prob)
optimizer = optimizers.Adam(lr=0.0004)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
```

## Results
Learning curves values for accuracy and loss are calculated during training using a validation set (10% of training set). 

### Only CNN 
#### non-static
 [![cnnaccnonstatic](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\cnnaccnonstatic.png)]() | [![cnnlossnonstatic](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\cnnlossnonstatic.png)]() 
|:---:|:---:|
| Accuracy | Loss |

On entire Test Set: `Accuracy = 89.96%`

#### static
 [![cnnaccstatic](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\cnnaccstatic.png)]() | [![cnnlossstatic](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\cnnlossstatic.png)]() 
|:---:|:---:|
| Accuracy | Loss |

On entire Test Set: `Accuracy = 88.98%`

#### random
 [![cnnaccrandom](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\cnnaccrand.png)]() | [![cnnlossrandom](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\cnnlossrand.png)]() 
|:---:|:---:|
| Accuracy | Loss |

On entire Test Set: `Accuracy = 87.72%`

### Only LSTM 
 [![lstmacc](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\lstmacc.png)]() | [![lstmloss](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\lstmloss.png)]() 
|:---:|:---:|
| Accuracy | Loss |

On entire Test Set: `Accuracy = 88.92%`

### Both CNN and LSTM
 [![dobleacc](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\dobleacc.png)]() | [![dobleloss](C:\Users\LENOV\Documents\GitHub\Natural-Language-Processing---Sentiment-Analysis\Images\dobleloss.png)]() 
|:---:|:---:|
| Accuracy | Loss |

On entire Test Set: `Accuracy = 90.14%`




## References
\[1\]: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

\[2\]: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

\[3\]: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

\[4\]: Takeru Miyato, Andrew M. Dai and Ian Goodfellow (2016) -"Virtual Adversarial Training for Semi-Supervised Text Classification"- https://pdfs.semanticscholar.org/a098/6e09559fa6cc173d5c5740aa17030087f0c3.pdf

