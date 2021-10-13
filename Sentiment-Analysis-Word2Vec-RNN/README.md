<h1><b>Description</b></h1>

Sentiment analysis is extensively used to study customer behaviors using reviews and survey responses, online and social media, and healthcare materials for marketing and costumer service applications. In this assignment we will be going over some important concepts of Natural language processing like Word embeddings and comparing them with models generated in previous assignment usinf TF-IDF. Furthermore, we will be doing a comparison of Simple ML models with Neural Networks. 

While sentiments are not just binary, we will also be learning to deal with ternary classes.

<b>Task 1 - Data Preparation</b>

We will use the Amazon reviews dataset which contains real reviews for kitchen products sold on Amazon. The dataset is downloadable at:
https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz

Read the data as a Pandas frame using Pandas package and only keep the Reviews and Ratings fields in the input data frame to generate data. Our goal is to train sentiment analysis classifiers that can predict the sentiment for a given review. Load the dataset and build a balanced dataset of 250K reviews along with their ratings (50K instances per each rating score) through random selection. Create ternary and binary labels using the ratings. We assume that ratings more than 3 denote positive sentiment (class 1) and rating less than 3 denote negative sentiment (class 2). Reviews with rating 3 are considered to have neutral sentiment (class 3).

<b>Task 2 - Word Embeddings</b>
In this part the of the assignment, you will learn how to generate two sets of Word2Vec features for the dataset you generated. You can use Gensim library for this purpose. A helpful tutorial is available in the following link: <a> https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec. </a>
html

We will be loading the pretrained \word2vec-google-news-300" Word2Vec model and learning how to extract word embeddings for your dataset.
Then we will train a Word2Vec model using our own dataset. Set the embedding size to be 300 and the window size to be 11. You can also consider a minimum word count of 10.

<b>Task 3 - Data Preprocessing</b>

Implement the following steps to preprocess the dataset you created:
- convert the all reviews into the lower case.
- remove the HTML and URLs from the reviews
- remove non-alphabetical characters
- remove extra spaces
- perform contractions on the reviews, e.g., won't ! will not. Include as many contractions in English that you can think of.

Use NLTK package to process your dataset:
- remove the stop words
- perform lemmatization


<b>Task 4 - Simple Models</b>
Using the Word2Vec features that you can generate using the two models you prepared in the Word Embedding section, train a perceptron and an SVM model similar to HW1 for class 1 and class 2 (binary models). For this purpose, you can just use the average Word2Vec vectors for each review as the input feature.

<b>Task 5 - Feedforward Neural Networks</b>
Using the features that you can generate using the models you prepared in the Word \Embedding section", train a feedforward multilayer perceptron network for sentiment analysis classification. Consider a network with two hidden layers, each with 50 and 10 nodes, respectively. You can use cross entropy loss and your own choice for other hyperparamters, e.g., nonlinearity, number of epochs, etc. Part of getting good results is to select good values for these hyperparamters.

You can also refer to the following tutorial to familiarize yourself: <a>  https://www.kaggle.com/mishra1993/pytorch-multi-layer-perceptron-mnist </a>
Although the above tutorial is for image data but the concept of training an MLP is very similar to what we want to do.


To generate the input features, use the average Word2Vec vectors similar to the Simple models" section and train the neural network. Train a network for binary classification using class 1 and class 2 and also a ternary model for the three classes.

To generate the input features, concatenate the first 10 Word2Vec vectors for each review as the input feature and train the neural network.

<b>Task 6 - Recurrent Neural Networks</b>
Using the features that you can generate using the models you prepared in the "Word Embedding" section, train a recurrent neural network (RNN) for sentiment analysis classification. You can refer to the following tutorial to familiarize yourself: <a> https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html </a>

Train a simple RNN for sentiment analysis. You can consider an RNN cell with the hidden state size of 50. To feed your data into our RNN, limit the maximum review length to 50 by truncating longer reviews and padding shorter reviews with a null value (0). Train the RNN network for binary classification using class 1 and class 2 and also a ternary model for the three classes.

Repeat the same for gated recurrent unit cell.
