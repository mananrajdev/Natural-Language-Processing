<h1><b>Description</b></h1>

This assignment gives me hands-on experience with text representations and the use of text classication for sentiment analysis. 
Sentiment analysis is extensively used to study customer behaviors using reviews and survey responses, online and social media, and healthcare materials for marketing and costumer service applications.

<b>Task 1 - Data Preparation</b>

We will use the Amazon reviews dataset which contains real reviews for kitchen products sold on Amazon. The dataset is downloadable at:
https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz

Read the data as a Pandas frame using Pandas package and only keep the Reviews and Ratings fields in the input data frame to generate data. Our goal is to train sentiment analysis classifiers that can predict the sentiment (positive/negative) for a given review. We create binary labels using the ratings. We assume that ratings more than 3 demonstrate positive sentiment (mapped to 1) and rating less than or equal 2 demonstrate negative sentiment (mapped to 0). Discard reviews with the rating 3 as neutral reviews. 

The original dataset is large. To avoid computational burden, I selected 100,000 reviews with positive sentiment along with 100,000 reviews withmnegative sentiment to preform the required tasks on the downsized dataset.

<b>Task 2 - Data Cleaning</b>

Implement the following steps to preprocess the dataset you created:
- convert the all reviews into the lower case.
- remove the HTML and URLs from the reviews
- remove non-alphabetical characters
- remove extra spaces
- perform contractions on the reviews, e.g., won't ! will not. Include as many contractions in English that you can think of.

<b>Task 3 - Preprocessing</b>

Use NLTK package to process your dataset:
- remove the stop words
- perform lemmatization

<b>Task 4 - Feature Extraction</b>

Use sklearn to extract TF-IDF features. At this point, you should have created a dataset which consists of features and binary labels for the reviews you selected.

<b>Task 5 - Perceptron</b>

Train a Perceptron model on your training dataset using the sklearn built-in implementation.

<b>Task 6 - SVM</b>

Train an SVM model on your training dataset using the sklearn built-in implementation.

<b>Task 7 - Logistic Regression</b>

Train a Logistic Regression model on your training dataset using the sklearn built-in implementation.
