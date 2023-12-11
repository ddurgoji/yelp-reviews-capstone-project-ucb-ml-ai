### Project Title
CapStone Project: Yelp Reviews Sentiment analysis by **Dhiraj Durgoji (djdhiraj.8189@gmail.com)**

#### Executive summary
Yelp Reviews Dataset contains reviews from Yelp. It is extracted from the Yelp dataset challenge 2015 data. For more information, please refer to http://www.yelp.com/dataset_challenge.

This Yelp reviews dataset is constructued by Xiang Zhang(xiang.zhang@nyu.edu) from the above dataset.  It is first used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

This Yelp reviews dataset is constructed by considering starts 1 and 2 as negative, 3 and 4 as positive. For each polarity 280k training samples and 19,000 testing samples are taken randomly. In total there are 560k training and 38k testing samples. Negative sentiment reviews are marked as 1 and Positive sentiment reviews are marked as 2.

#### Rationale
Yelp has been one of most popular sites for users to rate and review local businesses. Business organize their own listings and users rate the business from 1-5 stars and write text reviews. Users can also vote on helpful reviews written by other users.

In this project, I am trying to build various classifier models which helps businesses to determine Yelp review comments as positive or negative.
This model can be used in real-time to build various solution as listed below.
* Yelp could downgrade ranking of a restaurant if negative ratings increase and not show at the top. This improves the Yelp recommendations quality.
* Yelp could show a restaurant on top if their positive rating is high. Again this improves the Yelp recommendations quality.
* Use the sentiment to provide data to restaurant to improve on their service etc. This adds another revenue stream for Yelp from Businesses who are looking to improve their quality.

This helps Yelp to improve user experience by showing right results based on search and also helps Yelp to upsell value add services to restaurants etc.

#### Research Question
Yelp reviews sentiment analysis to upsell value addition products to both end users and restaurants, shops etc.

#### Data Sources
This dataset is from [Kaggle](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset/data)

#### Methodology
Below is the methodology used.

##### Data Understanding
* Read CSV Dataset using Pandas. DataFrame contains Review and Rating columns.
* Train DataFrame has 560,000 records and Test DataFrame has 38,000 records.
* Training dataset has both positive and negative classes balanced.
* Checked dataset for below items in Reviews.
  * Special characters - 410217 out of 560000
  * Short hand words - 394763 out of 560000
  * Stop words - 556099 out of 560000
  * HTTP(S) links - 2501 out of 560000

##### Data Preparation
* Performed below actions
  * Converted all letters to lower case.
  * Removed all special characters
  * Removed Stop words using NLTK
  * Removed HTTP(S) links
  * Removed Accents
  * Normalized spaces
  * Removed short hands
* Checked Sentiment among Positive and Negative classes.
  * Positive rating has majority of reviews with positive sentiment polarity
  * Negative rating has majority of reviews with positive sentiment polarity (Could this cause model to get confused and underfit?)

<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/sentiment.png" width="800" />

* Generated word cloud to get insight into most occuring words in both positive and negative rating </br>

**Positive Reviews Word Cloud**
</br>
<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/positive-sentiment.png" width="400" />

**Negative Reviews Word Cloud**
</br>
<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/negative-sentiment.png" width="400" title="Negative Reviews Word Cloud" />

#### Modelling
Using train_test_split seperated Training dataset = 80% and Validation Dataset = 20%.

Generated models with below differnet techniques.
* MultinomialNB
* LogisticRegression
* DecisionTreeClassifier
* RandomForestClassifier
* AdaBoostClassifier
* Convolutional Neural Networks(CNN) using Tensorflow/keras
* Recurrent Neural Networks(RNN) using Tensorflow/keras - LSTM & GRU

#### Results

* Deep learning CNN model performed better with 96.88% accuracy.
* LogisticRegression with Count Vectorizer came in second with 93.25% accuracy. 

Below is confusion matrix of Deep learning CNN high performance model.
<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/cnn_cm.png" width="500" height="500" />

Below is confusion matrix of Logistic Regression with Count Vectorizer high performance model.
<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/log_cm.png" width="500" height="500" />

Deep learning RNN LSTM model, Logistic Regression with TFIDF performed well as well.

Below are graphs comparing Accuracy, Precision and Recall for all 13 models.
<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/final_accuracy.png" width="500" height="500" />
<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/final_precision.png" width="500" height="500" />
<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/final_recall.png" width="500" height="500" />


#### Deployment
Used the real-time review from Yelp and used all 13 models to see the results they produce, High performing models mentioned above performed very well by classifying 100% correctly.
Some models with < 90% accuracy classified few of them incorrectly.

Check notebook for detailed results.

#### Conclusion
* Deep learning CNN model performed better than all other models with higher accuracy, precision and recall.
* Logistic regression with Count vectorizer and Deep learning RNN LSTM model performed equally well taking second place.

Give we have higher precision, recall and accuracy from above 3 models they can be used to build a recommendation system for Yelp, value add products to improve businesses etc.

#### Next steps

It was a great learning experiences and very good structured modules. This has helped me personally to stay on course and complete things on time.

I havee gained immense knowledge on ML/AI and looking forward to expand that knowledge further and implements great solution in my current work.

I would like to thank all the learning facilitators and Berkeley professors for all video lectures, Quiz, Codio activites, Practical application projects and this Capstone projects. Thanks for all the feedback on work i did as part of this project.

#### Outline of project
This repo contains the CRISP-DM framework applied on Yelp Review sentiment dataset from [Kaggle](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset/data).

* [Jupyter Notebook](https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/yelp-reviews-capstone-ucb-mlai.ipynb)
    * Jupyter Notebook containing code used to perform this analysis and model development.
* [README](https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/README.md)
    * Project description
* [Dataset](https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/tree/main/dataset)
    * This dataset is from [Kaggle](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset/data).

## Technologies Used
Below are some of important technologies used in this project.
* [Python](https://www.python.org)
* [Jupyter Lab Notebook](https://jupyter.org)
* [Plotly](https://plotly.com)
* [Seaborn](http://seaborn.pydata.org)
* [Pandas](http://pandas.pydata.org)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [TensorFlow](https://www.tensorflow.org)
* [NLTK](https://www.nltk.org)
* [WordCloud](https://pypi.org/project/wordcloud/)
and some more.

## Author
Dhiraj Durgoji (djdhiraj.8189@gmail.com)
