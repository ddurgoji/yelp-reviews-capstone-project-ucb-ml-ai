# Project Title
Capstone Project - Yelp Reviews Sentiment analysis.

## Description
This repo contains the CRISP-DM framework applied on Yelp Review sentiment dataset from [Kaggle](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset/data).

* [Jupyter Notebook](https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/yelp-reviews-capstone-ucb-mlai.ipynb)
    * Jupyter Notebook containing code used to perform this analysis and model development.
* [README](https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/README.md)
    * Project description
* [Dataset](https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/tree/main/dataset)
    * This dataset is from [Kaggle](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset/data).

## Summary
#### Business Understanding
Yelp Reviews Dataset contains reviews from Yelp. It is extracted from the Yelp dataset challenge 2015 data. For more information, please refer to http://www.yelp.com/dataset_challenge.

This Yelp reviews dataset is constructued by Xiang Zhang(xiang.zhang@nyu.edu) from the above dataset.  It is first used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

This Yelp reviews dataset is constructed by considering starts 1 and 2 as negative, 3 and 4 as positive. For each polarity 280k training samples and 19,000 testing samples are taken randomly. In total there are 560k training and 38k testing samples. Negative sentiment reviews are marked as 1 and Positive sentiment reviews are marked as 2.

In this project, i am trying to build various classifier models which helps banks to determine Yelp review comments as positive or negative.
This model can be used in real-time to build various solution as listed below.
* Yelp could downgrade ranking of a restaurant if negative ratings increase
* Yelp could show a restaurant on top if their positive rating is high
* Use the sentiment to provide data to restaurant to improve on their service etc.

#### Data Understanding
* Read CSV Dataset using Pandas. DataFrame contains Review and Rating columns.
* Train DataFrame has 560,000 records and Test DataFrame has 38,000 records.
* Training dataset has both positive and negative classes balanced.
* Checked dataset for below items in Reviews.
  * Special characters - 410217 out of 560000
  * Short hand words - 394763 out of 560000
  * Stop words - 556099 out of 560000
  * HTTP(S) links - 2501 out of 560000

#### Data Preparation
* Performed below actions
  * Converted all letters to lower case.
  * Removed all special characters
  * Removed Stop words
  * Removed HTTP(S) links
  * Removed Accents
  * Normalized spaces
  * Removed short hands
* Checked Sentiment among Positive and Negative classes.
  * Positive rating has majority of reviews with positive sentiment polarity
  * Negative rating has majority of reviews with positive sentiment polarity (Could this cause model to get confused?)
* Generated word cloud to get insight into most occuring words in both positive and negative rating
| <img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/positive-sentiment.png" width="400" title="Positive Reviews Word Cloud" /> | <img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/negative-sentiment.png" width="400" title="Negative Reviews Word Cloud" /> |

#### Modelling
Using train_test_split seperated Training dataset = 80% and Validation Dataset = 20%.

Generated models with below differnet techniques.
* MultinomialNB
* LogisticRegression
* DecisionTreeClassifier
* RandomForestClassifier
* AdaBoostClassifier

LogisticRegression performed better with 93% accuracy on Validation dataset.
Below DataFrame shows the results for all models.
<img src="https://github.com/ddurgoji/yelp-reviews-capstone-project-ucb-ml-ai/blob/main/images/model_eval.png" width="1200" height="600" />

#### Pending items for Capstone part 2 Module 24
* Find performance on test_df.
* Clean up the notebook and move around code blocks appropriately.
* Generate RandomForest and AdaBoostClassifiers with more n_iter's if possible.
* Build a performance graph for various model thats generated.
* Deploy the model on DigitalOcean or any other cloud provider.


#### Evaluation
Will be added as part of Module 24 Capstone part 2

#### Deployment
Will be added as part of Module 24 Capstone part 2

#### Next Steps
Will be added as part of Module 24 Capstone part 2

#### Conclusion
Will be added as part of Module 24 Capstone part 2


## Technologies Used
Below are some of important technologies used in this project.
* [Python](https://www.python.org)
* [Jupyter Lab Notebook](https://jupyter.org)
* [Plotly](https://plotly.com)
* [Seaborn](http://seaborn.pydata.org)
* [Pandas](http://pandas.pydata.org)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [NLTK](https://www.nltk.org)
and some more.


## Author
Dhiraj Durgoji (djdhiraj.8189@gmail.com)
