# pitchfork-score-predictor

##INTRODUCTION

Pitchfork is a popular music publication that has been the target of increased controversy over the years from internet-based music communities. One of the most famous Pitchfork series is its album review series. Along with a detailed review of one editor’s opinions about the album’s sonic and lyrical value, each of these reviews is published along with a score from 1-10. The issue of representing musical quality with a numeric value is undoubtedly a controversial one, but it does provide a very effective way for the editors to recommend music that they enjoyed and identify music that they didn’t. 

We acknowledge the usefulness of this numerical reduction, but were also curious about the complexity and consistency of this process. It is typically clear from an extensive review on Pitchfork’s site which aspects of the music they enjoyed and which they didn’t, but we wanted to go a bit further and see if any of these factors had an especially large part to play on the eventual score the album received. Are some scores biased towards more popular albums? Are most scores even consistent with the sentiments expressed by the reviewer in their long review? These were burning questions we felt would best be approached through a big data approach informed by machine learning. We sought to find which statistics of an album were most commonly correlated with higher scores, and hoped to be able to build an algorithm to predict those scores.

##DATA COLLECTION
	
	Our data collection process took place in 2 stages, first the collection of data from Pitchfork reviews, and secondly, a collection of data from Spotify’s API.

We were able to find a dataset of 24,169 Pitchfork reviews from Jan 5, 1999 to Dec 12, 2021 on Kaggle: https://www.kaggle.com/datasets/nolanbconaway/24169-pitchfork-reviews
I first converted them from SQLite tables to many different dataframes and merge all of them together based on their common column. Then, I realized that if I just merely add all the datasets together, some album would have duplicated entries just because they have multiple artists or agencies. Therefore, I grouped all the entries by their review_url and dropped some of the columns. Here is my reasoning:
artist_id and name are joined together because when multiple artists are working on the same album, the review is duplicated and each artist is listed separately.
Only the first author is chosen when grouping the data because I have manually checked that there is no album with two authors listed as working on it.
artist_url, is_standard_review, review_tombston_id, picker_index, best_new_music, and best_new_reissue are all parameters that we will not use in our analysis so I decided to leave them all out
title, score, release_year, pub_date, and body are the same for the same review about an album so I just chose the first occurrence.
label is joined together because an album can be published by multiple labels
Finally, in the end, I know that we do not need `artist_id` so I removed that column.
Then I noticed that there were duplicates in the genre and label column like an entry would be [‘rap’, ‘rap’, ‘rap’], so I removed all the duplicates by turning these entries into a list only kept the unique values. 

In addition to information about the Pitchfork reviews, however, we wanted to get some extra information about the albums themselves, so we turned to Spotify’s Spotipy API package. 
We built a script that used Spotipy’s search feature to query information about an album by searching for that album title, and then downloading things like its popularity score and its genres. Most of the time this information was absent for the album itself, so we instead had to turn to collecting the popularity and genre data from the album artist instead. There was some additional cleaning that needed to go into the artists’ names in order to search for them effectively in the API (like removing listed collaborators and grappling with different name formats) but we were eventually able to get everything we needed. 
Another important aspect of this stage of the process was ensuring that we would be able to submit enough API calls to handle the 20,000+ albums in the dataset. Using a Spotify developer account, we were able to make quite a few queries at once, but it did block our abilities at a certain point. To combat this, we implemented a sleeping function after each batch of albums to slow the request rate. Additionally, we created a cache of artist information that had already been pulled from the API. This way, artists that appeared in the dataset multiple times would only require one API call in total, the rest of the information could just be pulled from the cache instead. 

In the end, our dataset had x features, including album artist (name: string), title (title: string), genre (genre: list of strings), review body (body: string), release year (release_year: int), popularity (popularity: int), and score (score: float). After cleaning and removal of duplicates, we ended up with a complete set of data for 21,399 albums. 
Word Cloud: I wanted to look at the word cloud to see if there is any word that jumps out to me. However, the most commonly used words are ones that you would find in a song review regardless so I don’t think that the word cloud provides much insight into the data.


##METHOD:

###Correlation Analysis
> Release Year: Most of the albums are released in the 2000s but there are still a few reviews on albums released prior to this date

> Pitchfork Score Distribution: The distribution is relatively normal, but the mean is pretty high, indicating that Pitchfork generally give pretty high ratings.

> Spotify Popularity Distribution: The Spotify popularity score distribution is more scattered than the Pitchfork score distribution. This means that there are albums on this list that are not very popular, which achieve our goal of having a wide variety of albums in our dataset.

> Pitchfork Score vs Spotify Popularity Correlation: The correlation between the Pitchfork Score and Spotify Popularity is 0.015797, meaning that there is a very weak correlation between the two. This answers our question that Pitchfork Score is not a good predictor for Spotify Popularity.

> Release Year & Rating: We observed that there was a slight negative correlation between release year & Pitchfork rating, indicating that for every year further in the past, the Pitchfork rating of an album would be slightly higher on average. Though this effect was very slight (Correlation = -0.103255), it was found to be statistically significant (pvalue=8.405e-52), and is generally consistent with some further results we noticed.


### Sentiment Analysis The distribution for the sentiment analysis score of all the reviews is normal and there is no significant trend in any of the categories. 

### Sentiment Analysis Regressor: We used nltk.sentiment.vader’s SentimentIntensityAnalyzer to train a linear regressor on the sentiment of Pitchfork reviews to the received scores. We didn’t receive especially accurate results with this approach, but the real scores were balanced somewhat evenly in quantity above & below predictions (especially in regions of neutral sentiment).

## Machine Learning
### Bag of Words Analysis
> To make this bag-of-word, I decided to cut remove any word that have less than 20 entries as the list of unique words has become quite long. In addition, when I split the dataset into train and test as well as when I was running the different models, the dataset was too big. Therefore, of the whole bag-of-word matrix, I only used 10% of it for the train dataset and 1% of it for the test dataset.
###Decision Tree
> I started with creating a basic tree without any hyperparameter specified to determine the number of leaves and depth of the tree. 
Using this information, I tuned for the max_depth, max_leaf_nodes, max_features, and ccp_alphas. I used randomized search rather than grid search cv because the latter is too computationally taxing.
> My best set of hyperparameter is {'max_leaf_nodes': 1450, 'max_features': 81, 'max_depth': 92, 'ccp_alpha': 0.01297629959041774}
This model gets me an RMSE of 1.3334666884847093
### Bagging Model
> I decided to bag a few trees together to see if I would get a better model. 
> After some visualization, I realized that having around 150 trees for the bagging model is good enough.
> My set of hyperparameters is {base_estimator=DecisionTreeRegressor(random_state = 1), n_estimators= 150}
> This model gets me an RMSE of 1.1975704654549604
### Random Forest
> After some visualization, I realized that having around 150 trees for the bagging model is good enough
> Using this information, I tuned for the max_depth, max_leaf_nodes, max_features, and n_estimators. I used randomized search rather than grid search cv because the latter is too computationally taxing. 
> My set of hyperparameters is {'n_estimators': 100, 'max_leaf_nodes': 1400, 'max_features': 181, 'max_depth': 92}
> This model gets me an RMSE of 1.2574338722052685
> Adaptive Boosting
> I tuned for the learning_rate, base_estimator__max_depth, and n_estimators. I used randomized search rather than grid search cv because the latter is too computationally taxing. 
> My set of hyperparameters is {'n_estimators': 200, 'learning_rate': 0.1, 'base_estimator__max_depth': 10}
> This model gets me an RMSE of 1.24805533513015
### Voting Ensemble
> I only used Random Forest and Bagging Models because Adaptive Boosting Model is too computationally taxing and the Decision Tree Model has a much higher RMSE. 
This model gets me an RMSE of 1.2214853954786193, which is not that good.
Stacking ensemble
> I decided to use Random Forest, Bagging, and Decision Tree Models for this ensemble model because Adaptive Boosting Model is too computationally taxing. Stacking ensemble model can weigh in on their RMSE of the decision tree model so that’s why I included it. Finally, I use the linear model as the final estimator as I have yet to build a model on Linear Regression. 
> This model gets me an RMSE of 1.1399133673393813, which is the best RMSE that we have gotten so far.

## Neural Network: Prediction on Popularity & Release Year
> We used TensorFlow’s Keras packages to train on all 24k album release years & popularity scores.
> We trained for 100 epochs, and adjusted layer complexity through trial and error until getting a relatively small MAE (mean absolute error) and decent loss curve. Dropout layers were added to avoid overfitting, our loss was measured using mean squared error, and prediction accuracy was measured using mean absolute error.
This network again did not give great accuracy in its predictions, but it did capture the general trend of reviews slightly decreasing over time then stagnating somewhat. Incorporating popularity into this model apparently didn’t seem to have as much of a significant effect.



#CONCLUSION:

We learned a lot about data collection, data analysis, machine learning, and language processing through this rather complicated problem. We faced many obstacles, and had to course correct a few times, but we are proud of the work we’ve done. Ultimately, we were not able to build a predictive model that was impressively accurate, but there are a few improvements we could make in the future if we had more time:
> Spotify offers thousands of genre options, and multiple can be assigned to a single album. Though it would be impractical to train a neural network on each of these genres being present as a unique input to the network, there is a way we could incorporate genre into our prediction network. We would have to develop a sort of mapping from all or most of the genres present in the dataset to a smaller, more manageable set of more general genres to use as inputs.
> For more advanced NLP of the reviews it would be useful to use a large pre-trained model that has been trained on data orders of magnitudes larger than our review dataset. If we were to use one of these, it would be possible to implement transfer learning to more closely capture more nuanced tones in reviews. From here we could have more accurate predictions related to the presence of certain types of tone, or even build a generative model that would create Pitchfork-esque reviews for a given album.
> Once a satisfactory prediction network is set up, it would be convenient for users to be able to upload an un-reviewed album to a frontend that is hooked up to our code. This front end would take the title / link of an album, query the Spotify API for statistics, and then use our trained neural network to output a predicted score (or maybe even generated review).
