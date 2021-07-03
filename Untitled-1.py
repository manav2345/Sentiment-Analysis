#%%
from pandas_ods_reader import read_ods
data = read_ods("hi_3500.ods",1, headers=False)
print("Total Rows",len(data.index))
data.head(10)
data[2077:2085]
#%%
import seaborn as sns
sns.countplot(data["column_1"])
#%%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["column_1"]=le.fit_transform(data["column_1"])
data[2077:2085]
#%%
tokenized_tweet = data['column_0'].apply(lambda x: x.split())
tokenized_tweet.head()
#%%
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet
#%%
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(data['tidy_tweet'])
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(data['tidy_tweet'])


# %%
#Building model using Bag-of-Words features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(data['column_1'], data['column_0'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int))# calculating f1 score

# %%
