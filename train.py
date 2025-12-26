from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib

newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

print(f"Число наблюдений в обучающей выборке {newsgroups_train.filenames.shape}")

vectorizer = CountVectorizer()
sparse_train = vectorizer.fit_transform(newsgroups_train.data)

unique, counts = np.unique(newsgroups_train.target, return_counts=True)

class_priors = counts / counts.sum()

mnb = MultinomialNB(alpha=0.10, class_prior=class_priors)
mnb.fit(sparse_train, newsgroups_train.target)

joblib.dump((vectorizer), 'model_vec.joblib')
joblib.dump((mnb), 'model_mnb.joblib')