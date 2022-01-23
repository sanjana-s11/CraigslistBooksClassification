#importing libraries
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

#reading csv
books = pd.read_csv('book32-listing.csv',encoding = "ISO-8859-1", header=None)
books_craigs = pd.read_csv('craigs_listings.csv')
books_craigs = books_craigs[['Title','GenreNew']]

#assigning columns and cleaning data
books.columns = ['book_id','book_img','book_link','booktitle', 'author', 'class','genre']
books = books[['booktitle','author','genre']]
books['author'].fillna('', inplace=True)
books.drop(books[books['genre']=='Gay & Lesbian'].index, inplace=True)
books.reset_index(drop=True, inplace=True)

#creating variables to use as input and mergind craigslist data
books['title_author'] = books['booktitle']+' '+books['author']
books = books[['title_author','genre']]
books_craigs.rename(columns={'Title':'title_author','GenreNew':'genre'},inplace=True)
all_books = pd.concat([books,books_craigs],ignore_index=True)

#Bucketing and preserving relevant genres
all_books.genre.replace(to_replace={
       'Literature & Fiction':'Fiction',
       'Arts & Photography':'Arts, Crafts & Photography', 'Reference':'Education & Teaching',
       'Science Fiction & Fantasy':'Fiction',
       'Cookbooks, Food and Wine':'Cookbooks, Food & Wine', 'Law':'Education & Teaching',
       'Health, Fitness & Dieting':'Education & Teaching', 'Politics & Social Sciences':'Education & Teaching',
       'Test Preparation':'Education & Teaching', 'Christian Books & Bibles':'Religion & Spirituality',
       'Crafts, Hobbies & Home':'Arts, Crafts & Photography',
       'Business & Money':'Education & Teaching', 'Science & Math':'Education & Teaching',
       'Comics & Graphic Novels':"Children's Books",
       'Computers & Technology':'Education & Teaching', 'Romance':'Fiction',
       'Medical Books':'Education & Teaching', 'Mystery, Thriller & Suspense':'Fiction',
       'Self-Help':'Education & Teaching', 'Parenting & Relationships':"Children's Books"},inplace=True)

#encoding the genres
genre = all_books['genre'].unique()
le = LabelEncoder()
le.fit(genre)
all_books['genre_encoded'] = le.transform(all_books['genre'])

#removed stop-words
books_raw = all_books['title_author']
books_cleaned = []
for i in range(len(books_raw)):
               book_tokenized=nltk.tokenize.WhitespaceTokenizer().tokenize(books_raw[i])
               book_tokenized = [j.lower() for j in book_tokenized if not j in stopwords.words('english')]
               books_cleaned.append(' '.join(book_tokenized))

#Creating term-document matrix using TF-IDF
tfidf = TfidfVectorizer(min_df=2,analyzer='word',stop_words='english',strip_accents='unicode',sublinear_tf=True,token_pattern=r'\w+')
books_trans = tfidf.fit_transform(books_cleaned)
print(books_trans.shape)

#Creating train, validate and test sets
y = all_books.loc[:204867,'genre_encoded'].to_list()
X = books_trans[:-971]
craigs_y = all_books.loc[204868:,'genre_encoded'].to_list()
craigs_X = books_trans[-971:]
train_books, valid_books, train_genre, valid_genre = train_test_split(X,y,test_size=0.3,random_state=7,train_size=0.7,stratify=y)

#Neural Networks
nn_clf = MLPClassifier(activation='logistic',alpha=0.025,random_state=7,hidden_layer_sizes=(30,),solver='adam')
nn_clf.fit(train_books,train_genre)
train_predicted_nn = nn_clf.predict(train_books)
valid_predicted_nn = nn_clf.predict(valid_books)
craigs_predicted_nn = nn_clf.predict(craigs_X)
train_accuracy_nn = accuracy_score(train_genre,train_predicted_nn)
valid_accuracy_nn = accuracy_score(valid_genre,valid_predicted_nn)
craigs_accuracy_nn = accuracy_score(craigs_y,craigs_predicted_nn)
print('Neural Networks Train Accuracy:', train_accuracy_nn)
print('Neural Networks Valid Accuracy:',valid_accuracy_nn)
print('Neural Networks Craigslist Prediction:',craigs_accuracy_nn)

#Multinomial Naive Bayes
clf_mnb = MultinomialNB(alpha=0.25)
clf_mnb.fit(train_books,train_genre)
train_predicted_mnb = clf_mnb.predict(train_books)
valid_predicted_mnb = clf_mnb.predict(valid_books)
craigs_predicted_mnb = clf_mnb.predict(craigs_X)
train_accuracy_mnb = accuracy_score(train_genre,train_predicted_mnb)
valid_accuracy_mnb = accuracy_score(valid_genre,valid_predicted_mnb)
craigs_accuracy_mnb = accuracy_score(craigs_y,craigs_predicted_mnb)
print('MNB Train Accuracy:', train_accuracy_mnb)
print('MNB Valid Accuracy:',valid_accuracy_mnb)
print('MNB Craigslist Prediction:',craigs_accuracy_mnb)

#Decision Tree
clf_dt = DecisionTreeClassifier(min_samples_leaf = 5, random_state=7)
clf_dt.fit(train_books,train_genre)
train_predicted_dt = clf_dt.predict(train_books)
valid_predicted_dt = clf_dt.predict(valid_books)
craigs_predicted_dt = clf_dt.predict(craigs_X)
train_accuracy_dt = accuracy_score(train_genre,train_predicted_dt)
valid_accuracy_dt = accuracy_score(valid_genre,valid_predicted_dt)
craigs_accuracy_dt = accuracy_score(craigs_y,craigs_predicted_dt)
print('Decision Tree Train Accuracy:', train_accuracy_dt)
print('Decision Tree Valid Accuracy:',valid_accuracy_dt)
print('Decision Tree Craigslist Prediction:',craigs_accuracy_dt)

#Logistic Regression
logit = LogisticRegression(C=0.5)
logit.fit(train_books, train_genre)
train_predicted_logit = logit.predict(train_books)
valid_predicted_logit = logit.predict(valid_books)
craigs_predicted_logit = logit.predict(craigs_X)
train_accuracy_logit = accuracy_score(train_genre,train_predicted_logit)
valid_accuracy_logit = accuracy_score(valid_genre,valid_predicted_logit)
craigs_accuracy_logit = accuracy_score(craigs_y,craigs_predicted_logit)
print('Logit Train Accuracy:', train_accuracy_logit)
print('Logit Valid Accuracy:',valid_accuracy_logit)
print('Logit Craigslist Prediction:',craigs_accuracy_logit)

#random forest
rf = RandomForestClassifier(max_features= 244, max_depth=150 , n_estimators =200 )
rf.fit(train_books, train_genre)
train_predicted_rf = rf.predict(train_books)
valid_predicted_rf = rf.predict(valid_books)
craigs_predicted_rf = rf.predict(craigs_X)
train_accuracy_rf = accuracy_score(train_genre,train_predicted_rf)
valid_accuracy_rf = accuracy_score(valid_genre,valid_predicted_rf)
craigs_accuracy_rf = accuracy_score(craigs_y,craigs_predicted_rf)
print('RF Train Accuracy:', train_accuracy_rf)
print('RF Valid Accuracy:',valid_accuracy_rf)
print('RF Craigslist Prediction:',craigs_accuracy_rf)

#svm
clf_svm = LinearSVC(C=0.05,random_state=45)
clf_svm.fit(train_books,train_genre)
train_predicted_svm = clf_svm.predict(train_books)
valid_predicted_svm = clf_svm.predict(valid_books)
craigs_predicted_svm = clf_svm.predict(craigs_X)
train_accuracy_svm = accuracy_score(train_genre,train_predicted_svm)
valid_accuracy_svm = accuracy_score(valid_genre,valid_predicted_svm)
craigs_accuracy_svm = accuracy_score(craigs_y,craigs_predicted_svm)
print('SVM Train Accuracy:', train_accuracy_svm)
print('SVM Valid Accuracy:',valid_accuracy_svm)
print('SVM Craigslist Prediction:',craigs_accuracy_svm)
print('SVM Craigslist F1 Score:',craigs_accuracy_svm)

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(confusion_matrix(craigs_y,craigs_predicted_svm),xticklabels=le.inverse_transform(range(14)),yticklabels=le.inverse_transform(range(14)),cmap='Blues',annot=True,ax=ax,fmt='g')
plt.show()



