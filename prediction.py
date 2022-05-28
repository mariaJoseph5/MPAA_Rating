import os
import csv
import time
import joblib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings(action = 'ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('Data_Set.csv')
df = df[pd.notnull(df['Sub Category'])]
col = ['Category', 'Sub Category']
df = df[col]
df.columns = ['Category', 'Sub_Category']
df['Category_Id'] = df['Category'].factorize()[0]
from io import StringIO
category_id_df = df[['Category', 'Category_Id']].drop_duplicates().sort_values('Category_Id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Category_Id', 'Category']].values)


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Sub_Category).toarray()
labels = df.Category_Id

from sklearn.feature_selection import chi2
import numpy as np

N = 2
for Category, Category_Id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == Category_Id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]



X_train, X_test, y_train, y_test = train_test_split(df['Sub_Category'], df['Category'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
joblib.dump(clf,'Best_Model.pkl')
model = joblib.load('Best_Model.pkl')

def matching():
    with open('Data_Set.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            match = row[1]
            if match == 'I am stuck in traffic in a taxicab which is typical and not just of modern life':
                file  = (repr(match))
                print(file)
                file = 'file alredy exits'
                print(file)
            else:
                print('not match')

def find_catg(Trust_fet):
    new_pred=''
    print('Trust_fet',Trust_fet)
    new_prob = model.predict_proba(count_vect.transform([Trust_fet]))
    print(Trust_fet,'new_prob=',new_prob)
    test_confidance = 100*np.max(new_prob)
    acc = "{:.0f}".format(test_confidance)
    accuracy = "{:2f}%".format(test_confidance)
    print("Test_Accuracy:",accuracy)
    f = open("Test_Accuracy.txt","w")
    f.write(str(accuracy))
    f.close()
    print(acc,accuracy)
    if int(acc) >= 70:
        result = "Matching the category"
        print(result)
        f = open("Status.txt","w")
        f.write(str('Matched'))
        f.close()

        new_pred = model.predict(count_vect.transform([Trust_fet]))
        
        print('\n')
        new_pred=new_pred[0]
        print('Predict:>>',new_pred)
        
    else:
        print("Not Matching the Category")
        
        new_pred='Not Matching the Category'
    return str(new_pred)