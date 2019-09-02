import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

from ast import literal_eval
stop_words = set(stopwords.words('english'))
df = pd.read_csv('train_tweets.csv')
validation =pd.read_csv('test_tweets_unlabeled.csv')

print("数据总量: %d ." % len(validation))
print("在 ID 列中总共有 %d 个空值." % df['ID'].isnull().sum())
print("在 Content 列中总共有 %d 个空值." % df['Content'].isnull().sum())

def text_prepare(text):
    text = text.lower() # 字母小写化
    text = ' '.join([w for w in text.split() if w not in stop_words]) # 删除停用词
    return text
def http_prepare(text):
    text =text.lower()
    urls = re.findall('https?:/?/?(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)
    if urls:
        for url in urls:
            text = url + ' ' + text
    return text
             
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile('[^A-Z^a-z^0-9^ ]')
    line = rule.sub('',line)
    return line
df['Content'] = df['Content'].apply(http_prepare)
df['Content'] = df['Content'].apply(remove_punctuation)
df['Content'] = df['Content'].apply(text_prepare)
validation['Content'] = validation['Content'].apply(http_prepare)
validation['Content'] = validation['Content'].apply(remove_punctuation)
validation['Content'] = validation['Content'].apply(text_prepare)
print("数据总量: %d ." % len(validation))

print(df['Content'].sample(10))
print('----------')
print(validation['Content'].sample(10))
print('----------')



X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['ID'], random_state = 0)
#count_vect = TfidfVectorizer(max_features =80000,min_df=5,max_df=0.9,ngram_range=(1,4),token_pattern= '(\S+)')
count_vect = CountVectorizer(max_features =80000,min_df=1,max_df=0.9,ngram_range=(1,4),token_pattern= '(\S+)')
#print(X_train[:10])
print('-----------')
#print(df['Content'][:10])
X_test_tfidf = count_vect.fit_transform(validation['Content'])
X_train_tfidf = count_vect.transform(X_train)
print(X_train_tfidf.shape)
print(X_test_tfidf.shape)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
#clf = LinearSVC().fit(X_train_tfidf, y_train)
#X_test_tfidf = count_vect.transform(validation["Content"])
y_predict=clf.predict(X_test_tfidf)
print(y_predict[35436])
f = open('result.txt','w')
for i in range(35437):
    if y_predict[i] == None:
        print('有问题')
    f.write(str(y_predict[i]) + '\n')
    f.flush()


#from sklearn.metrics import accuracy_score
#print(len(y_predict))
#print('Accurancy')
#print(accuracy_score(y_test, y_predict))

        





#X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['ID'], random_state = 0)
#vec=TfidfVectorizer(min_df=5,max_df=0.6,ngram_range=(1,2),token_pattern= '(\S+)')

#X_train_counts = vec.fit_transform(X_train)
#X_test=vec.fit_transform(X_test)
#print(X_train_counts)
#print('-------')
#print(X_test)
#tfidf_transformer = TfidfTransformer()

#clf = MultinomialNB()
#clf.fit(X_train_counts, y_train)
#y_predict=clf.predict(X_test)
#from sklearn.metrics import classification_report
#print('The accuracy of Navie Bayes Classifier is',clf.score(X_test,y_test))
#print(classification_report(y_test,y_predict,target_names=df.ID))



#from sklearn.datasets import fetch_20newsgroups   #fetch_20newsgroups是新闻数据抓取器
#news=fetch_20newsgroups(subset='all')   #fetch_20newsgroups即时从互联网下载数据
#print("数据总量: %d ." % len(news))
#from distutils.version import LooseVersion as Version  
#from sklearn import __version__ as sklearn_version  
#from sklearn import datasets  
#if Version(sklearn_version) < '0.18':  
#    from sklearn.cross_validation import train_test_split  
#else:  
#    from sklearn.model_selection import train_test_split  
#X_train,X_test,y_train,y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
#from sklearn.feature_extraction.text import CountVectorizer   
#print(X_train[0])
##特征抽取,将文本特征向量化
#vec=TfidfVectorizer(min_df=5,max_df=0.6,ngram_range=(1,2),token_pattern= '(\S+)')
#X_train=vec.fit_transform(X_train)
#print(X_train)
#print('-----------------------------')
#print(y_train)
#X_test=vec.transform(X_test)
#print('--------------------------')
#print(X_test)
#print('------------------------------')
#from sklearn.naive_bayes import MultinomialNB
#mnb=MultinomialNB()
#mnb.fit(X_train,y_train)
#y_predict=mnb.predict(X_test)
#from sklearn.metrics import classification_report
#print('The accuracy of Navie Bayes Classifier is',mnb.score(X_test,y_test))
#print(classification_report(y_test,y_predict,target_names=news.target_names))










