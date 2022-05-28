import pandas as pd
import numpy as np
import os
df=pd.read_csv('MetaData.csv')
df.head(5)
df.info()
df.columns
df['Ratings'][0]
df.describe()
df['imdbRating']
df['imdbRating']>5
df['imdbRating'].hist(bins=10)
df['Title']
string = "Hey! What's up?(44)IV"

new_string = ''.join(char for char in string if char.isalnum())
print(new_string)
org_fnames=[]
list_fnames=[]
path = 'D:/PROJECT/FINAL_PROJECT_MPAA/scripts/'
files = os.listdir(path)
for f in files:
    print(f,type(f))
    new_string = ''.join(char for char in f[:-4] if char.isalnum())
    list_fnames.append(new_string.lower())
    org_fnames.append(f)
for i in list_fnames:
    print(i)
len(list_fnames)
org_fnames
len(org_fnames)
df['Year'][1]
c=0
for x in df['Title']:
    print(c,x)
    c=c+1
list_titles=[]
org_titles=[]
c=0
for x in df['Title']:
    new_string = ''.join(char for char in x if char.isalnum())
    print(x,new_string)
    list_titles.append(new_string.lower()+df['Year'][c])
    c=c+1
    org_titles.append(x)
for i in list_titles:
    print(i)
print(len(list_fnames))
print(len(list_titles))
print(len(org_titles))
c=0
list_script=[]
for i in list_titles:
    if i in list_fnames:
        print(i,'present')
        fname=path+org_fnames[c]
        f1=open(fname)
        data=f1.read()
        f1.close()
        list_script.append(data)
    else:
        list_script.append('NA')
    c=c+1
       
for x in list_script:
    print(x)
    print('______________________________________________________')
len(list_script)
l1=list_script[0].split('\n')
l1
df.columns
imd=[]
rt=[]
meta=[]

for i in range(len(df['Ratings'])):
    
    d=df['Ratings'][i].split(',')
    print(df['Ratings'][i],len(d))
    if len(d)==2:
        if d[0]=='Internet Movie Database':
            imd.append(d[1])
            rt.append('na')
            meta.append('na')
        if d[0]=='Rotten Tomatoes':
            rt.append(d[1])
            imd.append('na')
            meta.append('na')
        if d[0]=='Metacritic':
            meta.append(d[1])
            imd.append('na')
            rt.append('na')
    elif len(d)==4:
        if d[0]=='Internet Movie Database' and d[2]=='Rotten Tomatoes':
            imd.append(d[1])
            rt.append(d[3])
            meta.append('na')
        elif d[0]=='Internet Movie Database' and d[2]=='Metacritic':
            imd.append(d[1])
            rt.append('na')
            meta.append(d[3])
        elif d[0]=='Rotten Tomatoes' and d[2]=='Metacritic':
            imd.append('na')
            rt.append(d[0])
            meta.append(d[3])
        
    elif len(d)==6:
        imd.append(d[1])
        rt.append(d[3])
        meta.append(d[5])

for i in imd:
    print(i)
df['Genre']
list_script1=[]
for x in list_script:
    l1=x.split('\n')
    data=''
    for y in l1:
        if y!='':
            y=y.lstrip()
            data=data+'#'+y.rstrip()
        list_script1.append(data)
df['Script']=list_script
df.head(5)
df.to_csv('new1.csv')
df1=pd.read_csv('new1.csv')
df1
df1['Script']
f=open('partition.txt','r')
data=f.read()
f.close()
data
data=data[8:-4]
data
l=data.split(',')
l
l=l[1:-1]
list_train_imdbID=[]
for i in l:
    print(i[2:-1])
    list_train_imdbID.append(i[2:-1])
len(list_train_imdbID)
df1.index
df1.columns[1:]
list_train_imdbID[0]
count=0
l2=[]
for i in df1.index:
    print(df1['imdbID'][i])
    if df1['imdbID'][i] in list_train_imdbID:
        count=count+1
        l1=[]
        for j in df1.columns[1:]:
            l1.append(df1[j])
        l2.append(l1)
count
len(list_train_imdbID)
l2
len(l2)
df2=pd.DataFrame(l2,columns=df1.columns[1:])
df2
df2.to_csv("training1.csv")
df2
len(df2)
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()
# Setting stopwords
STOPWORDS = set(stopwords.words('english'))
import nltk

df2.info()
df2.columns
rating_series=df2['Ratings']
df2['Production']
from IPython import display

c=0

for i in range(len(df2)):
    print(df2['Poster'][i][c])
    script=df2['Script'][i][c]
    print(script)
    score = analyzer.polarity_scores(script)
    p=float(score['pos'])
    ne=float(score['neu'])
    n=float(score['neg'])
    performance = [p,n,ne]
    print(performance)
    index=performance.index(max(performance))
    result=''
    if (index==0):
        result='positive'
    elif (index==1):
        result='negative'

    elif (index==2):
        result='neutral'
    print('result',result)
    display.Image(df2['Poster'][i][c])
    print('_____________________________________')
    c=c+1
