# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:57:36 2016

@author: chirag
"""



import os
import string
import pandas as pd
import sys
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from anew_module import anew 
import nltk
import glob



os.chdir('C:\\Users\\chirag\\Desktop\\Data Science\\hw\\hw3')


f = open("twitter_trump.json", "rb")
tweets_list_trump = [json.loads(line.decode('utf-8', 'ignore')) for line in f]


f = open("twitter_cruz.json", "rb")
tweets_list_cruz = [json.loads(line.decode('utf-8', 'ignore')) for line in f]

f = open("twitter_clinton.json", "rb")
tweets_list_clinton = [json.loads(line.decode('utf-8', 'ignore')) for line in f]

f = open("twitter_sanders.json", "rb")
tweets_list_sanders = [json.loads(line.decode('utf-8', 'ignore')) for line in f]



# TimeZone


def fun_timezone(tweets):
        tweets_list=tweets
        timezone_list=[]
        i=0
        for tw in tweets_list:
            i =i+1
            if i==500:
                break
            if 'user' not in tw.keys():
                #tw.setdefault('user', 'NoValue')
                timezone_list.append("NA")
                #print("in if ")
                continue
            if "time_zone" not in tw["user"].keys():
                timezone_list.append("NA")
                continue
                #tw["user"].setdefault('time_zone', 'NoValue')   
            
            temp1=tw["user"]["time_zone"]
            timezone_list.append(temp1)
        return timezone_list
        
timezone_list_trump=fun_timezone(tweets_list_trump)
timezone_list_cruz=fun_timezone(tweets_list_cruz)
timezone_list_clinton=fun_timezone(tweets_list_clinton)
timezone_list_sanders=fun_timezone(tweets_list_sanders)


def fun_text(tweets):         
        text_list=[]
        tweets_list=tweets
        i=0
        for ln in tweets_list:
            i=i+1
            if i==500:
                break
            #print(i)
            if 'text' not in ln.keys():
                ln.setdefault('text', 'NoValue')
                
            txt=ln["text"]
            text_list.append(txt)
        return text_list


text_list_trump=fun_text(tweets_list_trump)
text_list_cruz=fun_text(tweets_list_cruz)
text_list_clinton=fun_text(tweets_list_clinton)
text_list_sanders=fun_text(tweets_list_sanders) 

p=string.punctuation
n=string.digits
x=p+n+'\n'
n_x=len(x)
table =str.maketrans(x,n_x*" ")

def fun_punctuation(text):
    parsed_text=[]
    text_list=text
    for l in text_list:
        
        temp=l.translate(table)
        parsed_text.append(temp)

    wo_stop_list=[]    
    for l in parsed_text:
        l2=l.lower()
        words=l2.split()     
        stopwords = nltk.corpus.stopwords.words("english")
        stopwords.append("trump")
        words = [w for w in words if w not in stopwords]
        wo_stop_list.append(words)
    return wo_stop_list     

wo_stop_list_trump=fun_punctuation(text_list_trump)
wo_stop_list_cruz=fun_punctuation(text_list_cruz)
wo_stop_list_clinton=fun_punctuation(text_list_clinton)
wo_stop_list_sanders=fun_punctuation(text_list_sanders)       

# AFINN


file = open("C:/Users/chirag/Desktop/Data Science/hw/hw3/AFINN.txt")
afinn=file.readlines()



scores={}
for l in afinn:
    l2=l.replace("\n","")
    key,value=l2.split("\t")
    scores[key]=int(value)


file = open("C:/Users/chirag/Desktop/Data Science/hw/hw3/lexicon/Positive.txt")
p=file.read()
pos=p.split("\n")


file = open("C:/Users/chirag/Desktop/Data Science/hw/hw3/lexicon/Negative.txt")
n=file.read()
neg=n.split("\n")


name_list=['Trump','Cruze','Clinton','Sanders']

i=0

tr_len=len(wo_stop_list_trump)
cr_len=len(wo_stop_list_cruz)
cl_len=len(wo_stop_list_clinton)
sa_len=len(wo_stop_list_sanders)

field1=[]
field2=[]
field3=[]
field4=[]


for i in range(tr_len):
    field1.append('Trump')
for i in range(cr_len):
    field2.append('Cruz')
for i in range(cl_len):
    field3.append('Clinton')
for i in range(sa_len):
    field4.append('Sanders')



def fun_sentiment(text_wo_stop,name):
    print("ALL SENTIMENT SCORE FOR  :  ",name)
    name1=name
    wo_stop_list=text_wo_stop
    sentiment=[]
    for tw in wo_stop_list:
        tw_score=0
        #print("this is c",c)
        #c=c+1
        for word in tw:
            for k in scores:
                if(word==k):
                    #print("word",word,"afinn",k)
                    value=scores[k]
                    tw_score=tw_score+value
        #print(tw_score) 
        sentiment.append(tw_score)     
        s=sum(sentiment)     
    print("1. Overall Afinn Score:", s)
    #all_senti=sentiment
    #Hu and Liu Lexicon    
    pos_sent=[]
    for tw in wo_stop_list:
        ps_score=0
        #print("this is c",c)
        #c=c+1
        for word in tw:
            for k in pos:
                value=0            
                if(word==k):
                    #print("word",word,"pos",k)
                    value=1
                    ps_score=ps_score+value
        #print(ps_score) 
        pos_sent.append(ps_score)
    #all_pos_senti=all_pos_senti+pos_sent
    p1=sum(pos_sent)     
    print("2.1  Lexicon Total Positive Score:", p1)
      
    neg_sent=[]
    for tw in wo_stop_list:
        ns_score=0
        #print("this is c",c)
        #c=c+1
        for word in tw:
            for k in neg:
                value=0            
                if(word==k):
                    #print("word",word,"Neg",k)
                    value=-1
                    ns_score=ns_score+value
        #print(ns_score) 
        neg_sent.append(ns_score)
    #all_neg_senti=all_neg_senti+neg_sent    
    n1=sum(neg_sent)
    print("2.2  Lexicon Total Negative Score:", n1)
    # ANEW sentimental analysis
    anew_arousal_list=[]
    anew_valence_list=[]
    for tw in wo_stop_list:
        anew_senti={'arousal': 0.0, 'valence': 0.0}
        for w in tw:
            temp=anew.sentiment(w)
            #print(temp["arousal"])
            anew_senti["arousal"]=anew_senti["arousal"]+temp['arousal']
           
            #print(temp["valence"])
            anew_senti["valence"]=anew_senti["valence"]+temp['valence']
            
        a=round(anew_senti["arousal"],2)
        v=round(anew_senti["valence"],2)
     
        anew_arousal_list.append(a)
        anew_valence_list.append(v)
    #all_anew_a=all_anew_a+anew_arousal_list
    #all_anew_v=all_anew_v+anew_valence_list
    p_a=sum(anew_arousal_list)
    p_v=sum(anew_valence_list)
    print("3.1  ANEW_arousal : ",round(p_a,2)) 
    print("3.2 ANEW_valence :  ",round(p_v,2),'\n')    
    if(name1 =='Trump'):
        
        df1 = pd.DataFrame({'Candidate' : field1, 'affin' :sentiment,"lexicon+pos":pos_sent,"lexicon+neg":neg_sent,
                       "time_zone":timezone_list_trump,"anew_arousal":anew_arousal_list,"anew_vaence":anew_valence_list})
        return df1               
    if(name1 =='Cruze'):
        df2 = pd.DataFrame({'Candidate' : field2, 'affin' :sentiment,"lexicon+pos":pos_sent,"lexicon+neg":neg_sent,
                       "time_zone":timezone_list_cruz,"anew_arousal":anew_arousal_list,"anew_vaence":anew_valence_list})
        return df2          
    if(name1 =='Clinton'):
        df3 = pd.DataFrame({'Candidate' : field3, 'affin' :sentiment,"lexicon+pos":pos_sent,"lexicon+neg":neg_sent,
                       "time_zone":timezone_list_clinton,"anew_arousal":anew_arousal_list,"anew_vaence":anew_valence_list})
        return df3     
    if(name1 == 'Sanders'):
        df4 = pd.DataFrame({'Candidate' : field4, 'affin' :sentiment,"lexicon+pos":pos_sent,"lexicon+neg":neg_sent,
                       "time_zone":timezone_list_sanders,"anew_arousal":anew_arousal_list,"anew_vaence":anew_valence_list})
        return df4     

df11=fun_sentiment(wo_stop_list_trump,name_list[0])
df22=fun_sentiment(wo_stop_list_cruz,name_list[1])
df33=fun_sentiment(wo_stop_list_clinton,name_list[2])
df44=fun_sentiment(wo_stop_list_sanders,name_list[3])

frames = [df11, df22, df33,df44]
df = pd.concat(frames)

df.to_csv('all.csv', sep=',',index=False)

# Topic Modeling



tweets=[]


all_tweets =text_list_trump+text_list_cruz+text_list_clinton+text_list_sanders 

for tx in all_tweets:
    temp=tx.translate(table)
    tweets.append(temp.lower())
    


'''
temp=''

for tx in text_list_trump:
    temp=tx+temp
tweets.append(temp)    
temp=''

for tx in text_list_cruz:
    temp=tx+temp
tweets.append(temp)

temp=''
for tx in text_list_clinton:
    temp=tx+temp
tweets.append(temp)

temp=''
for tx in text_list_sanders:
    temp=tx+temp
tweets.append(temp)
'''
corpus=tweets
    
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
dtm = vectorizer.fit_transform(corpus)

#names = [tw[:fn.find(".")] for tw in tweets] #get file names without .txt

from sklearn import decomposition

vocab = np.array(vectorizer.get_feature_names())

num_topics = 25
num_top_words = 20
clf = decomposition.NMF(n_components = num_topics, random_state=0)
doctopic = clf.fit_transform(dtm)
topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])
    
    
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))







    












