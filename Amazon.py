# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:01:03 2016

@author: chirag
"""
import os
import json
import pandas as pd
import string 
import matplotlib as mat
import nltk
from textblob import TextBlob
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




os.chdir('C:\\Users\\chirag\\Desktop\\Data Science\\project\\Amazon')


f = open("reviews_Cell_Phones_and_Accessories.json", "rb")


all_data = [json.loads(line.decode('utf-8', 'ignore')) for line in f]

#reduced=All[:10000]

df=pd.DataFrame(all_data)

df.to_csv("all.csv")

df3=pd.DataFrame.from_csv("all.csv")

import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df2 = getDF('meta_Cell_Phones_and_Accessories.json.gz')
df2.to_csv("meta_cell.csv")

df4=pd.DataFrame.from_csv("selected_cell_meta.csv")
result=pd.merge(df3, df4, left_on='asin', right_on='asin') 
result.to_csv("combined_sample.csv")



all_data=pd.DataFrame.from_csv("combined_sample.csv")




new_data=all_data.dropna(subset = ['title','reviewText'])


matrix=new_data.as_matrix()
col=new_data.columns


global temp


def parsing(l):
    p=string.punctuation
    #n=string.digits
    x=p.replace('.','')
    n_x=len(x)
    try:
        print("intry catch")
        table =str.maketrans(x,n_x*" ")
        temp=l.translate(table)
    except:
        print("--------this is exept----------")
        l= 'none'
        temp=l.translate(table)
    l2=temp.lower()
    words=l2.split()     
    stopwords = nltk.corpus.stopwords.words("english")
    words = [w for w in words if w not in stopwords]
    return words
    
    



f_b=[]


model=["iphone 4", "iphone 3","iphone 4s", "galaxy s3", 'galaxy s2','galaxy s4', "nokia n90", 'nokia n93', "iphone 3"]
brand=["samsung","apple","htc",'nokia','motorola']


              

counter=0

# model, brand, features



model_list=[]
brand_list=[]
features=[[],[],[]]               



review_list=[]
X=159000


for i in range(0,X):
    if(model[0] in matrix[i,16].lower()):
        print("fist")
        model_list.append(model[0])
    elif(model[1] in matrix[i,16].lower()):
        print("second")        
        model_list.append(model[1]) 
    elif(model[3] in matrix[i,16].lower()):
        print("third")   
        model_list.append(model[3])
    elif(model[2] in matrix[i,16].lower()):
        print("forth")
        model_list.append(model[2])    
    elif(model[4] in matrix[i,16].lower()):
        print("forth")
        model_list.append(model[4]) 
    elif(model[5] in matrix[i,16].lower()):
        print("forth")
        model_list.append(model[5]) 
    elif(model[6] in matrix[i,16].lower()):
        print("forth")
        model_list.append(model[6]) 
    elif(model[7] in matrix[i,16].lower()):
        print("forth")
        model_list.append(model[7])
    elif(model[8] in matrix[i,16].lower()):
        print("forth")
        model_list.append(model[8])    
    else: 
         model_list.append("")
     
for i in range(0,X):
    if(brand[0] in matrix[i,16].lower()):
        brand_list.append(brand[0])
    elif(brand[1] in matrix[i,16].lower()):
        brand_list.append(brand[1])
    elif(brand[2] in matrix[i,16].lower()):
        brand_list.append(brand[2])
    elif(brand[3] in matrix[i,16].lower()):
        brand_list.append(brand[3])
    elif(brand[4] in matrix[i,16].lower()):
        brand_list.append(brand[4])    
    else:
        brand_list.append("") 

        

time_list=[]        
for i in range(0,X):
    d = datetime.fromtimestamp(int(matrix[i,8]))
    time_list.append(d)

price_list=matrix[0:X,9]

text=[]
for i in range(0,X):   
    review_list=parsing(matrix[i,3])
    text.append(review_list)



counter=0
features=["camera","screen","battery"]
f_all=[[],[],[]]
for i in range(0,X):
    if("galaxy s3" or 'galazy sIII' in matrix[i,16].lower()):
        counter = counter +1
        if(matrix[i,2]>3):       
            review_list=parsing(matrix[i,3])
            for fl in range(len(features)):
                #if features[fl] in review_list:
                if features[fl] in review_list: 
                    print("this is i :",i)
                    print("this is features",features[fl])
                    #counter = counter +1
                    for j in range(len(review_list)):
                        if review_list[j] == features[fl]: 
                            print('review list:',review_list[j])
                            l=len(review_list)
                            print(j)
                            print('l',l)    
                            if(j>1 and j<l-2):
                                if '.' in review_list[j]:
                                    if(j>2 and j<l):
                                        print("------first------")
                                        print("reviw:  ",review_list[j-2],' ',review_list[j].replace('.',''),' ',review_list[j-1])                         
                                        f_all[fl].append(review_list[j-1].replace('.',''))
                                        f_all[fl].append(review_list[j+1].replace('.',''))
                                if '.' in review_list[j-1]:        
                                    if(j>1 and j<l-3):
                                        print("------second------")
                                        print("reviw:  ",review_list[j+1],' ',review_list[j],review_list[j+2])
                                        f_all[fl].append(review_list[j-1].replace('.',''))
                                        f_all[fl].append(review_list[j+1].replace('.',''))                                
                                if '.' in review_list[j+1]:        
                                    if(j>2 and j<l-1):
                                        print("------third------")
                                        print("reviw:  ",review_list[j-1],' ',review_list[j],' ',review_list[j+1].replace('.',''))         
                                        f_all[fl].append(review_list[j-1].replace('.',''))
                                        f_all[fl].append(review_list[j+1].replace('.','')) 
                                else:
                                    print("------forth------")
                                    print("reviw:  ",review_list[j-2],review_list[j-1],' ',review_list[j],' ',review_list[j+1],review_list[j+2])
                                    f_all[fl].append(review_list[j-1].replace('.',''))
                                    f_all[fl].append(review_list[j+1].replace('.',''))
                                    f_all[fl].append(review_list[j-2].replace('.',''))
                                    f_all[fl].append(review_list[j+2].replace('.',''))


                     
df_camera=pd.DataFrame({"galazyS3_camera_45":f_all[0]})

text=''
for f in f_all[0]:
    text=text+' '+ f

blob=TextBlob(text)
b=blob.tags
b=pd.DataFrame(b)


b.to_csv("galaxs3_camera_45_2.csv")
#b.to_csv("samsung_camera3.csv")

df_camera.to_csv("galazys3_camera45.csv")

df_screen=pd.DataFrame({"screen":f_all[1]})
df_screen.to_csv("galazys3 Screen45.csv")

df_battery=pd.DataFrame({"battery":f_all[2]})
df_battery.to_csv("galazys3 battery 45.csv")



# sentimental analytics

file = open("C:/Users/chirag/Desktop/Data Science/hw/hw3/AFINN.txt")
afinn=file.readlines()


scores={}
for l in afinn:
    l2=l.replace("\n","")
    key,value=l2.split("\t")
    scores[key]=int(value)


sentiment=[]
def fun_sentiment(text):
    review_list=text    
    for rw in review_list:
        tw_score=0
        #print("this is c")
        #c=c+1
        for word in rw:
            for k in scores:
                if(word==k):
                    value=scores[k]
                    tw_score=tw_score+value 
        sentiment.append(tw_score)        
    return sentiment       
    
senti=fun_sentiment(text)    
  

score_list=matrix[0:X,2]
                     

df_camera=pd.DataFrame({"model":model_list,"price":price_list,"brand":brand_list,"time":time_list,'affin':senti,'review_score':score_list})
df_camera.to_csv("model_brand_name.csv")


# Topic Modeling

features=["samsung","apple","motorola","nokia"]
brandwise_list=[[],[],[],[]]
for i in range(0,15900):
    if("samsung" in matrix[i,16].lower()):
        brandwise_list[0].append(matrix[i,3])
    elif("apple" in matrix[i,16].lower()):
        brandwise_list[1].append(matrix[i,3])
    elif("motorola" in matrix[i,16].lower()):
        brandwise_list[2].append(matrix[i,3])
    elif("nokia" in matrix[i,16].lower()):
        brandwise_list[3].append(matrix[i,3])    
ls=[]        
for i in range(0,4):
    temp=""
    for rw in brandwise_list[i]:
        print(rw)
        temp=rw+temp
    ls.append(temp)    
            
            
corpus=ls
    
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
dtm = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(dtm.toarray()) 

from sklearn import decomposition

vocab = np.array(vectorizer.get_feature_names())

num_topics = 4
num_top_words = 20
clf = decomposition.NMF(n_components = num_topics, random_state=0)
doctopic = clf.fit_transform(dtm)
topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])
    
    
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:20])))


