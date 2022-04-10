
import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import re


class preprocessing:
    tf=TfidfVectorizer()
    def __init__(self,logger,fil_obj):
        self.logger=logger
        self.file_obj=fil_obj

    def separate_label(self,data,label_column_name):
        self.X=data.drop([label_column_name,'Star'],axis=1)
        self.y=data[label_column_name]
        self.logger.log(self.file_obj,"label separation done")
        return  self.X,self.y

    def separate_label_test(self,data,label):
        self.X=data.drop(label,axis=1)
        self.logger.log(self.file_obj,"Label separation for prediction")
        return self.X

    def dropIrrelevantColumns(self,data):
        self.X= data.drop(['Review Date','App ID','Version','Developer Reply','Review URL','Thumbs Up','User Name'],axis=1)
        self.logger.log(self.file_obj,'Dropped Irrelevant columns')
        return self.X

    def IsBadReview(self,data,label):
        data['bad_review']=data[label].apply(lambda x:1 if x<=2 else 0)
        self.logger.log(self.file_obj,'Added new column as Bad_review')
        return data

    def DropNan(self,X):
        self.null_col=X.isna().sum()
        for i in self.null_col:
            if i >0:
               X[self.null_col.index[i]].dropna()
        self.logger.log(self.file_obj,'Nan values dropped from row')
        return X



    def cleaning(self,data):
        lm=WordNetLemmatizer()
        corpus=[]
        for i in range(len(data)):
            text=re.sub("[^a-zA-Z]",' ',str(data['Text'][i]))
            text=text.lower()
            text=text.split()
            text=[lm.lemmatize(j) for j in text if j not in set(stopwords.words('english'))]
            text= ' '.join(text)
            corpus.append(text)
        vect=preprocessing.tf.fit_transform(corpus).toarray()
        data=pd.concat([data.drop('Text',axis=1),pd.DataFrame(vect)],axis=1)
        self.logger.log(self.file_obj,'Data cleaning and converting text to vector')
        return data

    def Scaling(self,data):
        std=MinMaxScaler()
        data['ID']=std.fit_transform(data[['ID']])
        return data

    def CleansingPredict(self,data):
        lm=WordNetLemmatizer()
        corpus=[]
        for i in range(len(data)):
            text=re.sub("[^a-zA-Z]",' ',str(data['Text'][i]))
            text=text.lower()
            text=text.split()
            text=[lm.lemmatize(j) for j in text if j not in set(stopwords.words('english'))]
            text= ' '.join(text)
            corpus.append(text)
        vect=preprocessing.tf.transform(corpus).toarray()
        data=pd.concat([data.drop('Text',axis=1),pd.DataFrame(vect)],axis=1)
        self.logger.log(self.file_obj,'Data cleaning and converting text to vector')
        return data