import pandas as pd
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

data=pd.read_csv("InputCsv/chrome_reviews.csv")
data=data.drop(['Review Date','App ID','Version','Developer Reply','Review URL','Thumbs Up'],axis=1)
c=data.isna().sum()
corpus=[]
lm=WordNetLemmatizer()

for i in range(len(data)):
    text=re.sub("[^a-zA-Z]",' ',str(data['Text'][i]))
    text=text.lower()
    text=text.split()
    text=[lm.lemmatize(j) for j in text if j not in set(stopwords.words('english'))]
    text= ' '.join(text)
    corpus.append(text)

tf=TfidfVectorizer()
vect=tf.fit_transform(corpus).toarray()
print(type(vect))
data = pd.concat([data.drop('Text', axis=1), pd.DataFrame(vect)], axis=1)
data.to_csv('vect.csv')