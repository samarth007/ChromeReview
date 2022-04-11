# ChromeReview
In this project the goal is to identify such ratings where review text is good, but rating is negative- so that the support team can point this to users. 
We have ID,Review URL,Text,Star,Thumbs Up,User Name,Developer Reply,Version,Review Date,App ID features.
Basicly it is classification usecase where we have to classify those customers who is givin bad review but good rating
I've implemented this project using flask framework, Where I've used logger class for logging the activity of the each case.
Basically all the preprocessing activity like, label separation, detecting Nan values, splitting the sentence, removal of stopwords, lemmatization and converting word to vector using Tf-Idf vectorizer is performed in the preprocessing.py file.
Once preprocessing is done then we are feeding the matrix to model for training, In case of Model training I've adopted 2 models, Gaussian Navie Baye's and MultinomialNB. Data is trained on both the models and whichever model is giving good accuracy we are gonna save that model as pickle file in the Model folder.
At the time of prediction, Again the same preprocessing task is done on the test data but the only changes is instead of fit_transform we will be using transform method.
After preprocessing, The test data is applied on the model which we have saved while model training and output is saved as Predict.csv in Prediction folder.



**Suggested questions:**
1).Is there any co-relation between short description, long description and ranking? Does the placement of keyword (for example - using a keyword in the first 10 words - have any co-relation with the ranking)? ---- Yes there is a correlation..
2).Does APP ID (Also known as package name) play any role in ranking? --- app id is not playing any role in ranking..
3).Any other pattern or good questions that you can think of and answer?--- No..



**Write a regex to extract all the numbers with orange color background from the below text in italics.**

lst={"orders":[{"id":1},{"id":2},{"id":3},{"id":4},{"id":5},{"id":6},{"id":7},{"id":8},{"id":9},{"id":10},{"id":11},{"id":648},{"id":649},{"id":650},{"id":651},{"id":652},{"id":653}],
     "errors":[{"code":3,"message":"[PHP Warning #2] count(): Parameter must be an array or an object that implements Countable (153)"}]}

ss=re.compile(r'\d')

for i,j in lst.items():
    maths=ss.finditer(str(j))
    for i in maths:
        print(i.group())
