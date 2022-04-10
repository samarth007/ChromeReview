import pandas as pd
from preprocessing import preprocessing
from Logs import logger
from Create_File import CreateModel

class Predict:

    def __init__(self,path):
      self.d=path
      self.logger=logger.App_logger()
      self.file_obj=open('TestingLogs/Test_logs.txt','a+')

    def predictionFromModel(self):
        self.logger.log(self.file_obj,"Prediction started")
        self.data=pd.read_csv(self.d+"/"+'testdata.csv')
        preprocess = preprocessing(self.logger, self.file_obj)
        df=self.data  #assigning to new variable X
        df=preprocess.dropIrrelevantColumns(df)  #dropping irrelevant column
        df=preprocess.DropNan(df)               # dropping nan values
        X=preprocess.separate_label_test(df,'Star')    #separating independent variable
        X=preprocess.CleansingPredict(X)                # converting text into vectors
        X=preprocess.Scaling(X)                 # scaling ID feature
        getModel=CreateModel(self.logger,self.file_obj)
        model=getModel.load_model()
        result=list(model.predict(X))
        result=pd.DataFrame(list(zip(result)),columns=['prediction'])
        result.to_csv('Prediction/predict.csv',index=False)
        self.logger.log(self.file_obj,"Model prediction done")



