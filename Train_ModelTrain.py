from Logs import logger
from Create_File import CreateModel
import pandas as pd
from preprocessing import preprocessing
from ModelFind import Model_Finder


class train_model:
    def __init__(self):
        self.logger=logger.App_logger()
        self.file_object=open("TrainingLogs/TrainModel_log.txt",'a+')

    def training_model(self):
        self.logger.log(self.file_object, 'Model training starts')
        self.data=pd.read_csv("InputCsv/Chrome_Review.csv")
        preprocess=preprocessing(self.logger,self.file_object)
        self.data=preprocess.dropIrrelevantColumns(self.data)
        self.data=preprocess.DropNan(self.data)
        self.data=preprocess.IsBadReview(self.data,'Star')
        X,Y=preprocess.separate_label(self.data,'bad_review')
        X=preprocess.cleaning(X)
        X=preprocess.Scaling(X)
        model_finder=Model_Finder(self.logger,self.file_object)
        model_name,model=model_finder.getbestmodel(X,Y)
        save_model=CreateModel(self.logger,self.file_object)
        save_model.save_file(model,model_name)
        return model_name





