from ModelTrainer import Tuner
from sklearn.model_selection import train_test_split

class Model_Finder:

    def __init__(self,logger,fil_obj):
        self.logger=logger
        self.file_object=fil_obj


    def getbestmodel(self,X,Y):
        self.logger.log(self.file_object,"Retriving best model")
        model=Tuner(self.logger,self.file_object)
        x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.3,random_state=32)
        self.NGB,self.NGB_score=model.bestGaussNB(x_train,y_train,x_test,y_test)
        self.MGB,self.MGB_score=model.bestMultiNB(x_train,y_train,x_test,y_test)


        if(self.NGB_score > self.MGB_score):
            self.logger.log(self.file_object,'Model retrived was GNB')
            return "NormalGaussianNB",self.NGB
        else:
            self.logger.log(self.file_object, 'Model retrived was MultiNB')
            return "MultiNominalNB",self.MGB
