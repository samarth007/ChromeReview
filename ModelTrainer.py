
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import recall_score


class Tuner:
    def __init__(self,logger,file_obj):
        self.logger=logger
        self.file_object=file_obj

    def bestGaussNB(self,x,y,x_test,y_test):
        gb=GaussianNB()
        gb.fit(x,y)
        y_pred=gb.predict(x_test)
        score=recall_score(y_pred,y_test)
        self.logger.log(self.file_object,"Model training done on Gaussian NB"+" "+str(score))
        return gb,score

    def bestMultiNB(self,x,y,x_test,y_test):
        Mb=MultinomialNB()
        Mb.fit(x,y)
        y_pred=Mb.predict(x_test)
        score=recall_score(y_pred,y_test)
        self.logger.log(self.file_object, "Model training done on MultiNorma NB"+" "+str(score))
        return Mb,score









