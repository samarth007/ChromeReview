from flask import Flask,Response
from flask import request
from Train_ModelTrain import train_model
from PredictModel import Predict

app=Flask(__name__)

@app.route("/train",methods=['POST'])
def train():
  model_train=train_model()
  model_name=model_train.training_model()
  return Response(model_name)

@app.route("/predict",methods=['POST'])
def predict():
  if request.json['filePath'] is not None:
     path=request.json['filePath']
     pred=Predict(path)
     pred.predictionFromModel()
     return Response('Prediction Done')




if __name__=='__main__':
  app.run()