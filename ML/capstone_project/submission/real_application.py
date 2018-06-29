from flask import Flask,request
from json import dumps
from flask import jsonify
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import warnings
from Crypto.Cipher import AES
import base64
import time

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# print a welcome greeting.
@application.route('/')
def say_hello():
    return 'Machine Learning Auto Coding'

def _unpad(s):
  print('unpad begin')
  s2 = lambda s: s[0:-ord(s[-1:])]
  print('unpad completed')
  print(s2)
  return s2

## Decrypt the token from header request, token is valid when entityid and headerid
# are greater than zero
def decrypt(str):
  RandomBool = True
  try:
     key=b'dAtAbAsE98765432'
     iv=b'0000000000000000'
     cipher = AES.new(key, AES.MODE_CBC, iv)
     resolved=cipher.decrypt(base64.b64decode(str))
     unpad = lambda s: s[0:-ord(s[-1:])]
     unpad_token=unpad(resolved)
     decode_token=unpad_token.decode()
     tokendata=decode_token.split("|")
     entityId=int(tokendata[1])
     headerId=int(tokendata[3])
     token_ms=int(tokendata[4])
     current_ms = int(round(time.time() * 1000))
     #If the token generated is more than 1 minute back, set as false
     if (current_ms-token_ms) > 60000:
       RandomBool = False
     elif entityId > 0 and headerId > 0:
       RandomBool = True
     else:
       RandomBool = False
  except Exception as e:
    RandomBool = False
  return RandomBool

@application.route('/autoCoding', methods = ['POST'])
def auto_code():
    try:
        #Read the authorization from header and decrypt, if token is false don't train and predict
        #Train and Predict only when token is valid
        token = request.headers.get('Authorization')
        try:
          valid=decrypt(token)
          #Decrypt returned false then throw an error as Invalid Token
          if not valid:
            return  json.dumps({'returnCode':'F','returnText':'Failure because of Invalid Token' }), 401
        except Exception as e:
          return json.dumps({'returnCode':'F','returnText':'Failure because of Invalid Token'}), 401
        ## Read the train and test data from json requests
        parsed_json = request.get_json()
        train_data = parsed_json['trainingData']
        pred_data = parsed_json['predData']
    except Exception as e:
      return json.dumps({'returnCode':'F','returnText':'Exception raised in parsing json requests ' + str(e)}), 500

    try:
        #Convert the training data to dataframe, use label encoding for all features
        #fit & transform features, store the charge account id which is Y-array in coding
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        data = pd.DataFrame(train_data)
        features=data
        le_dict = {col: LabelEncoder() for col in data.columns }
        for col in data.columns:
          features[col]=le_dict[col].fit_transform(data[col])

        coding = features['final_value']
        ##Store only features by dropping final value
        features = features.drop('final_value', axis=1)
    except Exception as e:
      return json.dumps({'returnCode':'F','returnText':'Exception raised in reading training data ' + str(e)}), 500

    try:
        # Create the model, fit, predict and get the accuracy score
        X_train, X_test, y_train, y_test = train_test_split(features, coding, test_size = 0.2, random_state = 12)
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_score = accuracy_score(y_test, predictions)
    except Exception as e:
      return json.dumps({'returnCode':'F','returnText':'Exception raised in preparing Naive Bayes Model ' + str(e)}), 500
 
    try:       
        #Get the testdata, store the lineid in a separate object so that final return object can have these values
        preddata = pd.DataFrame(pred_data)
        lineIdValues = preddata['lineId']
        predfeatures_final = preddata.drop('lineId',axis=1)
        predictions2=[]
        predMessage=[]
        #Loop through each test set which will get the transformed value, predict charge account
        #In case of an exception because of new value error, set charge account as empty 
        for i in range(predfeatures_final.shape[0]):
          
          try:
            j = 0
            for col in predfeatures_final[i:i+1].columns:
              predfeatures_final.loc[i][j]=le_dict[col].transform(predfeatures_final[i:i+1][col])
              j=j+1
            predictions=[]
            predictions = model.predict(predfeatures_final[i:i+1])
            predictions2.append(le_dict['final_value'].inverse_transform(predictions[0]))
            predMessage.append('Success')
          except Exception as e:
            predictions2.append('')
            predMessage.append(str(e))

        preddata['final_value'] = predictions2
    except Exception as e:
      return json.dumps({'returnCode':'F','returnText':'Exception raised in model prediction ' + str(e)}), 500
      
    #Set the score, lineId and return json object         
    try:
      preddata['score'] = model_score * 100
      preddata['lineId'] = lineIdValues
      preddata['predMessage'] = predMessage
      return preddata.to_json(orient='records')        
    except Exception as e:
      return json.dumps({'returnCode':'F','returnText':'Exception raised in returning final json result ' + str(e)}), 500

# run the app.
if __name__ == '__main__':
    application.run()
