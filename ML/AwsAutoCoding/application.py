from flask import Flask,request
from json import dumps
from flask_jsonpify import jsonify
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# print a nice greeting.
@application.route('/')
def say_hello():
    return 'Machine Learning Auto Coding'

@application.route('/autocode', methods = ['POST'])
def auto_code():
    try:
        ## Read the train and test data from json requests
        parsed_json = request.get_json()
        train_data = parsed_json['trainingData']
        test_data = parsed_json['predData']
    except:
      return  jsonify({'exception'  : 'Exception raised in parsing json requests'})

    try:
        #Convert the training data to dataframe, get the final value and store it in a separate
        ## object called coding
        data = pd.DataFrame(train_data)
        coding = data['final_value']
        ##Store only features by dropping final value
        features = data.drop('final_value', axis=1)
        #Convert the characters to numerical value using get dummies
        features_final = pd.get_dummies(features)
    except:
      return  jsonify({'exception'  : 'Exception raised in reading training data'})
 
    try:   
        #Store the actual charge account in dictionary to avoid run time errors because of
        #long charge account value
        coding_dict = {}
        output_dict = {}
        coding_index = 0
        final_coding = coding
        
        i = 0
        for i in range(coding.size):
          if coding[i] in coding_dict:
            final_coding[i] = coding_dict[coding[i]]
          else:
            coding_index = coding_index + 1
            coding_dict[coding[i]] = coding_index
            output_dict[coding_index] = coding[i]
            final_coding[i] = coding_dict[coding[i]]
        
        #Get the final values and store in separate object
        d2 = { 'coding' : final_coding}
        values = pd.DataFrame(data=d2)
        values = values.as_matrix().astype(np.int)
    except:
      return  jsonify({'exception'  : 'Exception raised in reading final data values'})
 
    try:
        # Create the model, fit, predict and get the accuracy score
        X_train, X_test, y_train, y_test = train_test_split(features_final, values, test_size = 0.2, random_state = 12)
        model = GaussianNB()
        model.fit(X_train, np.ravel(y_train,order='C'))
        predictions = model.predict(X_test)
        model_score = accuracy_score(y_test, predictions)
    except:
      return  jsonify({'exception'  : 'Exception raised in preparing Naive Bayes Model'})
 
    try:       
        #Get the testdata, store the lineid in a separate object so that final return object can have these values
        testdata = pd.DataFrame(test_data)
        lineIdValues = testdata['lineId']
        testdata = testdata.drop('lineId',axis=1)
        testfeatures_final = pd.get_dummies(testdata)
        predictions = model.predict(testfeatures_final)
    except:
      return  jsonify({'exception'  : 'Exception raised in model prediction'})
               
    try:
        predictions2 = []
        for i in range(predictions.size):
          predictions2.append(output_dict[int(predictions[i])])
        
        #Add predicted value, score and line id to final object
        testdata['final_value'] = predictions2
        testdata['score'] = model_score * 100
        testdata['lineId'] = lineIdValues
        #final_json = {'predictions':testdata.to_dict(orient='record')}
        return testdata.to_json(orient='records')
         
    except:
      return  jsonify({'exception'  : 'Exception raised in returning final json result'})

# run the app.
if __name__ == '__main__':
    application.run()
