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
@application.route('/old')
def say_hello():
    if request.is_json:
      content = request.get_json()
      return 'Hello Python with JSON'
    else:
      return 'No JSON'
 
@application.route('/articles')
def api_articles():
    return 'List of article'

@application.route('/autocode', methods = ['POST'])
def auto_code():
    parsed_json = request.get_json()
    train_data = parsed_json[0]['train']
    test_data = parsed_json[1]['test']
    data = pd.DataFrame(train_data)
    coding = data['final_value']
    features = data.drop('final_value', axis=1)
    features_final = pd.get_dummies(features)
    
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
    
    
    d2 = { 'coding' : final_coding}
    values = pd.DataFrame(data=d2)
    values = values.as_matrix().astype(np.int)
    X_train, X_test, y_train, y_test = train_test_split(features_final, values, test_size = 0.2, random_state = 12)
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    model_score = accuracy_score(y_test, predictions)
    
    
    testdata = pd.DataFrame(test_data)
    testfeatures_final = pd.get_dummies(testdata)
    predictions = model.predict(testfeatures_final)
           
    predictions2 = {}
    for i in range(predictions.size):
      predictions2[i] = output_dict[int(predictions[i])]
    
    return jsonify({'score':model_score, 'predictions':predictions2})

# run the app.
if __name__ == '__main__':
    application.run()
