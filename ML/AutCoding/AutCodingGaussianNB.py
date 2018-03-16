from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from sqlalchemy import create_engine
from json import dumps
from flask.ext.jsonpify import jsonify
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,fbeta_score
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import cx_Oracle

app = Flask(__name__)
api = Api(app)


# Define parser and request args
parser = reqparse.RequestParser()
parser.add_argument('entity_id', type=int, required=True)
parser.add_argument('header_id', type=int, required=True)
parser.add_argument('vendor_id', type=str, required=False)

class Features(Resource):
    def get(self):
        args = parser.parse_args()
        entity_id = args['entity_id']
        header_id = args['header_id']
        vendor_id = args['vendor_id']
        print("entity id: " + str(args['entity_id']))
        print("header id: " + str(args['header_id']))
        query = """select feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, final_value from ml_auto_coding where entity_id="""+str(entity_id)+""" and vendor_id="""+"'"+str(vendor_id)+"'"
        print("query: " + query)
        #query = """select vendor_id as feature1, vendor_site_id as feature2, charge_account_id as final_value from inv_header h, inv_lines l where h.entity_id = l.entity_id
        #and h.header_id = l.inv_header_id and h.status in ('Imported to ERP', 'Fully Paid') and h.entity_id=2"""
        connection = cx_Oracle.connect("inspy_ap","fr3shSalt50","172.24.64.12:1521/devdb1.inspyrus.com")
        
        data = pd.read_sql(query,connection)
        
        coding = data['FINAL_VALUE']
        features = data.drop('FINAL_VALUE', axis=1)
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
        accuracy_score(y_test, predictions)
        
        testquery = """select feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10 from ml_auto_coding where entity_id="""+str(entity_id)+""" and header_id="""+"'"+str(header_id)+"'"
        print("test query: " + testquery)
        testdata = pd.read_sql(testquery,connection)
        testfeatures_final = pd.get_dummies(testdata)
        predictions = model.predict(testfeatures_final)
               
        connection.close()
        predictions2 = {}
        for i in range(predictions.size):
          predictions2[i] = output_dict[int(predictions[i])]
         
        return jsonify(predictions2)

api.add_resource(Features, '/features') 

if __name__ == '__main__':
  app.run(port=5002)

print("Running application on port 5002")
