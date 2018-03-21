from flask import Flask,request
from sqlalchemy import create_engine
from json import dumps
from flask_jsonpify import jsonify
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,fbeta_score
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import cx_Oracle

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# print a nice greeting.
@application.route('/', methods = ['POST'])
def say_hello():
    if request.is_json:
      content = request.get_json()
      train = content[0]['train']
      print("device: "+train[0]['device'])
      return 'Hello Python with JSON'
    else:
      return 'No JSON'
 
@application.route('/articles')
def api_articles():
    return 'List of article'

# run the app.
if __name__ == '__main__':
    application.run()
