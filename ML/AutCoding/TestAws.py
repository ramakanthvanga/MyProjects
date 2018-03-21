from flask import Flask, request, url_for
from flask_restful import Resource, Api, reqparse
from json import dumps
from flask.ext.jsonpify import jsonify

app = Flask(__name__)
api = Api(app)


@app.route('/')
def hello_world():
    return 'Welcome'

@app.route('/articles')
def api_articles():
    return 'List of ' + url_for('api_articles')

@app.route('/articles/<articleid>')
def api_article(articleid):
    return 'You are reading ' + articleid

if __name__ == '__main__':
    app.run()

print("Running application on port")
