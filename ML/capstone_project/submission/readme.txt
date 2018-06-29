1. Run 'python AutoCoding.py' to get the score of the model
2. Replace line 15 with different vendor id's like 23348, 462, 959, 60650 and 5628 to see different scores
3. I altered the code so that this application can run by reading the data from a csv file.
4. In a real environment, a java application will call the python application by passing training set and prediction set. 
   Python application will return predicted charge accounts and score in the response.
5. I placed the actual code in a different file call real_application.py, this application cannot run as it expects a json requests as input. 
   Getting the json requests would be a data breach at our company so I cannot do that.