from pandas import DataFrame, read_csv
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,fbeta_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

###
data=pd.read_csv('data.csv', sep=',',verbose = False,float_precision=None,dtype = str)

vendor = data['vendor_id']
shipto_id = data['shipto_id']
coding = data['charge_account_id']

coding_dict = {}
output_dict = {}
coding_index = 0
final_coding = coding

vendor_dict = {}
vendor_index = 0
final_vendor = vendor
n = coding.size
for i in range(n):
  if coding[i] in coding_dict:
    final_coding[i] = coding_dict[coding[i]]
  else:
    coding_index = coding_index + 1
    coding_dict[coding[i]] = coding_index
    output_dict[coding_index] = coding[i]
    final_coding[i] = coding_dict[coding[i]]
    
#print (final_coding)


n = vendor.size
for i in range(n):
  if vendor[i] in vendor_dict:
    final_vendor[i] = vendor_dict[vendor[i]]
  else:
    vendor_index = vendor_index + 1
    vendor_dict[vendor[i]] = vendor_index
    final_vendor[i] = vendor_dict[vendor[i]]

#print (final_vendor)

shipto_dict = {}
shipto_index = 0
final_shipto = shipto_id
n = shipto_id.size
for i in range(n):
  if shipto_id[i] in shipto_dict:
    final_shipto[i] = shipto_dict[shipto_id[i]]
  else:
    shipto_index = shipto_index + 1
    shipto_dict[shipto_id[i]] = shipto_index
    final_shipto[i] = shipto_dict[shipto_id[i]]

#print (final_shipto)

d1 = { 'vendor_id' : final_vendor, 'shipto_id' : final_shipto}
features = pd.DataFrame(data=d1)

d2 = { 'coding' : final_coding}
values = pd.DataFrame(data=d2)

features = features.as_matrix().astype(np.float)
values = values.as_matrix().astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(features, values, test_size = 0.2, random_state = 12)

#Create a Gaussian Classifier
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)

#int(predictions[0])
testdata=pd.read_csv('testdata.csv', sep=',',verbose = False,float_precision=None,dtype = str)
testvendor = testdata['vendor_id']
testshipto_id = testdata['shipto_id']
testvendor_final = testvendor
n = testvendor.size
for j in range(n):
  testvendor_final[j] = vendor_dict[testvendor[j]]

print (testvendor_final)

n = testshipto_id.size
testshipto_id_final = testshipto_id
for j in range(n):
  testshipto_id_final[j] = shipto_dict[testshipto_id[j]]

print (testshipto_id_final)
testd1 = { 'vendor_id' : testvendor_final, 'shipto_id' : testshipto_id_final}
testfeatures = pd.DataFrame(data=testd1)
testfeatures = testfeatures.as_matrix().astype(np.float)
predictions2 = model.predict(testfeatures)
for k in range(predictions2.size):
  print("Predicted charge account for index: " + str(k+1) + " is: " + output_dict[int(predictions2[k])])
